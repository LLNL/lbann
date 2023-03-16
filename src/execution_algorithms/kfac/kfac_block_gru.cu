////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/execution_algorithms/kfac/kfac_block_gru.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
__device__ __forceinline__ TensorDataType sigmoid(const TensorDataType& x)
{
  return (tanh(x * 0.5) + 1.0) * 0.5;
}
template <typename TensorDataType>
__device__ __forceinline__ TensorDataType sigmoid_deriv(const TensorDataType& x)
{
  const TensorDataType t = sigmoid(x);
  return t * (1.0 - t);
}
template <typename TensorDataType>
__device__ __forceinline__ TensorDataType sigmoid_inv(const TensorDataType& x)
{
  return log(x / (1.0 - x));
}
template <typename TensorDataType>
__device__ __forceinline__ TensorDataType tanh_deriv(const TensorDataType& x)
{
  const TensorDataType t = tanh(x);
  return 1.0 - t * t;
}
template <typename TensorDataType>
__device__ __forceinline__ TensorDataType tanh_inv(const TensorDataType& x)
{
  return 0.5 * log((1.0 + x) / (1.0 - x));
}

template <typename TensorDataType>
__global__ void unpack_reserve_space_kernel(
  const TensorDataType* __restrict__ reserve_space_fwd,
  TensorDataType* __restrict__ r, // (hidden_size*local_batch_size) x seq_length
  TensorDataType* __restrict__ i, // (hidden_size*local_batch_size) x seq_length
  const size_t hidden_size,
  const size_t seq_length,
  const size_t local_batch_size)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < hidden_size * seq_length * local_batch_size) {
    // TODO: We assume TensorDataType for the reserve space
    // r_0^(0) i_0^(0) *^{4*hidden_size} r_0^(1) i_0^(1) ...
    // where r_t^(n) is r of n-th sample at timestep t
    const size_t h = gid % hidden_size;
    const size_t n = (gid / hidden_size) % local_batch_size;
    const size_t t = gid / hidden_size / local_batch_size;
    const size_t r_offset =
      h + n * (hidden_size * 6) + t * (hidden_size * 6 * local_batch_size);
    r[gid] = reserve_space_fwd[r_offset];
    i[gid] = reserve_space_fwd[r_offset + hidden_size];
  }
}

template <typename TensorDataType>
__global__ void get_g_kernel(
  const TensorDataType* __restrict__ h,   // hidden_size*seq_length x
                                          // local_batch_size
  const TensorDataType* __restrict__ h0,  // hidden_size x local_batch_size
  const TensorDataType* __restrict__ dh,  // hidden_size*seq_length x
                                          // local_batch_size
  const TensorDataType* __restrict__ hfc, // hidden_size x
                                          // local_batch_size*seq_length
  const TensorDataType* __restrict__ r,   // hidden_size*local_batch_size x
                                          // seq_length
  const TensorDataType* __restrict__ i,   // hidden_size*local_batch_size x
                                          // seq_length
  TensorDataType* __restrict__ g_Rr,      // hidden_size x
                                          // local_batch_size*seq_length
  TensorDataType* __restrict__ g_Ri,      // hidden_size x
                                          // local_batch_size*seq_length
  TensorDataType* __restrict__ g_Rh,      // hidden_size x
                                          // local_batch_size*seq_length
  TensorDataType* __restrict__ g_Wr,      // hidden_size x
                                          // local_batch_size*seq_length
  TensorDataType* __restrict__ g_Wi,      // hidden_size x
                                          // local_batch_size*seq_length
  TensorDataType* __restrict__ g_Wh,      // hidden_size x
                                          // local_batch_size*seq_length
  const size_t hidden_size,
  const size_t seq_length,
  const size_t local_batch_size)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < hidden_size * seq_length * local_batch_size) {
    const size_t i_hidden = gid % hidden_size;
    const size_t i_seq = (gid / hidden_size) % seq_length;
    const size_t i_batch = gid / hidden_size / seq_length;
    const size_t i_hsl =
      i_hidden + i_seq * hidden_size + i_batch * hidden_size * seq_length;
    const size_t i_hl = i_hidden + i_batch * hidden_size;
    const size_t i_hls =
      i_hidden + i_batch * hidden_size + i_seq * hidden_size * local_batch_size;

    // dh/dh' = diag(1-i_t)
    // dh/dr = diag{(dh/dh')*tanh'(tanh^-1(h'_t))*hfc}
    // dh/di = diag{h_{t-1} - h'_t}
    // dr/dg_Rr = diag{sigmoid'(sigmoid^-1(r_t))}
    // g_Rr = Rr h_{t-1}
    const TensorDataType r_val = r[i_hls];
    const TensorDataType i_val = i[i_hls];
    const TensorDataType dh_val = dh[i_hsl];
    const TensorDataType hprev =
      (i_seq == 0 ? h0[i_hl] : h[i_hsl - hidden_size]);
    const TensorDataType hd = (h[i_hsl] - i_val * hprev) / (1.0 - i_val);
    const TensorDataType dhdhd = 1.0 - i_val;
    const TensorDataType dhdr = dhdhd * tanh_deriv(tanh_inv(hd)) * hfc[i_hls];
    const TensorDataType dhdi = hprev - hd;
    const TensorDataType drdg_Rr = sigmoid_deriv(sigmoid_inv(r_val));
    const TensorDataType didg_Ri = sigmoid_deriv(sigmoid_inv(i_val));
    const TensorDataType dhddg_Wh = tanh_deriv(tanh_inv(hd));
    const TensorDataType dhddg_Rh = dhddg_Wh * r_val;

    g_Wr[i_hls] = g_Rr[i_hls] = dh_val * dhdr * drdg_Rr;
    g_Wi[i_hls] = g_Ri[i_hls] = dh_val * dhdi * didg_Ri;
    g_Wh[i_hls] = dh_val * dhdhd * dhddg_Wh;
    g_Rh[i_hls] = dh_val * dhdhd * dhddg_Rh;
  }
}

} // namespace

template <>
void kfac_gru_util::get_g(const El::Matrix<DataType, El::Device::GPU>& h,
                          const El::Matrix<DataType, El::Device::GPU>& h0,
                          const El::Matrix<DataType, El::Device::GPU>& dh,
                          const El::Matrix<DataType, El::Device::GPU>& hfc,
                          const El::Matrix<DataType, El::Device::GPU>& r,
                          const El::Matrix<DataType, El::Device::GPU>& i,
                          El::Matrix<DataType, El::Device::GPU>& g_Rr,
                          El::Matrix<DataType, El::Device::GPU>& g_Ri,
                          El::Matrix<DataType, El::Device::GPU>& g_Rh,
                          El::Matrix<DataType, El::Device::GPU>& g_Wr,
                          El::Matrix<DataType, El::Device::GPU>& g_Wi,
                          El::Matrix<DataType, El::Device::GPU>& g_Wh,
                          const size_t hidden_size,
                          const size_t seq_length,
                          const size_t local_batch_size,
                          const El::SyncInfo<El::Device::GPU>& sync_info)
{
  constexpr size_t block_size = 256;
  const size_t count = hidden_size * seq_length * local_batch_size;
  const size_t grid_size = (count + block_size - 1) / block_size;
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(get_g_kernel<DataType>,
                                grid_size,
                                block_size,
                                0,
                                sync_info,
                                h.LockedBuffer(),
                                h0.LockedBuffer(),
                                dh.LockedBuffer(),
                                hfc.LockedBuffer(),
                                r.LockedBuffer(),
                                i.LockedBuffer(),
                                g_Rr.Buffer(),
                                g_Ri.Buffer(),
                                g_Rh.Buffer(),
                                g_Wr.Buffer(),
                                g_Wi.Buffer(),
                                g_Wh.Buffer(),
                                hidden_size,
                                seq_length,
                                local_batch_size);
  }
}

template <>
void kfac_gru_util::unpack_reserve_space(
  const DataType* reserve_space_fwd,
  El::Matrix<DataType, El::Device::GPU>& r,
  El::Matrix<DataType, El::Device::GPU>& i,
  const size_t hidden_size,
  const size_t seq_length,
  const size_t local_batch_size,
  const El::SyncInfo<El::Device::GPU>& sync_info)
{
  const size_t count = hidden_size * seq_length * local_batch_size;
  const dnn_lib::dnnMathType_t math_type =
    dnn_lib::get_default_convolution_math_type();
  size_t offset = 0;
  if (math_type == dnn_lib::DNN_TENSOR_OP_MATH_ALLOW_CONVERSION) {
    const size_t align_base = 4;
    offset = ((size_t)((0.5 * hidden_size * seq_length * local_batch_size +
                        align_base - 1) /
                       align_base)) *
             align_base;
  }
  else if (math_type != dnn_lib::DNN_DEFAULT_MATH)
    LBANN_ERROR("Unsupported dnn lib math type.");

  constexpr size_t block_size = 256;
  const size_t grid_size = (count + block_size - 1) / block_size;
  if (grid_size > 0) {
    hydrogen::gpu::LaunchKernel(unpack_reserve_space_kernel<DataType>,
                                grid_size,
                                block_size,
                                0,
                                sync_info,
                                reserve_space_fwd + offset * sizeof(DataType),
                                r.Buffer(),
                                i.Buffer(),
                                hidden_size,
                                seq_length,
                                local_batch_size);
  }
}

} // namespace lbann
