////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/kfac/kfac_block_gru.hpp"

namespace lbann {
namespace callback {

namespace {

template <typename TensorDataType>
__device__ __forceinline__ TensorDataType sigmoid(const TensorDataType& x) {
  return (tanh(x*0.5)+1.0)*0.5;
}
template <typename TensorDataType>
__device__ __forceinline__ TensorDataType sigmoid_deriv(const TensorDataType& x) {
  const TensorDataType t = sigmoid(x);
  return t*(1.0-t);
}
template <typename TensorDataType>
__device__ __forceinline__ TensorDataType sigmoid_inv(const TensorDataType& x) {
  return log(x/(1.0-x));
}
template <typename TensorDataType>
__device__ __forceinline__ TensorDataType tanh_deriv(const TensorDataType& x) {
  const TensorDataType t = tanh(x);
  return 1.0-t*t;
}
template <typename TensorDataType>
__device__ __forceinline__ TensorDataType tanh_inv(const TensorDataType& x) {
  return 0.5*log((1.0+x)/(1.0-x));
}

template <typename TensorDataType>
__global__ void unpack_reserve_space_kernel(
    const TensorDataType * __restrict__ reserve_space_fwd,
    TensorDataType * __restrict__ r, // (hidden_size*local_batch_size) x seq_length
    TensorDataType * __restrict__ i, // (hidden_size*local_batch_size) x seq_length
    const size_t hidden_size,
    const size_t seq_length,
    const size_t local_batch_size) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < hidden_size*seq_length*local_batch_size) {
    // TODO: We assume TensorDataType for the reserve space
    // r_0^(0) i_0^(0) *^{4*hidden_size} r_0^(1) i_0^(1) ...
    // where r_t^(n) is r of n-th sample at timestep t
    const size_t h = gid%hidden_size;
    const size_t n = (gid/hidden_size)%local_batch_size;
    const size_t t = gid/hidden_size/local_batch_size;
    const size_t r_offset = h+n*(hidden_size*6)
        +t*(hidden_size*6*local_batch_size);
    r[gid] = reserve_space_fwd[r_offset];
    i[gid] = reserve_space_fwd[r_offset+hidden_size];
  }
}

template <typename TensorDataType>
__global__ void get_g_kernel(
    const TensorDataType * __restrict__ h,
    const TensorDataType * __restrict__ hprev,
    const TensorDataType * __restrict__ dh,
    const TensorDataType * __restrict__ hfc,
    const TensorDataType * __restrict__ r,
    const TensorDataType * __restrict__ i,
    TensorDataType * __restrict__ g_Rr,
    TensorDataType * __restrict__ g_Ri,
    TensorDataType * __restrict__ g_Rh,
    TensorDataType * __restrict__ g_Wr,
    TensorDataType * __restrict__ g_Wi,
    TensorDataType * __restrict__ g_Wh,
    const size_t count) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < count) {
    // dh/dh' = diag(1-i_t)
    // dh/dr = diag{(dh/dh')*tanh'(tanh^-1(h'_t))*hfc}
    // dh/di = diag{h_{t-1} - h'_t}
    // dr/dg_Rr = diag{sigmoid'(sigmoid^-1(r_t))}
    // g_Rr = Rr h_{t-1}
    const TensorDataType hd = (h[gid]-i[gid]*hprev[gid])/(1.0-i[gid]);
    const TensorDataType dhdhd = 1.0-i[gid];
    const TensorDataType dhdr = dhdhd * tanh_deriv(tanh_inv(hd)) * hfc[gid];
    const TensorDataType dhdi = hprev[gid] - hd;
    const TensorDataType drdg_Rr = sigmoid_deriv(sigmoid_inv(r[gid]));
    const TensorDataType didg_Ri = sigmoid_deriv(sigmoid_inv(i[gid]));
    const TensorDataType dhddg_Wh = tanh_deriv(tanh_inv(hd));
    const TensorDataType dhddg_Rh = dhddg_Wh * r[gid];
    g_Wr[gid] = g_Rr[gid] = dh[gid] * dhdr * drdg_Rr;
    g_Wi[gid] = g_Ri[gid] = dh[gid] * dhdi * didg_Ri;
    g_Wh[gid] = dh[gid] * dhdhd * dhddg_Wh;
    g_Rh[gid] = dh[gid] * dhdhd * dhddg_Rh;
  }
}

}

template <>
void kfac_gru_util::get_g(
    const El::Matrix<DataType, El::Device::GPU>& h,
    const El::Matrix<DataType, El::Device::GPU>& hprev,
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
    const size_t count,
    const El::SyncInfo<El::Device::GPU>& sync_info) {
  constexpr size_t block_size = 256;
  const size_t grid_size = (count + block_size - 1) / block_size;
  get_g_kernel<DataType>
      <<<grid_size, block_size, 0, sync_info.Stream()>>>(
          h.LockedBuffer(),
          hprev.LockedBuffer(),
          dh.LockedBuffer(),
          hfc.LockedBuffer(),
          r.LockedBuffer(),
          i.LockedBuffer(),
          g_Rr.Buffer(), g_Ri.Buffer(), g_Rh.Buffer(),
          g_Wr.Buffer(), g_Wi.Buffer(), g_Wh.Buffer(),
          count);
}

template <>
void kfac_gru_util::unpack_reserve_space(
    const DataType* reserve_space_fwd,
    El::Matrix<DataType, El::Device::GPU>& r,
    El::Matrix<DataType, El::Device::GPU>& i,
    const size_t hidden_size,
    const size_t seq_length,
    const size_t local_batch_size,
    const El::SyncInfo<El::Device::GPU>& sync_info) {
  const size_t count = hidden_size*seq_length*local_batch_size;
  const cudnnMathType_t math_type = dnn_lib::get_default_convolution_math_type();
  size_t offset = 0;
  if(math_type == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) {
    const size_t align_base = 4;
    offset = ((size_t) ((0.5*hidden_size*seq_length*local_batch_size+align_base-1)/align_base))*align_base;
  } else if(math_type != CUDNN_DEFAULT_MATH)
    LBANN_ERROR("Unsupported cuDNN math type.");

  constexpr size_t block_size = 256;
  const size_t grid_size = (count + block_size - 1) / block_size;
  unpack_reserve_space_kernel<DataType>
      <<<grid_size, block_size, 0, sync_info.Stream()>>>(
          reserve_space_fwd+offset*sizeof(DataType),
          r.Buffer(),
          i.Buffer(),
          hidden_size,
          seq_length,
          local_batch_size);
}

} // namespace callback
} // namespace lbann
