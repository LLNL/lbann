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
    const size_t count) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < count) {
    // dh/dr = diag{(1-i_t)*tanh'(tanh^-1(h'_t))*hfc}
    // dh/di = diag{h_{t-1} - h'_t}
    // dr/dg_Rr = diag{sigmoid'(sigmoid^-1(r_t))}
    // g_Rr = Rr h_{t-1}
    const TensorDataType hd = tanh_inv(h[gid]);
    const TensorDataType dhdr = (1.0-i[gid]) * tanh_deriv(hd) * hfc[gid];
    const TensorDataType dhdi = hprev[gid] - hd;
    const TensorDataType drdg_Rr = sigmoid_deriv(sigmoid_inv(r[gid]));
    const TensorDataType didg_Ri = sigmoid_deriv(sigmoid_inv(i[gid]));
    g_Rr[gid] = dh[gid] * dhdr * drdg_Rr;
    g_Ri[gid] = dh[gid] * dhdi * didg_Ri;
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
          g_Rr.Buffer(),
          g_Ri.Buffer(),
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
  constexpr size_t block_size = 256;
  const size_t grid_size = (count + block_size - 1) / block_size;
  unpack_reserve_space_kernel<DataType>
      <<<grid_size, block_size, 0, sync_info.Stream()>>>(
          reserve_space_fwd,
          r.Buffer(),
          i.Buffer(),
          hidden_size,
          seq_length,
          local_batch_size);
}

} // namespace callback
} // namespace lbann
