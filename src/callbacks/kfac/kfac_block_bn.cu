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

#include "lbann/callbacks/kfac/kfac_block_bn.hpp"


namespace lbann {
namespace callback {

namespace {

template <typename TensorDataType>
__global__ void kfac_compute_bn_factor_kernel(
    const TensorDataType * __restrict__ activations,
    const TensorDataType * __restrict__ errors,
    const TensorDataType * __restrict__ scales,
    const TensorDataType * __restrict__ biases,
    TensorDataType * __restrict__ factor,
    const size_t batch_size,
    const size_t num_channels,
    const size_t spatial_prod,
    const size_t num_threads) { // = batch_size*num_channels
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < num_threads) {
    const size_t i_c = gid%num_channels;
    const size_t i_n = gid/num_channels;

    const auto scale = scales[i_c];
    const auto bias = biases[i_c];

    TensorDataType sum_ea = 0.0, sum_e = 0.0;
    // TODO: This loop would be slow in large (3D) CNNs
    for(size_t i_s = 0; i_s < spatial_prod; i_s++) {
      const auto i_act = i_s+gid*spatial_prod;
      const auto error = errors[i_act];
      const auto act = (activations[i_act]-bias)/scale;
      sum_ea += error * act;
      sum_e += error;
    }
    factor[i_c+i_n*num_channels*2] = sum_ea;
    factor[i_c+num_channels+i_n*num_channels*2] = sum_e;
  }
}

template <typename TensorDataType>
__global__ void kfac_compute_bn_factor_data2col_kernel(
    const TensorDataType * __restrict__ activations,
    const TensorDataType * __restrict__ errors,
    const TensorDataType * __restrict__ scales,
    const TensorDataType * __restrict__ biases,
    TensorDataType * __restrict__ cols,
    const size_t batch_size,
    const size_t num_channels,
    const size_t spatial_prod,
    const size_t num_threads) { // = batch_size*num_channels*spatial_prod
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < num_threads) {
    const size_t i_c = gid%num_channels;
    const size_t i_n = (gid/num_channels)%batch_size;
    const size_t i_s = gid/num_channels/batch_size;
    const auto scale = scales[i_c];
    const auto bias = biases[i_c];
    const auto i_act = i_s+i_c*spatial_prod+i_n*spatial_prod*num_channels;
    const auto error = errors[i_act];
    const auto act = (activations[i_act]-bias)/scale;
    const auto i_out = i_c+i_n*num_channels*2 + i_s*(num_channels*2*batch_size);
    cols[i_out] = error*act;
    cols[i_out+num_channels] = error;
  }
}

} // namespace

namespace kfac_bn_util {

template <typename TensorDataType>
void compute_bn_factor(
    const TensorDataType * __restrict__ activations,
    const TensorDataType * __restrict__ errors,
    const TensorDataType * __restrict__ scales,
    const TensorDataType * __restrict__ biases,
    TensorDataType * __restrict__ factor,
    const size_t batch_size,
    const size_t num_channels,
    const size_t spatial_prod,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = batch_size * num_channels;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  kfac_compute_bn_factor_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          activations, errors, scales, biases,
          factor,
          batch_size, num_channels, spatial_prod,
          num_threads);
}

template <typename TensorDataType>
void compute_bn_factor_data2col(
    const TensorDataType * __restrict__ activations,
    const TensorDataType * __restrict__ errors,
    const TensorDataType * __restrict__ scales,
    const TensorDataType * __restrict__ biases,
    TensorDataType * __restrict__ cols,
    const size_t batch_size,
    const size_t num_channels,
    const size_t spatial_prod,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = batch_size * num_channels * spatial_prod;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  kfac_compute_bn_factor_data2col_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          activations, errors, scales, biases,
          cols,
          batch_size, num_channels, spatial_prod,
          num_threads);
}

#define PROTO(T)                                \
  template void compute_bn_factor<T>(           \
      const T * __restrict__ activations,       \
      const T * __restrict__ errors,            \
      const T * __restrict__ scales,            \
      const T * __restrict__ biases,            \
      T * __restrict__ factor,                  \
      const size_t batch_size,                  \
      const size_t num_channels,                \
      const size_t spatial_prod,                \
      const cudaStream_t& stream);              \
  template void compute_bn_factor_data2col<T>(  \
      const T * __restrict__ activations,       \
      const T * __restrict__ errors,            \
      const T * __restrict__ scales,            \
      const T * __restrict__ biases,            \
      T * __restrict__ cols,                    \
      const size_t batch_size,                  \
      const size_t num_channels,                \
      const size_t spatial_prod,                \
      const cudaStream_t& stream);

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace kfac_bn_util

} // namespace callback
} // namespace lbann
