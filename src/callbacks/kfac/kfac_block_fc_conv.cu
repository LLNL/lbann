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

#include "lbann/callbacks/kfac/kfac_block_fc_conv.hpp"


namespace lbann {
namespace callback {

namespace {

template <typename TensorDataType>
__global__ void kfac_get_diagonal_kernel(
    TensorDataType * __restrict__ diag,
    const TensorDataType * __restrict__ A,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height)
    diag[gid] = A[gid+gid*height];
}

template <typename TensorDataType>
__global__ void kfac_conv_transpose_kernel(
    const TensorDataType * __restrict__ A,
    TensorDataType * __restrict__ Acol,
    const size_t mini_batch_size, const size_t num_channels,
    const size_t spatial_prod, const size_t num_elems) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < num_elems) {
    const auto i_spatial = gid%spatial_prod;
    const auto i_c = (gid/spatial_prod)%num_channels;
    const auto i_n = (gid/spatial_prod/num_channels);
    Acol[i_c+i_spatial*num_channels+i_n*num_channels*spatial_prod] = A[gid];
  }
}

} // namespace

template <typename TensorDataType>
void kfac_block_fc_conv::get_diagonal(
    TensorDataType * __restrict__ diag,
    const TensorDataType * __restrict__ A,
    const size_t height,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  kfac_get_diagonal_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          diag, A, height);
}

template <typename TensorDataType>
void kfac_block_fc_conv::conv_transpose(
    const TensorDataType * __restrict__ activations,
    TensorDataType * __restrict__ act_columns,
    const size_t mini_batch_size, const size_t num_channels,
    const size_t spatial_prod,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_elems = mini_batch_size*num_channels*spatial_prod;
  const size_t grid_size = (num_elems + block_size - 1) / block_size;
  kfac_conv_transpose_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      activations, act_columns, mini_batch_size, num_channels, spatial_prod,
      num_elems);
}

#define PROTO(T)                                        \
  template void kfac_block_fc_conv::get_diagonal<T>(                  \
      T * __restrict__ diag,                            \
      const T * __restrict__ A,                         \
      const size_t height,                              \
      const cudaStream_t& stream);                      \
  template void kfac_block_fc_conv::conv_transpose<T>(  \
      const T * __restrict__ activations,               \
      T * __restrict__ act_columns,                     \
      const size_t batch_size,                          \
      const size_t num_channels,                        \
      const size_t spatial_prod,                        \
      const cudaStream_t& stream);

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace callback
} // namespace lbann
