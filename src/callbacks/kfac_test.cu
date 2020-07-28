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

#include "lbann/callbacks/kfac_test.hpp"


namespace lbann {
namespace callback {

namespace {

template <typename TensorDataType>
__global__ void kfac_test_add_to_diagonal_kernel(
    TensorDataType * __restrict__ A,
    const size_t height,
    const TensorDataType damping) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height) {
    A[gid+gid*height] += damping;
  }
}

template <typename TensorDataType>
__global__ void kfac_test_fill_upper_tri_kernel(
    TensorDataType * __restrict__ A,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  // REVIEW
  const size_t row = gid%height, col = gid/height;
  if(row < height && col < height && row < col) {
    A[row+col*height] += A[col+row*height];
  }
}

template <typename TensorDataType>
__global__ void kfac_test_update_kronecker_average_kernel(
    TensorDataType * __restrict__ Aave,
    const TensorDataType * __restrict__ A,
    const size_t count, const DataType decay) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < count) {
    Aave[gid] = Aave[gid]*decay + A[gid]*(DataType(1.0)-decay);
  }
}

} // namespace

// TODO: static function
template <typename TensorDataType>
void kfac_test_add_to_diagonal(
    TensorDataType * __restrict__ A,
    const size_t height,
    const TensorDataType damping) {
  constexpr size_t block_size = 256;
  const size_t grid_size = (height + block_size - 1) / block_size;
  auto&& stream = El::GPUManager::Stream();
  kfac_test_add_to_diagonal_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      A, height, damping);
}

// TODO: static function
template <typename TensorDataType>
void kfac_test_fill_upper_tri(
    TensorDataType * __restrict__ A,
    const size_t height) {
  constexpr size_t block_size = 256;
  // OPTIMIZE: Launch N^2/2 threads instead of N^2
  const size_t grid_size = (height*height + block_size - 1) / block_size;
  auto&& stream = El::GPUManager::Stream();
  kfac_test_fill_upper_tri_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      A, height);
}

// TODO: static function
template <typename TensorDataType>
void kfac_test_update_kronecker_average(
    TensorDataType * __restrict__ Aave,
    const TensorDataType * __restrict__ A,
    const size_t count, const DataType decay) {
  constexpr size_t block_size = 256;
  const size_t grid_size = (count + block_size - 1) / block_size;
  auto&& stream = El::GPUManager::Stream();
  kfac_test_update_kronecker_average_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      Aave, A, count, decay);
}

#define PROTO(T)                                        \
  template void kfac_test_add_to_diagonal<T>(           \
      T* __restrict__ A,                                \
      const size_t height,                              \
      const T damping);                                 \
  template void kfac_test_fill_upper_tri<T>(            \
      T * __restrict__ A,                               \
      const size_t height);                             \
  template void kfac_test_update_kronecker_average<T>(  \
      T * __restrict__ Aave,                            \
      const T * __restrict__ A,                         \
      const size_t count, const DataType decay)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace callback
} // namespace lbann
