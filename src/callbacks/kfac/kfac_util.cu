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

#include "lbann/callbacks/kfac/kfac_util.hpp"

namespace lbann {
namespace callback {
namespace kfac_util {

namespace {

template <typename TensorDataType>
__global__ void kfac_add_to_diagonal_kernel(
    TensorDataType * __restrict__ A,
    const size_t height,
    const TensorDataType value,
    const TensorDataType value_bn_err,
    const bool is_bn) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height) {
    A[gid+gid*height] += (is_bn && gid >= height/2 ? value_bn_err : value);
  }
}

template <typename TensorDataType>
__global__ void kfac_fill_upper_tri_kernel(
    TensorDataType * __restrict__ A,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t row = gid%height, col = gid/height;
  if(row < height && col < height && row < col) {
    A[row+col*height] += A[col+row*height];
  }
}

template <typename TensorDataType>
__global__ void kfac_update_kronecker_average_kernel(
    TensorDataType * __restrict__ Aave,
    const TensorDataType * __restrict__ A,
    const size_t count,
    const double decay) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < count) {
    Aave[gid] = Aave[gid]*decay + A[gid]*(1.0-decay);
  }
}

#ifdef LBANN_HAS_HALF
template <>
__global__ void kfac_update_kronecker_average_kernel<__half>(
    __half * __restrict__ Aave,
    const __half * __restrict__ A,
    const size_t count,
    const double decay) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < count) {
    Aave[gid] = (float) Aave[gid]*decay + (float) A[gid]*(1.0-decay);
  }
}
#endif // LBANN_HAS_HALF

template <typename TensorDataType>
__global__ void kfac_identity_kernel(
    TensorDataType * __restrict__ A,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height*height) {
    const size_t row = gid%height;
    const size_t col = gid/height;
    A[gid] = (row == col ? 1.0 : 0.0);
  }
}

template <typename TensorDataType>
__global__ void kfac_pack_lower_tri_kernel(
    TensorDataType * __restrict__ L,
    const TensorDataType * __restrict__ A,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height*height) {
    const size_t row = gid%height;
    const size_t col = gid/height;
    if(row >= col) {
      L[row+(2*height-(col-1))*col/2-col] = A[gid];
    }
  }
}

template <typename TensorDataType>
__global__ void kfac_unpack_lower_tri_kernel(
    TensorDataType * __restrict__ A,
    const TensorDataType * __restrict__ L,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height*height) {
    const size_t row = gid%height;
    const size_t col = gid/height;
    if(row >= col) {
      A[gid] = A[col+row*height] = L[row+(2*height-(col-1))*col/2-col];
    }
  }
}

} // namespace

template <typename TensorDataType>
void add_to_diagonal(
    TensorDataType * __restrict__ A,
    const size_t height,
    const TensorDataType damping,
    const TensorDataType damping_bn_err,
    const bool is_bn,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t grid_size = (height + block_size - 1) / block_size;
  kfac_add_to_diagonal_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      A, height, damping,
      damping_bn_err, is_bn);
}

template <typename TensorDataType>
void fill_upper_tri(
    TensorDataType * __restrict__ A,
    const size_t height,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  // TODO: Launch N^2/2 threads instead of N^2
  const size_t grid_size = (height*height + block_size - 1) / block_size;
  kfac_fill_upper_tri_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      A, height);
}

template <typename TensorDataType>
void update_kronecker_average(
    TensorDataType * __restrict__ Aave,
    const TensorDataType * __restrict__ A,
    const size_t count, const double decay,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t grid_size = (count + block_size - 1) / block_size;
  kfac_update_kronecker_average_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      Aave, A, count, decay);
}

template <typename TensorDataType>
void identity(
    TensorDataType * __restrict__ A,
    const size_t height,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = height*height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  kfac_identity_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          A, height);
}

template <typename TensorDataType>
void pack_lower_tri(
    TensorDataType * __restrict__ L,
    const TensorDataType * __restrict__ A,
    const size_t height,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = height*height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  // TODO: Launch N^2/2 threads instead of N^2
  kfac_pack_lower_tri_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          L, A, height);
}

template <typename TensorDataType>
void unpack_lower_tri(
    TensorDataType * __restrict__ A,
    const TensorDataType * __restrict__ L,
    const size_t height,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = height*height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  // TODO: Launch N^2/2 threads instead of N^2
  kfac_unpack_lower_tri_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          A, L, height);
}

#define PROTO(T)                                \
  template void add_to_diagonal<T>(             \
      T* __restrict__ A,                        \
      const size_t height,                      \
      const T value,                            \
      const T value_bn_err,                     \
      const bool is_bn,                         \
      const cudaStream_t& stream);              \
  template void fill_upper_tri<T>(              \
      T * __restrict__ A,                       \
      const size_t height,                      \
      const cudaStream_t& stream);              \
  template void update_kronecker_average<T>(    \
      T * __restrict__ Aave,                    \
      const T * __restrict__ A,                 \
      const size_t count,                       \
      const double decay,                       \
      const cudaStream_t& stream);              \
  template void identity<T>(                    \
      T * __restrict__ A,                       \
      const size_t height,                      \
      const cudaStream_t& stream);              \
  template void pack_lower_tri<T>(              \
      T * __restrict__ L,                       \
      const T * __restrict__ A,                 \
      const size_t height,                      \
      const cudaStream_t& stream);              \
  template void unpack_lower_tri<T>(            \
      T * __restrict__ A,                       \
      const T * __restrict__ L,                 \
      const size_t height,                      \
      const cudaStream_t& stream);

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace kfac_util
} // namespace callback
} // namespace lbann
