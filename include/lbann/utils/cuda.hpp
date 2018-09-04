////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_UTILS_CUDA_HPP
#define LBANN_UTILS_CUDA_HPP

#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_GPU

#include <cuda.h>
#include <thrust/memory.h>
#include <thrust/detail/allocator/tagged_allocator.h>
#ifdef __CUDACC__
#include <cuda_fp16.hpp>
#endif // __CUDACC__

// -------------------------------------------------------------
// Error utility macros
// -------------------------------------------------------------
#define LBANN_CUDA_SYNC(async)                                  \
  do {                                                          \
    /* Synchronize GPU and check for errors. */                 \
    cudaError_t status_CUDA_SYNC = cudaDeviceSynchronize();     \
    if (status_CUDA_SYNC == cudaSuccess)                        \
      status_CUDA_SYNC = cudaGetLastError();                    \
    if (status_CUDA_SYNC != cudaSuccess) {                      \
      cudaDeviceReset();                                        \
      std::stringstream err_CUDA_SYNC;                          \
      if (async) { err_CUDA_SYNC << "Asynchronous "; }          \
      err_CUDA_SYNC << "CUDA error ("                           \
                    << cudaGetErrorString(status_CUDA_SYNC)     \
                    << ")";                                     \
      LBANN_ERROR(err_CUDA_SYNC.str());                         \
    }                                                           \
  } while (0)
#define FORCE_CHECK_CUDA(cuda_call)                             \
  do {                                                          \
    /* Call CUDA API routine, synchronizing before and */       \
    /* after to check for errors. */                            \
    LBANN_CUDA_SYNC(true);                                      \
    cudaError_t status_CHECK_CUDA = (cuda_call);                \
    if (status_CHECK_CUDA != cudaSuccess) {                     \
      LBANN_ERROR(std::string("CUDA error (")                   \
                  + cudaGetErrorString(status_CHECK_CUDA)       \
                  + std::string(")"));                          \
    }                                                           \
    LBANN_CUDA_SYNC(false);                                     \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUDA(cuda_call) FORCE_CHECK_CUDA(cuda_call);
#else
#define CHECK_CUDA(cuda_call) (cuda_call)
#endif // #ifdef LBANN_DEBUG

namespace lbann {
namespace cuda {

// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------
#ifdef __CUDACC__

// Atomic add functions
#if __CUDA_ARCH__ >= 530
__device__ __inline__ __half atomic_add(__half* address, __half val) {
#if 0 // TODO: replace this once Nvidia implements atomicAdd for __half
  return atomicAdd(address, val);
#else
  unsigned int* address_as_uint = (unsigned int*) address;
  unsigned int old = *address_as_uint;
  __half* old_as_half = (__half*) &old;
  unsigned int assumed;
  unsigned int updated;
  __half* updated_as_half = (__half*) &updated;
  do {
    assumed = old;
    updated = old;
    *updated_as_half += val;
    old = atomicCAS(address_as_uint, assumed, updated);
  } while (assumed != old);
  return *old_as_half;
#endif // 0
}
#endif // __CUDA_ARCH__ >= 530
__device__ __inline__ float atomic_add(float* address, float val) {
  return atomicAdd(address, val);
}
__device__ __inline__ double atomic_add(double* address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif // __CUDA_ARCH__ < 600
}

// Min and max
__device__ __inline__ int min(int x, int y) { return x <= y ? x : y; }
__device__ __inline__ El::Int min(El::Int x, El::Int y) { return x <= y ? x : y; }
__device__ __inline__ float min(float x, float y) { return fminf(x, y); }
__device__ __inline__ double min(double x, double y) { return fmin(x, y); }
__device__ __inline__ int max(int x, int y) { return x >= y ? x : y; }
__device__ __inline__ El::Int max(El::Int x, El::Int y) { return x >= y ? x : y; }
__device__ __inline__ float max(float x, float y) { return fmaxf(x, y); }
__device__ __inline__ double max(double x, double y) { return fmax(x, y); }
  
#endif // __CUDACC__
  
// -------------------------------------------------------------
// Helper functions for entrywise operations
// -------------------------------------------------------------
#ifdef __CUDACC__

/** CUDA kernel to apply an entry-wise unary operator. */
template <typename UnaryOperator>
__global__
void entrywise_unary_operator_kernel(El::Int height, El::Int width,
                                     const DataType* __restrict__ input,
                                     El::Int input_ldim,
                                     DataType* __restrict__ output,
                                     El::Int output_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  UnaryOperator op;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_ldim];
    auto& y = output[row + col * output_ldim];
    y = op(x);
  }
}

/** CUDA kernel to apply an entry-wise binary operator. */
template <typename BinaryOperator>
__global__
void entrywise_binary_operator_kernel(El::Int height, El::Int width,
                                     const DataType* __restrict__ input1,
                                     El::Int input1_ldim,
                                     const DataType* __restrict__ input2,
                                     El::Int input2_ldim,
                                     DataType* __restrict__ output,
                                     El::Int output_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  BinaryOperator op;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x1 = input1[row + col * input1_ldim];
    const auto& x2 = input2[row + col * input2_ldim];
    auto& y = output[row + col * output_ldim];
    y = op(x1, x2);
  }
}

/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <typename UnaryOperator>
void apply_entrywise_unary_operator(const AbsMat& input,
                                    AbsMat& output) {

  // Check that input and output are valid
  std::stringstream err;
  if (input.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("input is not on GPU");
  } else if (output.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("output is not on GPU");
  } else if (input.Height() != output.Height()
             || input.Width() != output.Width()) {
    err << "input matrix dimensions "
        << "(" << input.Height() << " x " << input.Width() << ")"
        << "don't match output matrix dimensions "
        << "(" << output.Height() << " x " << output.Width() << ")";
    LBANN_ERROR(err.str());
  }

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = input.Height();
  const El::Int width = input.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    entrywise_unary_operator_kernel<UnaryOperator>
      <<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
        height, width, input.LockedBuffer(), input.LDim(),
        output.Buffer(), output.LDim());
  }
  
}

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <typename BinaryOperator>
void apply_entrywise_binary_operator(const AbsMat& input1,
                                     const AbsMat& input2,
                                     AbsMat& output) {

  // Check that input and output are valid
  std::stringstream err;
  if (input1.GetDevice() != El::Device::GPU
      || input2.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("input is not on GPU");
  } else if (output.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("output is not on GPU");
  } else if (input1.Height() != input2.Height()
             || input1.Width() != input2.Width()
             || input1.Height() != output.Height()
             || input1.Width() != output.Width()) {
    err << "input matrix dimensions "
        << "(" << input1.Height() << " x " << input1.Width() << ", "
        << input2.Height() << " x " << input2.Width() << ")"
        << "don't match output matrix dimensions "
        << "(" << output.Height() << " x " << output.Width() << ")";
    LBANN_ERROR(err.str());
  }

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = input1.Height();
  const El::Int width = input1.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    entrywise_binary_operator_kernel<BinaryOperator>
      <<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
        height, width,
        input1.LockedBuffer(), input1.LDim(),
        input2.LockedBuffer(), input2.LDim(),
        output.Buffer(), output.LDim());
  }
  
}

  
/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <typename UnaryOperator>
void apply_entrywise_unary_operator(const AbsDistMat& input,
                                    AbsDistMat& output) {
  std::stringstream err;
  if (input.Height() != output.Height()
      || input.Width() != output.Width()) {
    err << "input matrix dimensions "
        << "(" << input.Height() << " x " << input.Width() << ")"
        << "don't match output matrix dimensions "
        << "(" << output.Height() << " x " << output.Width() << ")";
    LBANN_ERROR(err.str());
  } else if (input.DistData() != output.DistData()) {
    LBANN_ERROR("input and output matrix distributions don't match");
  }
  apply_entrywise_unary_operator<UnaryOperator>(input.LockedMatrix(),
                                                output.Matrix());
}

template <typename BinaryOperator>
void apply_entrywise_binary_operator(const AbsDistMat& input1,
                                     const AbsDistMat& input2,
                                     AbsDistMat& output) {
  if (input1.Height() != input2.Height()
      || input1.Width() != input2.Width()
      || input1.Height() != output.Height()
      || input1.Width() != output.Width()) {
    std::stringstream err;
    err << "input matrix dimensions "
        << "(" << input1.Height() << " x " << input1.Width() << ", "
        << input2.Height() << " x " << input2.Width() << ")"
        << "don't match output matrix dimensions "
        << "(" << output.Height() << " x " << output.Width() << ")";
    LBANN_ERROR(err.str());
  } else if (input1.DistData() != input2.DistData()
             || input1.DistData() != output.DistData()) {
    LBANN_ERROR("input and output matrix distributions don't match");
  }
  apply_entrywise_binary_operator<BinaryOperator>(input1.LockedMatrix(),
                                                  input2.LockedMatrix(),
                                                  output.Matrix());
}
  
#endif // __CUDACC__

// -------------------------------------------------------------
// Utilities for Thrust
// -------------------------------------------------------------
namespace thrust {

/** GPU memory allocator that can interact with Thrust.
 *  Uses Hydrogen's CUB memory pool if available.
 */
template <typename T = El::byte>
class allocator
  : public ::thrust::detail::tagged_allocator<
      T,
      ::thrust::system::cuda::tag,
      ::thrust::pointer<T, ::thrust::system::cuda::tag>> {
private:
  typedef typename ::thrust::detail::tagged_allocator<
    T,
    ::thrust::system::cuda::tag,
    ::thrust::pointer<T, ::thrust::system::cuda::tag>> parent_class;

  /** Active CUDA stream. */
  cudaStream_t m_stream;

public:
  typedef typename parent_class::value_type value_type;
  typedef typename parent_class::pointer    pointer;
  typedef typename parent_class::size_type  size_type;

  allocator(cudaStream_t stream = El::GPUManager::Stream())
    : m_stream(stream) {}

  /** Allocate GPU buffer. */
  pointer allocate(size_type size) {
    value_type* buffer = nullptr;
    if (size > 0) {
#ifdef HYDROGEN_HAVE_CUB
      auto& memory_pool = El::cub::MemoryPool();
      CHECK_CUDA(memory_pool.DeviceAllocate(reinterpret_cast<void**>(&buffer),
                                            size * sizeof(value_type),
                                            m_stream));
#else
      CHECK_CUDA(cudaMalloc(&buffer, size * sizeof(value_type)));
#endif // HYDROGEN_HAVE_CUB
    }
    return pointer(buffer);
  }

  /** Deallocate GPU buffer.
   *  'size' is unused and maintained for compatibility with Thrust.
   */
  void deallocate(pointer buffer, size_type size = 0) {
    auto&& ptr = buffer.get();
    if (ptr != nullptr) {
#ifdef HYDROGEN_HAVE_CUB
      auto& memory_pool = El::cub::MemoryPool();
      CHECK_CUDA(memory_pool.DeviceFree(ptr));
#else
      CHECK_CUDA(cudaFree(ptr));
#endif // HYDROGEN_HAVE_CUB
    }
  }

};

} // namespace thrust

} // namespace cuda
} // namespace lbann

#endif // LBANN_HAS_GPU
#endif // LBANN_UTILS_CUDA_HPP
