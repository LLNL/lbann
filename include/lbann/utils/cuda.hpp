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

#ifdef __CUDACC__
// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------

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
__device__ __inline__ float min(float x, float y) { return fminf(x, y); }
__device__ __inline__ double min(double x, double y) { return fmin(x, y); }
__device__ __inline__ float max(float x, float y) { return fmaxf(x, y); }
__device__ __inline__ double max(double x, double y) { return fmax(x, y); }
  
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
