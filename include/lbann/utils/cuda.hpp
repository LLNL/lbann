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

// Error utility macros
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
      err_CUDA_SYNC << "CUDA error: "                           \
                    << cudaGetErrorString(status_CUDA_SYNC);    \
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
      LBANN_ERROR(std::string("CUDA error: ")                   \
                  + cudaGetErrorString(status_CHECK_CUDA));     \
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
#ifdef HYDROGEN_HAVE_CUB
    auto& memory_pool = El::cub::MemoryPool();
    CHECK_CUDA(memory_pool.DeviceAllocate(reinterpret_cast<void**>(&buffer),
                                          size * sizeof(value_type),
                                          m_stream));
#else
    CHECK_CUDA(cudaMalloc(&buffer, size * sizeof(value_type)));
#endif // HYDROGEN_HAVE_CUB
    return pointer(buffer);
  }

  /** Deallocate GPU buffer.
   *  'size' is unused and maintained for compatibility with Thrust.
   */
  void deallocate(pointer buffer, size_type size = 0) {
#ifdef HYDROGEN_HAVE_CUB
    auto& memory_pool = El::cub::MemoryPool();
    CHECK_CUDA(memory_pool.DeviceFree(buffer.get()));
#else
    CHECK_CUDA(cudaFree(buffer.get()));
#endif // HYDROGEN_HAVE_CUB
  }

};

} // namespace thrust

} // namespace cuda
} // namespace lbann

#endif // LBANN_HAS_GPU
#endif // LBANN_UTILS_CUDA_HPP
