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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_UTILS_CUDA_HPP
#define LBANN_UTILS_CUDA_HPP

#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_GPU

#include <cuda.h>
#include <thrust/memory.h>
#include <thrust/version.h>
#include <thrust/detail/allocator/tagged_allocator.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/device_vector.h>

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
#define LBANN_CUDA_CHECK_LAST_ERROR(async)                              \
  do {                                                                  \
    cudaError_t status = cudaGetLastError();                            \
    if (status != cudaSuccess) {                                        \
      cudaDeviceReset();                                                \
      std::stringstream err_CUDA_CHECK_LAST_ERROR;                      \
      if (async) { err_CUDA_CHECK_LAST_ERROR << "Asynchronous "; }      \
      err_CUDA_CHECK_LAST_ERROR << "CUDA error ("                       \
                                << cudaGetErrorString(status)           \
                                << ")";                                 \
      LBANN_ERROR(err_CUDA_CHECK_LAST_ERROR.str());                     \
    }                                                                   \
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
#define FORCE_CHECK_CUDA_NOSYNC(cuda_call)                      \
  do {                                                          \
    cudaError_t status_CHECK_CUDA = (cuda_call);                \
    if (status_CHECK_CUDA != cudaSuccess) {                     \
      LBANN_ERROR(std::string("CUDA error (")                   \
                  + cudaGetErrorString(status_CHECK_CUDA)       \
                  + std::string(")"));                          \
    }                                                           \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUDA(cuda_call) FORCE_CHECK_CUDA(cuda_call);
#else
#define CHECK_CUDA(cuda_call) FORCE_CHECK_CUDA_NOSYNC(cuda_call)
#endif // #ifdef LBANN_DEBUG

namespace lbann {
namespace cuda {

// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------
#ifdef __CUDACC__

// Atomic add
template <typename T> __device__ __forceinline__
T atomic_add(T* address, T val);

// Unary math functions
template <typename T> __device__ __forceinline__ T abs(const T& x);
template <typename T> __device__ __forceinline__ T round(const T& x);
template <typename T> __device__ __forceinline__ T ceil(const T& x);
template <typename T> __device__ __forceinline__ T floor(const T& x);
template <typename T> __device__ __forceinline__ T sqrt(const T& x);
template <typename T> __device__ __forceinline__ T rsqrt(const T& x);
template <typename T> __device__ __forceinline__ T exp(const T& x);
template <typename T> __device__ __forceinline__ T expm1(const T& x);
template <typename T> __device__ __forceinline__ T log(const T& x);
template <typename T> __device__ __forceinline__ T log1p(const T& x);
template <typename T> __device__ __forceinline__ T cos(const T& x);
template <typename T> __device__ __forceinline__ T sin(const T& x);
template <typename T> __device__ __forceinline__ T tan(const T& x);
template <typename T> __device__ __forceinline__ T acos(const T& x);
template <typename T> __device__ __forceinline__ T asin(const T& x);
template <typename T> __device__ __forceinline__ T atan(const T& x);
template <typename T> __device__ __forceinline__ T cosh(const T& x);
template <typename T> __device__ __forceinline__ T sinh(const T& x);
template <typename T> __device__ __forceinline__ T tanh(const T& x);
template <typename T> __device__ __forceinline__ T acosh(const T& x);
template <typename T> __device__ __forceinline__ T asinh(const T& x);
template <typename T> __device__ __forceinline__ T atanh(const T& x);

// Binary math functions
template <typename T> __device__ __forceinline__ T min(const T& x, const T& y);
template <typename T> __device__ __forceinline__ T max(const T& x, const T& y);
template <typename T> __device__ __forceinline__ T mod(const T& x, const T& y);
template <typename T> __device__ __forceinline__ T pow(const T& x, const T& y);

// Numeric limits
template <typename T> constexpr __device__ __forceinline__ T min();
template <typename T> constexpr __device__ __forceinline__ T max();
template <typename T> constexpr __device__ __forceinline__ T epsilon();
template <typename T> __device__ __forceinline__ T infinity();

#endif // __CUDACC__

// -------------------------------------------------------------
// Utilities for CUDA events
// -------------------------------------------------------------

/** Wrapper class for a CUDA event. */
class event_wrapper {
public:
  event_wrapper();
  event_wrapper(const event_wrapper& other);
  event_wrapper& operator=(const event_wrapper& other);
  ~event_wrapper();
  /** Enqueue CUDA event on a CUDA stream. */
  void record(cudaStream_t stream);
  /** Check whether CUDA event has completed. */
  bool query() const;
  /** Wait until CUDA event has completed. */
  void synchronize();
  /** Get CUDA event object. */
  cudaEvent_t& get_event();
private:
  /** CUDA event object.
   *  The event object lifetime is managed internally.
   */
  cudaEvent_t m_event;
  /** CUDA stream object.
   *  The stream object lifetime is assumed to be managed externally.
   */
  cudaStream_t m_stream;
};

// -------------------------------------------------------------
// Helper functions for entrywise operations
// -------------------------------------------------------------
#ifdef __CUDACC__

/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <typename UnaryOperator>
void apply_entrywise_unary_operator(const AbsMat& input,
                                    AbsMat& output);

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <typename BinaryOperator>
void apply_entrywise_binary_operator(const AbsMat& input1,
                                     const AbsMat& input2,
                                     AbsMat& output);


/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <typename UnaryOperator>
void apply_entrywise_unary_operator(const AbsDistMat& input,
                                    AbsDistMat& output);

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <typename BinaryOperator>
void apply_entrywise_binary_operator(const AbsDistMat& input1,
                                     const AbsDistMat& input2,
                                     AbsDistMat& output);

#endif // __CUDACC__

// -------------------------------------------------------------
// Utilities for Thrust
// -------------------------------------------------------------
namespace thrust {

/** Thrust execution policy. */
using execute_on_stream
#if THRUST_MAJOR_VERSION > 1 || THRUST_MINOR_VERSION >= 9
  = ::thrust::cuda_cub::execute_on_stream; // >= 1.9.1
#elif THRUST_MAJOR_VERSION == 1 && THRUST_MINOR_VERSION == 8
  = ::thrust::system::cuda::detail::execute_on_stream;
#else
  = std::nullptr_t;
  static_assert(false, "Thrust 1.8 or newer is required");
#endif

/** GPU memory allocator that can interact with Thrust.
 *  Operations are performed on a provided CUDA stream. Uses
 *  Hydrogen's CUB memory pool if available.
 */
template <typename T = El::byte>
class allocator
  : public ::thrust::detail::tagged_allocator<
               T, execute_on_stream,
               ::thrust::pointer<T, execute_on_stream>> {
public:
  // Convenient typedefs
  typedef ::thrust::detail::tagged_allocator<
              T, execute_on_stream,
              ::thrust::pointer<T, execute_on_stream>> parent_class;
  typedef typename parent_class::value_type  value_type;
  typedef typename parent_class::pointer     pointer;
  typedef typename parent_class::size_type   size_type;
  typedef typename parent_class::system_type system_type;

  /** Default constructor. */
  allocator(cudaStream_t stream = El::GPUManager::Stream());
  /** Allocate GPU buffer. */
  pointer allocate(size_type size);
  /** Deallocate GPU buffer.
   *  'size' is unused and maintained for compatibility with Thrust.
   */
  void deallocate(pointer buffer, size_type size = 0);
  /** Get Thrust execution policy. */
  system_type& system();

private:
  /** Active CUDA stream. */
  cudaStream_t m_stream;
  /** Thrust execution policy. */
  system_type m_system;

};

/** Thrust device vector. */
template <typename T>
using vector = ::thrust::device_vector<T, allocator<T>>;

} // namespace thrust

} // namespace cuda
} // namespace lbann

// Header implementations
#include "lbann/utils/impl/cuda.hpp"

#endif // LBANN_HAS_GPU
#endif // LBANN_UTILS_CUDA_HPP
