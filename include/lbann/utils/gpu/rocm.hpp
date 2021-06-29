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

#ifndef LBANN_UTILS_ROCM_HPP
#define LBANN_UTILS_ROCM_HPP

#include "lbann_config.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_ROCM

#include <hip/hip_runtime.h>
#include <thrust/memory.h>
#include <thrust/version.h>
#include <thrust/detail/allocator/tagged_allocator.h>
#include <thrust/system/hip/detail/par.h>
#include <thrust/device_vector.h>

// -------------------------------------------------------------
// Error utility macros
// -------------------------------------------------------------
#define LBANN_ROCM_SYNC(async)                                  \
  do {                                                          \
    /* Synchronize GPU and check for errors. */                 \
    hipError_t status_ROCM_SYNC = hipDeviceSynchronize();       \
    if (status_ROCM_SYNC == hipSuccess)                         \
      status_ROCM_SYNC = hipGetLastError();                     \
    if (status_ROCM_SYNC != hipSuccess) {                       \
      LBANN_ERROR((async ? "Asynchronous " : ""),               \
                  "ROCm error (",                               \
                  hipGetErrorString(status_ROCM_SYNC),          \
                  ")");                                         \
    }                                                           \
  } while (0)
#define LBANN_ROCM_CHECK_LAST_ERROR(async)                      \
  do {                                                          \
    hipError_t status = hipGetLastError();                      \
    if (status != hipSuccess) {                                 \
      LBANN_ERROR((async ? "Asynchronous " : ""),               \
                  "ROCm error (",                               \
                  hipGetErrorString(status),                    \
                  ")");                                         \
    }                                                           \
  } while (0)
#define FORCE_CHECK_ROCM(rocm_call)                             \
  do {                                                          \
    /* Call ROCM API routine, synchronizing before and */       \
    /* after to check for errors. */                            \
    LBANN_ROCM_SYNC(true);                                      \
    hipError_t status_CHECK_ROCM = (rocm_call);                 \
    if (status_CHECK_ROCM != hipSuccess) {                      \
      LBANN_ERROR("ROCm error (",                               \
                  hipGetErrorString(status_CHECK_ROCM),         \
                  ")");                                         \
    }                                                           \
    LBANN_ROCM_SYNC(false);                                     \
  } while (0)
#define FORCE_CHECK_ROCM_NOSYNC(rocm_call)                      \
  do {                                                          \
    hipError_t status_CHECK_ROCM = (rocm_call);                 \
    if (status_CHECK_ROCM != hipSuccess) {                      \
      LBANN_ERROR("ROCm error (",                               \
                  hipGetErrorString(status_CHECK_ROCM),         \
                  ")");                                         \
    }                                                           \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_ROCM(rocm_call) FORCE_CHECK_ROCM(rocm_call);
#else
#define CHECK_ROCM(rocm_call) FORCE_CHECK_ROCM_NOSYNC(rocm_call)
#endif // #ifdef LBANN_DEBUG

namespace lbann {
namespace rocm {

// -------------------------------------------------------------
// Wrapper classes
// -------------------------------------------------------------

/** Wrapper class for a HIP event. */
class event_wrapper {
public:
  event_wrapper();
  event_wrapper(const event_wrapper& other);
  event_wrapper& operator=(const event_wrapper& other);
  ~event_wrapper();
  /** Enqueue HIP event on a HIP stream. */
  void record(hipStream_t stream);
  /** Check whether HIP event has completed. */
  bool query() const;
  /** Wait until HIP event has completed. */
  void synchronize();
  /** Get HIP event object. */
  hipEvent_t& get_event();
private:
  /** HIP event object.
   *  The event object lifetime is managed internally.
   */
  hipEvent_t m_event;
  /** HIP stream object.
   *  The stream object lifetime is assumed to be managed externally.
   */
  hipStream_t m_stream;
};

// -------------------------------------------------------------
// Helper functions for tensor operations
// -------------------------------------------------------------

/** Copy entries between GPU tensors. */
template <typename TensorDataType>
void copy_tensor(
  hipStream_t stream,
  const std::vector<size_t>& dims,
  const TensorDataType* input,
  const std::vector<size_t>& input_strides,
  TensorDataType* output,
  const std::vector<size_t>& output_strides);

template <typename TensorDataType>
void mem_copy_async(
  TensorDataType* output,
  const TensorDataType* input,
  const std::vecotr<size_t>& dims,
  hipMemcpyKind kind,
  hipStream_t stream);

// -------------------------------------------------------------
// Utilities for Thrust
// -------------------------------------------------------------
namespace thrust {

/** Thrust execution policy. */
using execute_on_stream = ::thrust::hip_rocprim::execute_on_stream;

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
  allocator(hipStream_t stream = hydrogen::rocm::GetDefaultStream());
  /** Allocate GPU buffer. */
  pointer allocate(size_type size);
  /** Deallocate GPU buffer.
   *  'size' is unused and maintained for compatibility with Thrust.
   */
  void deallocate(pointer buffer, size_type size = 0);
  /** Get Thrust execution policy. */
  system_type& system();

private:
  /** Active HIP stream. */
  hipStream_t m_stream;
  /** Thrust execution policy. */
  system_type m_system;

};

/** Thrust device vector. */
template <typename T>
using vector = ::thrust::device_vector<T, allocator<T>>;

} // namespace thrust
} // namespace rocm
} // namespace lbann

#endif // LBANN_HAS_ROCM
#endif // LBANN_UTILS_ROCM_HPP
