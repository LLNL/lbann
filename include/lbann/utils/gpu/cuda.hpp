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

#include "lbann_config.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_CUDA

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
      LBANN_ERROR((async ? "Asynchronous " : ""),               \
                  "CUDA error (",                               \
                  cudaGetErrorString(status_CUDA_SYNC),         \
                  ")");                                         \
    }                                                           \
  } while (0)
#define LBANN_CUDA_CHECK_LAST_ERROR(async)                      \
  do {                                                          \
    cudaError_t status = cudaGetLastError();                    \
    if (status != cudaSuccess) {                                \
      LBANN_ERROR((async ? "Asynchronous " : ""),               \
                  "CUDA error (",                               \
                  cudaGetErrorString(status),                   \
                  ")");                                         \
    }                                                           \
  } while (0)
#define FORCE_CHECK_CUDA(cuda_call)                             \
  do {                                                          \
    /* Call CUDA API routine, synchronizing before and */       \
    /* after to check for errors. */                            \
    LBANN_CUDA_SYNC(true);                                      \
    cudaError_t status_CHECK_CUDA = (cuda_call);                \
    if (status_CHECK_CUDA != cudaSuccess) {                     \
      LBANN_ERROR("CUDA error (",                               \
                  cudaGetErrorString(status_CHECK_CUDA),        \
                  ")");                                         \
    }                                                           \
    LBANN_CUDA_SYNC(false);                                     \
  } while (0)
#define FORCE_CHECK_CUDA_NOSYNC(cuda_call)                      \
  do {                                                          \
    cudaError_t status_CHECK_CUDA = (cuda_call);                \
    if (status_CHECK_CUDA != cudaSuccess) {                     \
      LBANN_ERROR("CUDA error (",                               \
                  cudaGetErrorString(status_CHECK_CUDA),        \
                  ")");                                         \
    }                                                           \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUDA(cuda_call) FORCE_CHECK_CUDA(cuda_call);
#else
#define CHECK_CUDA(cuda_call) FORCE_CHECK_CUDA_NOSYNC(cuda_call)
#endif // #ifdef LBANN_DEBUG

namespace lbann {
namespace cuda {

constexpr cudaMemcpyKind GPU_MEMCPY_DEVICE_TO_DEVICE = cudaMemcpyDeviceToDevice;

// -------------------------------------------------------------
// Wrapper classes
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

/** Wrapper around @c cudaGraph_t */
class Graph {

public:

  Graph(cudaGraph_t graph=nullptr);
  ~Graph();

  // Copy-and-swap idiom
  Graph(const Graph&);
  Graph(Graph&&);
  Graph& operator=(Graph);
  friend void swap(Graph& first, Graph& second);

  /** @brief Take ownership of CUDA object */
  void reset(cudaGraph_t graph=nullptr);
  /** @brief Return CUDA object and release ownership */
  cudaGraph_t release();
  /** @brief Return CUDA object without releasing ownership */
  cudaGraph_t get() const noexcept;
  /** @brief Return CUDA object without releasing ownership */
  operator cudaGraph_t() const noexcept;

  /** @brief Create CUDA object
   *
   *  Does nothing if already created.
   */
  void create();

  /** @begin Begin stream capture */
  static void begin_capture(
    cudaStream_t stream,
    cudaStreamCaptureMode mode=cudaStreamCaptureModeGlobal);
  /** @begin End stream capture and return the resulting CUDA graph */
  static Graph end_capture(cudaStream_t stream);

private:

  cudaGraph_t graph_{nullptr};

};

/** Wrapper around @c cudaGraphExec_t */
class ExecutableGraph {

public:

  ExecutableGraph(cudaGraphExec_t graph_exec=nullptr);
  ExecutableGraph(cudaGraph_t graph);
  ~ExecutableGraph();

  // Copy-and-swap idiom
  ExecutableGraph(const ExecutableGraph&) = delete;
  ExecutableGraph(ExecutableGraph&&);
  ExecutableGraph& operator=(ExecutableGraph);
  friend void swap(ExecutableGraph& first, ExecutableGraph& second);

  /** @brief Take ownership of CUDA object */
  void reset(cudaGraphExec_t graph=nullptr);
  /** @brief Return CUDA object and release ownership */
  cudaGraphExec_t release();
  /** @brief Return CUDA object without releasing ownership */
  cudaGraphExec_t get() const noexcept;
  /** @brief Return CUDA object without releasing ownership */
  operator cudaGraphExec_t() const noexcept;

  /** @brief Execute CUDA graph */
  void launch(cudaStream_t stream) const;

  /** @brief Update CUDA graph
   *
   *  Creates new executable graph if it has not already been created
   *  or if update fails.
   */
  void update(cudaGraph_t graph);

private:

  cudaGraphExec_t graph_exec_{nullptr};

};

// -------------------------------------------------------------
// Helper functions for tensor operations
// -------------------------------------------------------------

/** Copy entries between GPU tensors. */
template <typename TensorDataType>
void copy_tensor(
  cudaStream_t stream,
  const std::vector<size_t>& dims,
  const TensorDataType* input,
  const std::vector<size_t>& input_strides,
  TensorDataType* output,
  const std::vector<size_t>& output_strides);

void mem_copy_async(
  void* output,
  const void* input,
  const size_t count,
  cudaMemcpyKind kind,
  cudaStream_t stream);

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
  allocator(cudaStream_t stream = hydrogen::cuda::GetDefaultStream());
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

#endif // LBANN_HAS_CUDA

#endif // LBANN_UTILS_CUDA_HPP
