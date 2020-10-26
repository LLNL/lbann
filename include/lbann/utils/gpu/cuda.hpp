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
__device__ __forceinline__
__half atomic_add(__half* address, __half val);
__device__ __forceinline__
float atomic_add(float* address, float val);
__device__ __forceinline__
double atomic_add(double* address, double val);

/** @brief Sum over threads in CUDA block
 *
 *  Every thread in a CUDA block must enter this function. The sum is
 *  returned on thread 0.
 *
 *  @tparam bdimx   x-dimension of CUDA block
 *  @tparam bdimy   y-dimension of CUDA block
 *  @tparam bdimz   z-dimension of CUDA block
 *  @tparam T       Data type
 *  @param  val     Contribution from thread
 *  @returns On thread 0, the sum. Not meaningful on other threads.
 */
template <size_t bdimx, size_t bdimy, size_t bdimz, class T>
__device__ __forceinline__
T block_reduce(T val);

/** @brief Reduction over threads in CUDA block
 *
 *  Every thread in a CUDA block must enter this function. The reduced
 *  value is returned on thread 0.
 *
 *  @tparam bdimx   x-dimension of CUDA block
 *  @tparam bdimy   y-dimension of CUDA block
 *  @tparam bdimz   z-dimension of CUDA block
 *  @tparam T       Data type
 *  @tparam Op      Functor for reduction operation
 *  @param  val     Contribution from each thread
 *  @returns On thread 0, the reduced value. Not meaningful on other
 *  threads.
 */
template <size_t bdimx, size_t bdimy, size_t bdimz, class T, class Op>
__device__ __forceinline__
T block_reduce(T val);

// Unary math functions
#define DECLARE_UNARY_MATH_FUNC_WITH_TYPE(func, type)    \
  __device__ __forceinline__ type func(type const& x)
#define DECLARE_UNARY_MATH_FUNC(func)                 \
  DECLARE_UNARY_MATH_FUNC_WITH_TYPE(func, __half);    \
  DECLARE_UNARY_MATH_FUNC_WITH_TYPE(func, float);     \
  DECLARE_UNARY_MATH_FUNC_WITH_TYPE(func, double)
template <typename T> __device__ __forceinline__ T abs(const T& x);
__device__ __forceinline__ float abs(float const& x);
__device__ __forceinline__ double abs(double const& x);
DECLARE_UNARY_MATH_FUNC(round);
DECLARE_UNARY_MATH_FUNC(ceil);
DECLARE_UNARY_MATH_FUNC(floor);
DECLARE_UNARY_MATH_FUNC(sqrt);
DECLARE_UNARY_MATH_FUNC(rsqrt);
DECLARE_UNARY_MATH_FUNC(exp);
DECLARE_UNARY_MATH_FUNC(expm1);
DECLARE_UNARY_MATH_FUNC(log);
DECLARE_UNARY_MATH_FUNC(log1p);
DECLARE_UNARY_MATH_FUNC(cos);
DECLARE_UNARY_MATH_FUNC(sin);
DECLARE_UNARY_MATH_FUNC(tan);
DECLARE_UNARY_MATH_FUNC(acos);
DECLARE_UNARY_MATH_FUNC(asin);
DECLARE_UNARY_MATH_FUNC(atan);
DECLARE_UNARY_MATH_FUNC(cosh);
DECLARE_UNARY_MATH_FUNC(sinh);
DECLARE_UNARY_MATH_FUNC(tanh);
DECLARE_UNARY_MATH_FUNC(acosh);
DECLARE_UNARY_MATH_FUNC(asinh);
DECLARE_UNARY_MATH_FUNC(atanh);
DECLARE_UNARY_MATH_FUNC(erf);
DECLARE_UNARY_MATH_FUNC(erfinv);
template <typename T> __device__ __forceinline__ bool isfinite(const T& x);
template <typename T> __device__ __forceinline__ bool isinf(const T& x);
template <typename T> __device__ __forceinline__ bool isnan(const T& x);
#undef DECLARE_UNARY_MATH_FUNC
#undef DECLARE_UNARY_MATH_FUNC_WITH_TYPE

// Binary math functions
#define DECLARE_BINARY_UNARY_MATH_FUNC_WITH_TYPE(func, type)            \
  __device__ __forceinline__ type func(type const& x, type const& y)
#define DECLARE_BINARY_UNARY_MATH_FUNC(func)                 \
  DECLARE_BINARY_UNARY_MATH_FUNC_WITH_TYPE(func, __half);    \
  DECLARE_BINARY_UNARY_MATH_FUNC_WITH_TYPE(func, float);     \
  DECLARE_BINARY_UNARY_MATH_FUNC_WITH_TYPE(func, double)
template <typename T> __device__ __forceinline__ T min(const T& x, const T& y);
DECLARE_BINARY_UNARY_MATH_FUNC(min);
template <typename T> __device__ __forceinline__ T max(const T& x, const T& y);
DECLARE_BINARY_UNARY_MATH_FUNC(max);
DECLARE_BINARY_UNARY_MATH_FUNC(mod);
DECLARE_BINARY_UNARY_MATH_FUNC(pow);
#undef DECLARE_BINARY_UNARY_MATH_FUNC
#undef DECLARE_BINARY_UNARY_MATH_FUNC_WITH_TYPE

// Numeric limits
template <typename T> constexpr __device__ __forceinline__ T min();
template <typename T> constexpr __device__ __forceinline__ T max();
template <typename T> constexpr __device__ __forceinline__ T epsilon();
template <typename T> __device__ __forceinline__ T infinity();

/** @brief Array with fixed type and size. */
template <typename T, size_t N>
struct array {
  T vals[N];
  __host__ __device__ __forceinline__ size_t size() const;
  __host__ __device__ __forceinline__ T& operator[](size_t i);
  __host__ __device__ __forceinline__ const T& operator[](size_t i) const;
};

#endif // __CUDACC__

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

#ifdef __CUDACC__

/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <template <typename> class UnaryOperator, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractMatrix<TensorDataType>& input,
  El::AbstractMatrix<TensorDataType>& output);

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <template <typename> class BinaryOperator, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractMatrix<TensorDataType>& input1,
  const El::AbstractMatrix<TensorDataType>& input2,
  El::AbstractMatrix<TensorDataType>& output);


/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class UnaryOperator, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input,
  El::AbstractDistMatrix<TensorDataType>& output);

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class BinaryOperator, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input1,
  const El::AbstractDistMatrix<TensorDataType>& input2,
  El::AbstractDistMatrix<TensorDataType>& output);

#endif // __CUDACC__

/** Copy entries between GPU tensors. */
template <typename TensorDataType>
void copy_tensor(
  cudaStream_t stream,
  const std::vector<size_t>& dims,
  const TensorDataType* input,
  const std::vector<size_t>& input_strides,
  TensorDataType* output,
  const std::vector<size_t>& output_strides);

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

// Header implementations
#include "lbann/utils/impl/cuda.hpp"

#endif // LBANN_HAS_GPU
#endif // LBANN_UTILS_CUDA_HPP
