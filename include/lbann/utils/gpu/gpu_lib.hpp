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

#ifndef LBANN_UTILS_GPU_LIB_HPP
#define LBANN_UTILS_GPU_LIB_HPP

#include "lbann_config.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_GPU

#include <cuda.h>
#include <thrust/memory.h>
#include <thrust/version.h>
#include <thrust/detail/allocator/tagged_allocator.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/device_vector.h>

namespace lbann {
//namespace cuda {
namespace gpu_lib {
  using namespace cuda;

// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------
#ifdef __CUDACC__
__device__ __forceinline__
__half atomic_add(__half* address, __half val);
__device__ __forceinline__
float atomic_add(float* address, float val);
__device__ __forceinline__
double atomic_add(double* address, double val);
  /*
// Atomic add
template <typename T> __device__ __forceinline__
T atomic_add(T* address, T val);
*/
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
  __device__ __forceinline__ type name(type const&)
#define DECLARE_UNARY_MATH_FUNC(func)                \
  DECLARE_UNARY_MATH_FUNC_WITH_TYPE(func, __half);   \
  DECLARE_UNARY_MATH_FUNC_WITH_TYPE(func, float);    \
  DECLARE_UNARY_MATH_FUNC_WITH_TYPE(func, double)
template <typename T> __device__ __forceinline__ T abs(const T& x);
DECLARE_UNARY_MATH_FUNC(abs);
DECLARE_UNARY_MATH_FUNC(ceil);
DECLARE_UNARY_MATH_FUNC(round);
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
template <typename T> __device__ __forceinline__ bool isfinite(const T& x);
template <typename T> __device__ __forceinline__ bool isinf(const T& x);
template <typename T> __device__ __forceinline__ bool isnan(const T& x);
#undef DECLARE_UNARY_MATH_FUNC
#undef DECLARE_UNARY_MATH_FUNC_WITH_TYPE

// Binary math functions
template <typename T> __device__ __forceinline__ T min(const T& x, const T& y);
template <typename T> __device__ __forceinline__ T max(const T& x, const T& y);
template <typename T> __device__ __forceinline__ T mod(const T& x, const T& y);
template <typename T> __device__ __forceinline__ T pow(const T& x, const T& y);
#define DECLARE_UNARY_MATH_BINARY_FUNC_WITH_TYPE(func, type)    \
  __device__ __forceinline__ type name(type const& x, type const& y)
#define DECLARE_UNARY_MATH_BINARY_FUNC(func)                \
  DECLARE_UNARY_MATH_BINARY_FUNC_WITH_TYPE(func, __half);   \
  DECLARE_UNARY_MATH_BINARY_FUNC_WITH_TYPE(func, float);    \
  DECLARE_UNARY_MATH_BINARY_FUNC_WITH_TYPE(func, double)
DECLARE_UNARY_MATH_BINARY_FUNC(min);
DECLARE_UNARY_MATH_BINARY_FUNC(max);
DECLARE_UNARY_MATH_BINARY_FUNC(mod);
DECLARE_UNARY_MATH_BINARY_FUNC(pow);
#undef DECLARE_UNARY_MATH_BINARY_FUNC
#undef DECLARE_UNARY_MATH_BINARY_FUNC_WITH_TYPE


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


} // namespace cuda
} // namespace lbann

// Header implementations
#include "lbann/utils/impl/gpu_lib.hpp"
#if defined LBANN_HAS_CUDA
#include "lbann/utils/impl/cuda.hpp"
#else
#include "lbann/utils/impl/rocm.hpp"
#endif // LBANN_HAS_CUDA

#endif // LBANN_HAS_GPU
#endif // LBANN_UTILS_GPU_LIB_HPP
