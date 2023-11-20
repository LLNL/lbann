////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

// Headers for HIP
#ifdef __HIPCC__
#include <thrust/system/hip/execution_policy.h>
#ifdef HYDROGEN_HAVE_CUB
#include "hipcub/block/block_reduce.hpp"
#endif // HYDROGEN_HAVE_CUB
#include <hip/hip_fp16.h>
#include <limits>
#endif // __HIPCC__

namespace lbann {

// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------
#ifdef __HIPCC__

// Atomic add function
__device__ __forceinline__ __half gpu_lib::atomic_add(__half* address,
                                                      __half val)
{
  if (((uintptr_t)address) % 4 > 0) // odd-indexed element
  {
    // Previous element
    unsigned int* prv_address_as_uint = (unsigned int*)(address - 1);
    unsigned int old = *prv_address_as_uint;
    unsigned int assumed, updated;
    __half* old_as_half = ((__half*)&old) + 1;
    __half* updated_as_half = ((__half*)&updated) + 1;
    do {
      assumed = old;
      updated = old;
      *updated_as_half += val;
      old = atomicCAS(prv_address_as_uint, assumed, updated);
    } while (assumed != old);
    return *old_as_half;
  }
  else {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    __half* old_as_half = (__half*)&old;
    unsigned int assumed;
    unsigned int updated;
    __half* updated_as_half = (__half*)&updated;
    do {
      assumed = old;
      updated = old;
      *updated_as_half += val;
      old = atomicCAS(address_as_uint, assumed, updated);
    } while (assumed != old);
    return *old_as_half;
  }
}
__device__ __forceinline__ float gpu_lib::atomic_add(float* address, float val)
{
  return atomicAdd(address, val);
}
__device__ __forceinline__ double gpu_lib::atomic_add(double* address,
                                                      double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

// Block reduction
template <size_t bdimx, size_t bdimy, size_t bdimz, class T>
__device__ __forceinline__ T gpu_lib::block_reduce(T val)
{
#ifdef HYDROGEN_HAVE_CUB
  constexpr auto reduce_algo = hipcub::BLOCK_REDUCE_WARP_REDUCTIONS;
  using BlockReduce = hipcub::BlockReduce<T, bdimx, reduce_algo, bdimy, bdimz>;
  __shared__ typename BlockReduce::TempStorage workspace;
  val = BlockReduce(workspace).Sum(val);
#else
  const size_t tid =
    threadIdx.x + threadIdx.y * bdimx + threadIdx.z * bdimx * bdimy;
  constexpr size_t bsize = bdimx * bdimy * bdimz;
  //__shared__ T shared_max_vals[bsize];
  __shared__ char shared_max_vals_buffer[bsize * sizeof(T)];
  T* shared_max_vals = reinterpret_cast<T*>(shared_max_vals);
  shared_max_vals[tid] = val;
  for (size_t stride = bsize / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared_max_vals[tid] =
        shared_max_vals[tid] + shared_max_vals[tid + stride];
    }
  }
  if (tid == 0) {
    val = shared_max_vals[0];
  }
#endif // HYDROGEN_HAVE_CUB
  return val;
}
template <size_t bdimx, size_t bdimy, size_t bdimz, class T, class Op>
__device__ __forceinline__ T gpu_lib::block_reduce(T val)
{
#ifdef HYDROGEN_HAVE_CUB
  constexpr auto reduce_algo = hipcub::BLOCK_REDUCE_WARP_REDUCTIONS;
  using BlockReduce = hipcub::BlockReduce<T, bdimx, reduce_algo, bdimy, bdimz>;
  __shared__ typename BlockReduce::TempStorage workspace;
  val = BlockReduce(workspace).Reduce(val, Op());
#else
  Op op;
  const size_t tid =
    threadIdx.x + threadIdx.y * bdimx + threadIdx.z * bdimx * bdimy;
  constexpr size_t bsize = bdimx * bdimy * bdimz;
  //__shared__ DataType shared_max_vals[bsize];
  __shared__ char shared_max_vals_buffer[bsize * sizeof(T)];
  T* shared_max_vals = reinterpret_cast<T*>(shared_max_vals);
  shared_max_vals[tid] = val;
  for (size_t stride = bsize / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared_max_vals[tid] =
        op(shared_max_vals[tid], shared_max_vals[tid + stride]);
    }
  }
  if (tid == 0) {
    val = shared_max_vals[0];
  }
#endif // HYDROGEN_HAVE_CUB
  return val;
}

// Unary math functions
// This support is far from complete!
#define WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(func)                               \
  __device__ __forceinline__ __half gpu_lib::func(__half const& x)             \
  {                                                                            \
    return ::h##func(x);                                                       \
  }

// FIXME (trb): This is maybe not the best long-term solution, but it
// might be the best we can do without really digging into
// half-precision implementation.
#define WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(func)                 \
  __device__ __forceinline__ __half gpu_lib::func(__half const& x)             \
  {                                                                            \
    return gpu_lib::func(float(x));                                            \
  }

WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(round)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(ceil)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(floor)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(sqrt)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(rsqrt)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(exp)

// WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(expm1)
//
//  FIXME (trb): This is not going to be as accurate as a native expm1
//  implementation could be:
__device__ __forceinline__ __half gpu_lib::expm1(__half const& x)
{
  return ::__hsub(::hexp(x), ::__float2half(1.f));
}

WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(log)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(log1p)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(cos)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(sin)

// WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(tan)
//
//  FIXME (trb): This just uses the trig identity. Probably less
//  accurate than a native implementation.
__device__ __forceinline__ __half gpu_lib::tan(__half const& x)
{
  return ::__hdiv(::hsin(x), ::hcos(x));
}

WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(acos)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(asin)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(atan)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(cosh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(sinh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(tanh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(acosh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(asinh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(atanh)

WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(erf)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(erfinv)
#undef WRAP_UNARY_ROCM_HALF_MATH_FUNCTION

template <>
__device__ __forceinline__ bool gpu_lib::isnan(__half const& x)
{
  return __hisnan(x);
}

template <>
__device__ __forceinline__ bool gpu_lib::isinf(__half const& x)
{
  return __hisinf(x);
}

template <>
__device__ __forceinline__ bool gpu_lib::isfinite(__half const& x)
{
  return !(__hisnan(x) || __hisinf(x));
}

// Binary math functions
__device__ __forceinline__ __half gpu_lib::min(const __half& x, const __half& y)
{
  return ::__hle(x, y) ? x : y;
}

__device__ __forceinline__ __half gpu_lib::max(const __half& x, const __half& y)
{
  return ::__hle(x, y) ? y : x;
}

// Numeric limits
template <typename T>
constexpr __device__ __forceinline__ T gpu_lib::min()
{
  return std::numeric_limits<T>::min();
}
template <typename T>
constexpr __device__ __forceinline__ T gpu_lib::max()
{
  return std::numeric_limits<T>::max();
}
template <typename T>
constexpr __device__ __forceinline__ T gpu_lib::epsilon()
{
  return std::numeric_limits<T>::epsilon();
}
template <typename T>
__device__ __forceinline__ T gpu_lib::infinity()
{
  return std::numeric_limits<T>::infinity();
}

namespace rocm {

// -------------------------------------------------------------
// Utilities for Thrust
// -------------------------------------------------------------
#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace thrust {

template <typename T>
allocator<T>::allocator(hipStream_t stream) : m_stream(stream), m_system(stream)
{}

template <typename T>
typename allocator<T>::pointer
allocator<T>::allocate(allocator<T>::size_type size)
{
  value_type* buffer = nullptr;
  if (size > 0) {
#ifdef HYDROGEN_HAVE_CUB
    auto& memory_pool = El::cub::MemoryPool();
    CHECK_ROCM(memory_pool.DeviceAllocate(reinterpret_cast<void**>(&buffer),
                                          size * sizeof(value_type),
                                          m_stream));
#else
    CHECK_ROCM(hipMalloc(&buffer, size * sizeof(value_type)));
#endif // HYDROGEN_HAVE_CUB
  }
  return pointer(buffer);
}

template <typename T>
void allocator<T>::deallocate(allocator<T>::pointer buffer,
                              allocator<T>::size_type size)
{
  auto&& ptr = buffer.get();
  if (ptr != nullptr) {
#ifdef HYDROGEN_HAVE_CUB
    auto& memory_pool = El::cub::MemoryPool();
    CHECK_ROCM(memory_pool.DeviceFree(ptr, m_stream));
#else
    CHECK_ROCM(hipFree(ptr));
#endif // HYDROGEN_HAVE_CUB
  }
}

template <typename T>
typename allocator<T>::system_type& allocator<T>::system()
{
  return m_system;
}

} // namespace thrust
#endif // !DOXYGEN_SHOULD_SKIP_THIS

} // namespace rocm

#endif // __HIPCC__

} // namespace lbann
