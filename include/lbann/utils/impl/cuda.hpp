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

#include <thrust/system/cuda/execution_policy.h>

// Headers for NVCC
#ifdef __CUDACC__
#ifdef HYDROGEN_HAVE_CUB
#include "cub/block/block_reduce.cuh"
#endif // HYDROGEN_HAVE_CUB
#include <math_constants.h>
#include <cuda_fp16.h>
#endif // __CUDACC__

namespace lbann {
namespace cuda {

// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------
#ifdef __CUDACC__
// Atomic add function
#if __CUDA_ARCH__ >= 530
template <> __device__ __forceinline__
__half atomic_add<__half>(__half* address, __half val) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
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
#endif // __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
}
#endif // __CUDA_ARCH__ >= 530
template <> __device__ __forceinline__
float atomic_add<float>(float* address, float val) {
  return atomicAdd(address, val);
}
template <> __device__ __forceinline__
double atomic_add<double>(double* address, double val) {
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

// Unary math functions
#if __CUDA_ARCH__ >= 530
__device__ __forceinline__
bool isfinite(__half const& x) { return !(::__isnan(x) || ::__hisinf(x)); }
__device__ __forceinline__
bool isinf(__half const& x) { return ::__hisinf(x); }
__device__ __forceinline__
bool isnan(__half const& x) { return ::__hisnan(x); }

// This support is far from complete!
#define WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(func)              \
  __device__ __forceinline__                      \
  __half func(__half const& x) { return ::h##func(x); }

// FIXME (trb): This is maybe not the best long-term solution, but it
// might be the best we can do without really digging into
// half-precision implementation.
#define WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(func) \
  __device__ __forceinline__                       \
  __half func(__half const& x) { return func(float(x)); }

WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(round)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(ceil)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(floor)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(sqrt)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(rsqrt)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(exp)
//WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(expm1)
//
// FIXME (trb): This is not going to be as accurate as a native expm1
// implementation could be:
__device__ __forceinline__
__half expm1(__half const& x) {
    return ::__hsub(::hexp(x), ::__float2half(1.f));
}

WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(log)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(log1p)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(cos)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(sin)
//WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(tan)
//
// FIXME (trb): This just uses the trig identity. Probably less
// accurate than a native implementation.
__device__ __forceinline__
__half tan(__half const& x) { return ::__hdiv(::hsin(x), ::hcos(x)); }

WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(acos)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(asin)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(atan)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(cosh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(sinh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(tanh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(acosh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(asinh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(atanh)
#undef WRAP_UNARY_CUDA_HALF_MATH_FUNCTION

__device__ __forceinline__
__half min(const __half& x, const __half& y)
{ return ::__hle(x, y) ? x : y; }

 __device__ __forceinline__
__half max(const __half& x, const __half& y)
{ return ::__hle(x, y) ? y : x; }
#endif // __CUDA_ARCH__ >= 530

// Numeric limits
#ifdef __CUDACC_RELAXED_CONSTEXPR__
template <typename T> constexpr __device__ __forceinline__ T min() {
  return std::numeric_limits<T>::min();
}
template <typename T> constexpr __device__ __forceinline__ T max() {
  return std::numeric_limits<T>::min();
}
template <typename T> constexpr __device__ __forceinline__ T epsilon() {
  return std::numeric_limits<T>::epsilon();
}
template <typename T> __device__ __forceinline__ T infinity() {
  return std::numeric_limits<T>::infinity();
}
#else // __CUDACC_RELAXED_CONSTEXPR__
#define SPECIFIERS template <> __device__ __forceinline__
SPECIFIERS constexpr float min<float>()                 { return FLT_MIN;   }
SPECIFIERS constexpr double min<double>()               { return DBL_MIN;   }
SPECIFIERS constexpr int min<int>()                     { return INT_MIN;   }
SPECIFIERS constexpr long int min<long int>()           { return LONG_MIN;  }
SPECIFIERS constexpr long long int min<long long int>() { return LLONG_MIN; }
SPECIFIERS constexpr float max<float>()                 { return FLT_MAX;   }
SPECIFIERS constexpr double max<double>()               { return DBL_MAX;   }
SPECIFIERS constexpr int max<int>()                     { return INT_MAX;   }
SPECIFIERS constexpr long int max<long int>()           { return LONG_MAX;  }
SPECIFIERS constexpr long long int max<long long int>() { return LLONG_MAX; }
SPECIFIERS constexpr float epsilon<float>()   { return FLT_EPSILON; }
SPECIFIERS constexpr double epsilon<double>() { return DBL_EPSILON; }
SPECIFIERS float infinity<float>()   { return CUDART_INF_F; }
SPECIFIERS double infinity<double>() { return CUDART_INF;   }
#undef SPECIFIERS
#endif // __CUDACC_RELAXED_CONSTEXPR__

#endif // __CUDACC__

// -------------------------------------------------------------
// Utilities for Thrust
// -------------------------------------------------------------
#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace thrust {
template <typename T>
allocator<T>::allocator(cudaStream_t stream)
  : m_stream(stream),
    m_system(stream) {}

template <typename T>
typename allocator<T>::pointer allocator<T>::allocate(allocator<T>::size_type size) {
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

template <typename T>
void allocator<T>::deallocate(allocator<T>::pointer buffer,
                              allocator<T>::size_type size) {
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

template <typename T>
typename allocator<T>::system_type& allocator<T>::system() {
  return m_system;
}

} // namespace thrust
#endif // !DOXYGEN_SHOULD_SKIP_THIS

} // namespace cuda
} // namespace lbann
