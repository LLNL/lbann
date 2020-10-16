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

#include <thrust/system/hip/execution_policy.h>

// Headers for HIP
#ifdef __HIPCC__
#ifdef HYDROGEN_HAVE_CUB
#include "hipcub/block/block_reduce.hpp"
#endif // HYDROGEN_HAVE_CUB
#include <limits>
#include <hip/hip_fp16.h>
#endif // __HIPCC__

namespace lbann {

// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------
#ifdef __HIPCC__
// Atomic add function
__device__ __forceinline__
__half gpu_lib::atomic_add(__half* address, __half val) {
  return atomicAdd(address, val);
}
__device__ __forceinline__
float gpu_lib::atomic_add(float* address, float val) {
  return atomicAdd(address, val);
}
__device__ __forceinline__
double gpu_lib::atomic_add(double* address, double val) {
  return atomicAdd(address, val);
}

// This support is far from complete!
#define WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(func)              \
  __device__ __forceinline__                      \
  __half gpu_lib::func(__half const& x) { return ::h##func(x); }

// FIXME (trb): This is maybe not the best long-term solution, but it
// might be the best we can do without really digging into
// half-precision implementation.
#define WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(func) \
  __device__ __forceinline__                       \
  __half gpu_lib::func(__half const& x) { return ::func(float(x)); }

WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(round)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(ceil)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(floor)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(sqrt)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(rsqrt)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(exp)
//WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(expm1)
//
// FIXME (trb): This is not going to be as accurate as a native expm1
// implementation could be:
__device__ __forceinline__
__half gpu_lib::expm1(__half const& x) {
    return ::__hsub(::hexp(x), ::__float2half(1.f));
}

WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(log)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(log1p)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(cos)
WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(sin)
//WRAP_UNARY_ROCM_HALF_MATH_FUNCTION(tan)
//
// FIXME (trb): This just uses the trig identity. Probably less
// accurate than a native implementation.
__device__ __forceinline__
__half gpu_lib::tan(__half const& x) { return ::__hdiv(::hsin(x), ::hcos(x)); }

WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(acos)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(asin)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(atan)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(cosh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(sinh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(tanh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(acosh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(asinh)
WRAP_UNARY_ROCM_HALF_CAST_TO_FLOAT_MATH_FUNCTION(atanh)
#undef WRAP_UNARY_ROCM_HALF_MATH_FUNCTION

__device__ __forceinline__
__half gpu_lib::min(const __half& x, const __half& y)
{ return ::__hle(x, y) ? x : y; }

 __device__ __forceinline__
__half gpu_lib::max(const __half& x, const __half& y)
{ return ::__hle(x, y) ? y : x; }

// Numeric limits
#define SPECIFIERS template <> __device__ __forceinline__
SPECIFIERS constexpr float gpu_lib::min<float>()                 { return FLT_MIN;   }
SPECIFIERS constexpr double gpu_lib::min<double>()               { return DBL_MIN;   }
SPECIFIERS constexpr int gpu_lib::min<int>()                     { return INT_MIN;   }
SPECIFIERS constexpr long int gpu_lib::min<long int>()           { return LONG_MIN;  }
SPECIFIERS constexpr long long int gpu_lib::min<long long int>() { return LLONG_MIN; }
SPECIFIERS constexpr float gpu_lib::max<float>()                 { return FLT_MAX;   }
SPECIFIERS constexpr double gpu_lib::max<double>()               { return DBL_MAX;   }
SPECIFIERS constexpr int gpu_lib::max<int>()                     { return INT_MAX;   }
SPECIFIERS constexpr long int gpu_lib::max<long int>()           { return LONG_MAX;  }
SPECIFIERS constexpr long long int gpu_lib::max<long long int>() { return LLONG_MAX; }
SPECIFIERS constexpr float gpu_lib::epsilon<float>()   { return FLT_EPSILON; }
SPECIFIERS constexpr double gpu_lib::epsilon<double>() { return DBL_EPSILON; }
SPECIFIERS float gpu_lib::infinity<float>()   { return HIPRT_INF_F; }
SPECIFIERS double gpu_lib::infinity<double>() { return HIPRT_INF;   }
#undef SPECIFIERS

#endif // __HIPCC__

namespace rocm {

// -------------------------------------------------------------
// Utilities for Thrust
// -------------------------------------------------------------
#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace thrust {
template <typename T>
allocator<T>::allocator(hipStream_t stream)
  : m_stream(stream),
    m_system(stream) {}

template <typename T>
typename allocator<T>::pointer allocator<T>::allocate(allocator<T>::size_type size) {
  value_type* buffer = nullptr;
  if (size > 0) {
#ifdef HYDROGEN_HAVE_CUB
    auto& memory_pool = El::hipcub::MemoryPool();
    CHECK_CUDA(memory_pool.DeviceAllocate(reinterpret_cast<void**>(&buffer),
                                          size * sizeof(value_type),
                                          m_stream));
#else
    CHECK_CUDA(hipMalloc(&buffer, size * sizeof(value_type)));
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
    auto& memory_pool = El::hipcub::MemoryPool();
    CHECK_CUDA(memory_pool.DeviceFree(ptr));
#else
    CHECK_CUDA(hipFree(ptr));
#endif // HYDROGEN_HAVE_CUB
  }
}

template <typename T>
typename allocator<T>::system_type& allocator<T>::system() {
  return m_system;
}

} // namespace thrust
#endif // !DOXYGEN_SHOULD_SKIP_THIS

} // namespace rocm
} // namespace lbann
