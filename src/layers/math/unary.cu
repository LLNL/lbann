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

#include "lbann/layers/math/unary.hpp"

namespace lbann {

namespace {

// Wrappers for CUDA math API functions
// Note: For example, the CUDA math API provides the 'sqrtf' function
// for floats and 'sqrt' function for doubles. We wrap these with the
// overloaded function 'sqrt_'.
#define WRAP_CUDA_MATH_FUNCTION(func)                                   \
  __device__ __forceinline__ float func##_(const float& x) {            \
    static_cast<void>(static_cast<float (*)(const float&)>(func##_));   \
    return func##f(x);                                                  \
  }                                                                     \
  __device__ __forceinline__ double func##_(const double& x) {          \
    static_cast<void>(static_cast<double (*)(const double&)>(func##_)); \
    return func(x);                                                     \
  }
WRAP_CUDA_MATH_FUNCTION(fabs)
WRAP_CUDA_MATH_FUNCTION(round)
WRAP_CUDA_MATH_FUNCTION(ceil)
WRAP_CUDA_MATH_FUNCTION(floor)
WRAP_CUDA_MATH_FUNCTION(sqrt)
WRAP_CUDA_MATH_FUNCTION(exp)
WRAP_CUDA_MATH_FUNCTION(expm1)
WRAP_CUDA_MATH_FUNCTION(log)
WRAP_CUDA_MATH_FUNCTION(log1p)
WRAP_CUDA_MATH_FUNCTION(cos)
WRAP_CUDA_MATH_FUNCTION(sin)
WRAP_CUDA_MATH_FUNCTION(tan)
WRAP_CUDA_MATH_FUNCTION(acos)
WRAP_CUDA_MATH_FUNCTION(asin)
WRAP_CUDA_MATH_FUNCTION(atan)
WRAP_CUDA_MATH_FUNCTION(cosh)
WRAP_CUDA_MATH_FUNCTION(sinh)
WRAP_CUDA_MATH_FUNCTION(tanh)
WRAP_CUDA_MATH_FUNCTION(acosh)
WRAP_CUDA_MATH_FUNCTION(asinh)
WRAP_CUDA_MATH_FUNCTION(atanh)
  
// =========================================================
// Operator objects for entry-wise unary layers
// =========================================================
// Note: Unary operator corresponds to forward prop step
// (\f$ y = f(x) \f$) and binary operator corresponds to 
// back prop step
// (\f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$).

/** Logical not operator. */
struct not_op {
  inline __device__ DataType operator()(const DataType& x) const {
    const bool b = x != DataType(0) && !isnan(x);
    return !b ? DataType(1) : DataType(0);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return DataType(0);
  }
};
  
/** Absolute value operator. */
struct abs_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return fabs_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    constexpr DataType zero = 0;
    if      (x > zero) { return dy;   }
    else if (x < zero) { return -dy;  }
    else               { return zero; }
  }
};

/** Negative operator. */
struct negative_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return -x;
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return -dy;
  }
};

/** Sign operator. */
struct sign_op {
  inline __device__ DataType operator()(const DataType& x) const {
    constexpr DataType zero = 0;
    constexpr DataType one = 1;
    if      (x > zero) { return one;  }
    else if (x < zero) { return -one; }
    else               { return zero; }
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return DataType(0);
  }
};

/** Round operator. */
struct round_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return round_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return DataType(0);
  }
};

/** Ceiling operator. */
struct ceil_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return ceil_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return DataType(0);
  }
};

/** Floor operator. */
struct floor_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return floor_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return DataType(0);
  }
};

/** Reciprocal operator. */
struct reciprocal_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return 1 / x;
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    if (dy == DataType(0)) { return DataType(0); }
    else                   { return - dy / (x*x); }
    
  }
};

/** Square operator. */
struct square_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return x*x;
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return 2*x * dy;
  }
};


/** Square root operator. */
struct sqrt_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return sqrt_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (2 * sqrt_(x));
  }
};

/** Reciprocal square root operator. */
struct rsqrt_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return 1 / sqrt_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& s = sqrt_(x);
    return - dy / (2 * s*s*s);
  }
};

/** Exponential operator. */
struct exp_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return exp_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * exp_(x);
  }
};

/** Exponential minus one operator. */
struct expm1_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return expm1_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * exp_(x);
  }
};

/** Natural logarithm operator. */
struct log_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return log_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / x;
  }
};

/** Natural logarithm one plus operator. */
struct log1p_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return log1p_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (x + DataType(1));
  }
};

/** Cosine operator. */
struct cos_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return cos_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return -dy * sin_(x);
  }
};

/** Sine operator. */
struct sin_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return sin_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * cos_(x);
  }
};

/** Tangent operator. */
struct tan_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return tan_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& c = cos_(x);
    return dy / (c*c);
  }
};

/** Arccosine operator. */
struct acos_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return acos_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return -dy / sqrt_(DataType(1) - x*x);
  }
};

/** Arcsine operator. */
struct asin_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return asin_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / sqrt_(DataType(1) - x*x);
  }
};

/** Arctangent operator. */
struct atan_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return atan_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (DataType(1) + x*x);
  }
};

/** Hyperbolic cosine operator. */
struct cosh_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return cosh_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * sinh_(x);
  }
};

/** Hyperbolic sine operator. */
struct sinh_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return sinh_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * cosh_(x);
  }
};

/** Hyperbolic tangent operator. */
struct tanh_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return tanh_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& c = cosh_(x);
    return dy / (c*c);
  }
};

/** Hyperbolic arccosine operator. */
struct acosh_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return acosh_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return -dy / (sqrt_(x - DataType(1)) * sqrt_(x + DataType(1)));
  }
};

/** Hyperbolic arcsine operator. */
struct asinh_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return asinh_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / sqrt_(DataType(1) + x*x);
  }
};

/** Hyperbolic arctangent operator. */
struct atanh_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return atanh_(x);
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (DataType(1) - x*x);
  }
};
  
} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                          \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::GPU>              \
  ::fp_compute() {                                                      \
    cuda::apply_entrywise_unary_operator<op>(get_prev_activations(),    \
                                             get_activations());        \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::GPU>              \
  ::bp_compute() {                                                      \
    cuda::apply_entrywise_binary_operator<op>(get_prev_activations(),   \
                                              get_prev_error_signals(), \
                                              get_error_signals());     \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::GPU>               \
  ::fp_compute() {                                                      \
    cuda::apply_entrywise_unary_operator<op>(get_prev_activations(),    \
                                             get_activations());        \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::GPU>               \
  ::bp_compute() {                                                      \
    cuda::apply_entrywise_binary_operator<op>(get_prev_activations(),   \
                                              get_prev_error_signals(), \
                                              get_error_signals());     \
  }
  INSTANTIATE(not_layer, not_op)
  INSTANTIATE(abs_layer, abs_op)
  INSTANTIATE(negative_layer, negative_op)
  INSTANTIATE(sign_layer, sign_op)
  INSTANTIATE(round_layer, round_op)
  INSTANTIATE(ceil_layer, ceil_op)
  INSTANTIATE(floor_layer, floor_op)
  INSTANTIATE(reciprocal_layer, reciprocal_op)
  INSTANTIATE(square_layer, square_op)
  INSTANTIATE(sqrt_layer, sqrt_op)
  INSTANTIATE(rsqrt_layer, rsqrt_op)
  INSTANTIATE(exp_layer, exp_op)
  INSTANTIATE(expm1_layer, expm1_op)
  INSTANTIATE(log_layer, log_op)
  INSTANTIATE(log1p_layer, log1p_op)
  INSTANTIATE(cos_layer, cos_op)
  INSTANTIATE(sin_layer, sin_op)
  INSTANTIATE(tan_layer, tan_op)
  INSTANTIATE(acos_layer, acos_op)
  INSTANTIATE(asin_layer, asin_op)
  INSTANTIATE(atan_layer, atan_op)
  INSTANTIATE(cosh_layer, cosh_op)
  INSTANTIATE(sinh_layer, sinh_op)
  INSTANTIATE(tanh_layer, tanh_op)
  INSTANTIATE(acosh_layer, acosh_op)
  INSTANTIATE(asinh_layer, asinh_op)
  INSTANTIATE(atanh_layer, atanh_op)
  
} // namespace lbann
