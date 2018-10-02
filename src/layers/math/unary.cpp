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
#include "lbann/utils/entrywise_operator.hpp"

namespace lbann {

namespace {

// Helpful constants
constexpr DataType zero = 0;
constexpr DataType one = 1;
  
// =========================================================
// Operator objects for entry-wise unary layers
// =========================================================
// Note: Unary operator corresponds to forward prop step
// (\f$ y = f(x) \f$) and binary operator corresponds to 
// back prop step
// (\f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$).

/** Logical not operator. */
struct not_op {
  inline DataType operator()(const DataType& x) const {
    const bool b = x != zero && !std::isnan(x);
    return !b ? one : zero;
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return zero;
  }
};
  
/** Absolute value operator. */
struct abs_op {
  inline DataType operator()(const DataType& x) const {
    return x >= zero ? x : -x;
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    if      (x > zero) { return dy;   }
    else if (x < zero) { return -dy;  }
    else               { return zero; }
  }
};

/** Negative operator. */
struct negative_op {
  inline DataType operator()(const DataType& x) const {
    return -x;
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return -dy;
  }
};

/** Sign operator. */
struct sign_op {
  inline DataType operator()(const DataType& x) const {
    if      (x > zero) { return one;  }
    else if (x < zero) { return -one; }
    else               { return zero; }
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return zero;
  }
};

/** Round operator. */
struct round_op {
  inline DataType operator()(const DataType& x) const {
    return std::round(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return zero;
  }
};

/** Ceiling operator. */
struct ceil_op {
  inline DataType operator()(const DataType& x) const {
    return std::ceil(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return zero;
  }
};

/** Floor operator. */
struct floor_op {
  inline DataType operator()(const DataType& x) const {
    return std::floor(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return zero;
  }
};

/** Reciprocal operator. */
struct reciprocal_op {
  inline DataType operator()(const DataType& x) const {
    return 1 / x;
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    if (dy == zero) { return zero; }
    else            { return - dy / (x*x); }
  }
};

/** Square operator. */
struct square_op {
  inline DataType operator()(const DataType& x) const {
    return x*x;
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return 2*x * dy;
  }
};


/** Square root operator. */
struct sqrt_op {
  inline DataType operator()(const DataType& x) const {
    return std::sqrt(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (2 * std::sqrt(x));
  }
};

/** Reciprocal square root operator. */
struct rsqrt_op {
  inline DataType operator()(const DataType& x) const {
    return 1 / std::sqrt(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& s = std::sqrt(x);
    return - dy / (2 * s*s*s);
  }
};

/** Safe reciprocal operator. */
struct safe_reciprocal_op {
  inline DataType operator()(const DataType& x) const {
    const auto& y = 1 / x;
    if (std::isfinite(y)) { return y; }
    else                  { return zero; }
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& y = 1 / x;
    if (std::isfinite(y)) { return - dy * y*y; }
    else                  { return zero; }
  }
};
  
/** Exponential operator. */
struct exp_op {
  inline DataType operator()(const DataType& x) const {
    return std::exp(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * std::exp(x);
  }
};

/** Exponential minus one operator. */
struct expm1_op {
  inline DataType operator()(const DataType& x) const {
    return std::expm1(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * std::exp(x);
  }
};

/** Natural logarithm operator. */
struct log_op {
  inline DataType operator()(const DataType& x) const {
    return std::log(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / x;
  }
};

/** Natural logarithm one plus operator. */
struct log1p_op {
  inline DataType operator()(const DataType& x) const {
    return std::log1p(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (x + one);
  }
};

/** Cosine operator. */
struct cos_op {
  inline DataType operator()(const DataType& x) const {
    return std::cos(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return -dy * std::sin(x);
  }
};

/** Sine operator. */
struct sin_op {
  inline DataType operator()(const DataType& x) const {
    return std::sin(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * std::cos(x);
  }
};

/** Tangent operator. */
struct tan_op {
  inline DataType operator()(const DataType& x) const {
    return std::tan(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& c = std::cos(x);
    return dy / (c*c);
  }
};

/** Arccosine operator. */
struct acos_op {
  inline DataType operator()(const DataType& x) const {
    return std::acos(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return -dy / std::sqrt(one - x*x);
  }
};

/** Arcsine operator. */
struct asin_op {
  inline DataType operator()(const DataType& x) const {
    return std::asin(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / std::sqrt(one - x*x);
  }
};

/** Arctangent operator. */
struct atan_op {
  inline DataType operator()(const DataType& x) const {
    return std::atan(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (one + x*x);
  }
};

/** Hyperbolic cosine operator. */
struct cosh_op {
  inline DataType operator()(const DataType& x) const {
    return std::cosh(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * std::sinh(x);
  }
};

/** Hyperbolic sine operator. */
struct sinh_op {
  inline DataType operator()(const DataType& x) const {
    return std::sinh(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy * std::cosh(x);
  }
};

/** Hyperbolic tangent operator. */
struct tanh_op {
  inline DataType operator()(const DataType& x) const {
    return std::tanh(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& c = std::cosh(x);
    return dy / (c*c);
  }
};

/** Hyperbolic arccosine operator. */
struct acosh_op {
  inline DataType operator()(const DataType& x) const {
    return std::acosh(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return -dy / (std::sqrt(x - one) * std::sqrt(x + one));
  }
};

/** Hyperbolic arcsine operator. */
struct asinh_op {
  inline DataType operator()(const DataType& x) const {
    return std::asinh(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / std::sqrt(one + x*x);
  }
};

/** Hyperbolic arctangent operator. */
struct atanh_op {
  inline DataType operator()(const DataType& x) const {
    return std::atanh(x);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (one - x*x);
  }
};
  
} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                          \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::CPU>              \
         ::fp_compute() {                                               \
    apply_entrywise_unary_operator<op>(get_prev_activations(),          \
                                       get_activations());              \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::CPU>              \
         ::bp_compute() {                                               \
    apply_entrywise_binary_operator<op>(get_prev_activations(),         \
                                        get_prev_error_signals(),       \
                                        get_error_signals());           \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::CPU>               \
         ::fp_compute() {                                               \
    apply_entrywise_unary_operator<op>(get_prev_activations(),          \
                                       get_activations());              \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::CPU>               \
         ::bp_compute() {                                               \
    apply_entrywise_binary_operator<op>(get_prev_activations(),         \
                                        get_prev_error_signals(),       \
                                        get_error_signals());           \
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
  INSTANTIATE(safe_reciprocal_layer, safe_reciprocal_op)
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
