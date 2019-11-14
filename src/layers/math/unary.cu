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

#define LBANN_UNARY_LAYER_INSTANTIATE
#include "lbann/layers/math/unary.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {

namespace {

// =========================================================
// Operator objects for entry-wise unary layers
// =========================================================
// Note: Unary operator corresponds to forward prop step
// (\f$ y = f(x) \f$) and binary operator corresponds to
// back prop step
// (\f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$).

/** Logical not operator. */
template <typename TensorDataType>
struct logical_not_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    const auto& b = x != TensorDataType(0) && !isnan(x);
    return !b ? TensorDataType(1) : TensorDataType(0);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0);
  }
};

/** Absolute value operator. */
template <typename TensorDataType>
struct abs_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::abs(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    constexpr TensorDataType zero = 0;
    if      (x > zero) { return dy;   }
    else if (x < zero) { return -dy;  }
    else               { return zero; }
  }
};

/** Negative operator. */
template <typename TensorDataType>
struct negative_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return -x;
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy;
  }
};

/** Sign operator. */
template <typename TensorDataType>
struct sign_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    constexpr TensorDataType zero = 0;
    constexpr TensorDataType one = 1;
    if      (x > zero) { return one;  }
    else if (x < zero) { return -one; }
    else               { return zero; }
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0);
  }
};

/** Round operator. */
template <typename TensorDataType>
struct round_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::round(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0);
  }
};

/** Ceiling operator. */
template <typename TensorDataType>
struct ceil_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::ceil(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0);
  }
};

/** Floor operator. */
template <typename TensorDataType>
struct floor_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::floor(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0);
  }
};

/** Reciprocal operator. */
template <typename TensorDataType>
struct reciprocal_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return 1 / x;
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    if (dy == TensorDataType(0)) { return TensorDataType(0); }
    else                   { return - dy / (x*x); }

  }
};

/** Square operator. */
template <typename TensorDataType>
struct square_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return x*x;
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return 2*x * dy;
  }
};


/** Square root operator. */
template <typename TensorDataType>
struct sqrt_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::sqrt(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (2 * cuda::sqrt(x));
  }
};

/** Reciprocal square root operator. */
template <typename TensorDataType>
struct rsqrt_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::rsqrt(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& s = cuda::sqrt(x);
    return - dy / (2 * x * s);
  }
};

/** Safe reciprocal operator.
 *  If a standard reciprocal produces an infinity or NaN, zero is
 *  output instead.
 */
template <typename TensorDataType>
struct safe_reciprocal_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    const auto& y = 1 / x;
    if (isfinite(y)) { return y; }
    else             { return TensorDataType(0); }
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& y = 1 / x;
    if (isfinite(y)) { return - dy * y*y; }
    else             { return TensorDataType(0); }
  }
};

/** Exponential operator. */
template <typename TensorDataType>
struct exp_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::exp(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * cuda::exp(x);
  }
};

/** Exponential minus one operator. */
template <typename TensorDataType>
struct expm1_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::expm1(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * cuda::exp(x);
  }
};

/** Natural logarithm operator. */
template <typename TensorDataType>
struct log_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::log(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / x;
  }
};

/** Natural logarithm one plus operator. */
template <typename TensorDataType>
struct log1p_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::log1p(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (x + TensorDataType(1));
  }
};

/** Cosine operator. */
template <typename TensorDataType>
struct cos_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::cos(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy * cuda::sin(x);
  }
};

/** Sine operator. */
template <typename TensorDataType>
struct sin_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::sin(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * cuda::cos(x);
  }
};

/** Tangent operator. */
template <typename TensorDataType>
struct tan_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::tan(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& c = cuda::cos(x);
    return dy / (c*c);
  }
};

/** Arccosine operator. */
template <typename TensorDataType>
struct acos_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::acos(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy / cuda::sqrt(TensorDataType(1) - x*x);
  }
};

/** Arcsine operator. */
template <typename TensorDataType>
struct asin_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::asin(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / cuda::sqrt(TensorDataType(1) - x*x);
  }
};

/** Arctangent operator. */
template <typename TensorDataType>
struct atan_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::atan(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (TensorDataType(1) + x*x);
  }
};

/** Hyperbolic cosine operator. */
template <typename TensorDataType>
struct cosh_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::cosh(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * cuda::sinh(x);
  }
};

/** Hyperbolic sine operator. */
template <typename TensorDataType>
struct sinh_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::sinh(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * cuda::cosh(x);
  }
};

/** Hyperbolic tangent operator. */
template <typename TensorDataType>
struct tanh_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::tanh(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& c = cuda::cosh(x);
    return dy / (c*c);
  }
};

/** Hyperbolic arccosine operator. */
template <typename TensorDataType>
struct acosh_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::acosh(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy / (cuda::sqrt(x - TensorDataType(1)) * cuda::sqrt(x + TensorDataType(1)));
  }
};

/** Hyperbolic arcsine operator. */
template <typename TensorDataType>
struct asinh_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::asinh(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / cuda::sqrt(TensorDataType(1) + x*x);
  }
};

/** Hyperbolic arctangent operator. */
template <typename TensorDataType>
struct atanh_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::atanh(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (TensorDataType(1) - x*x);
  }
};

} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                                                          \
  template <typename TensorDataType>                                                                    \
  void fp_compute_impl(layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::GPU>& l) {        \
    cuda::apply_entrywise_unary_operator<TensorDataType, op<TensorDataType>>(l.get_prev_activations(),  \
                                                             l.get_activations());                      \
  }                                                                                                     \
  template <typename TensorDataType>                                                                    \
  void bp_compute_impl(layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::GPU>& l) {        \
    cuda::apply_entrywise_binary_operator<TensorDataType, op<TensorDataType>>(l.get_prev_activations(), \
                                                              l.get_prev_error_signals(),               \
                                                              l.get_error_signals());                   \
  }                                                                                                     \
  template <typename TensorDataType>                                                                    \
  void fp_compute_impl(layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l) {         \
    cuda::apply_entrywise_unary_operator<TensorDataType, op<TensorDataType>>(l.get_prev_activations(),  \
                                                             l.get_activations());                      \
  }                                                                                                     \
  template <typename TensorDataType>                                                                    \
  void bp_compute_impl(layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l) {         \
    cuda::apply_entrywise_binary_operator<TensorDataType, op<TensorDataType>>(l.get_prev_activations(), \
                                                              l.get_prev_error_signals(),               \
                                                              l.get_error_signals());                   \
  }                                                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device>                             \
  void layer<TensorDataType, Layout, Device>::fp_compute() {                                            \
    fp_compute_impl<TensorDataType>(*this);                                                             \
  }                                                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device>                             \
  void layer<TensorDataType, Layout, Device>::bp_compute() {                                            \
    bp_compute_impl<TensorDataType>(*this);                                                             \
  }                                                                                                     \
  UNARY_ETI_INST_MACRO_DEV(layer, El::Device::GPU)

INSTANTIATE(logical_not_layer, logical_not_op);
INSTANTIATE(abs_layer, abs_op);
INSTANTIATE(negative_layer, negative_op);
INSTANTIATE(sign_layer, sign_op);
INSTANTIATE(round_layer, round_op);
INSTANTIATE(ceil_layer, ceil_op);
INSTANTIATE(floor_layer, floor_op);
INSTANTIATE(reciprocal_layer, reciprocal_op);
INSTANTIATE(square_layer, square_op);
INSTANTIATE(sqrt_layer, sqrt_op);
INSTANTIATE(safe_reciprocal_layer, safe_reciprocal_op);
INSTANTIATE(rsqrt_layer, rsqrt_op);
INSTANTIATE(exp_layer, exp_op);
INSTANTIATE(expm1_layer, expm1_op);
INSTANTIATE(log_layer, log_op);
INSTANTIATE(log1p_layer, log1p_op);
INSTANTIATE(cos_layer, cos_op);
INSTANTIATE(sin_layer, sin_op);
INSTANTIATE(tan_layer, tan_op);
INSTANTIATE(acos_layer, acos_op);
INSTANTIATE(asin_layer, asin_op);
INSTANTIATE(atan_layer, atan_op);
INSTANTIATE(cosh_layer, cosh_op);
INSTANTIATE(sinh_layer, sinh_op);
INSTANTIATE(tanh_layer, tanh_op);
INSTANTIATE(acosh_layer, acosh_op);
INSTANTIATE(asinh_layer, asinh_op);
INSTANTIATE(atanh_layer, atanh_op);

} // namespace lbann
