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
#include "lbann/utils/gpu/helpers.hpp"

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
    const auto& b = x != TensorDataType(0.0) && !cuda::isnan(x);
    return !b ? TensorDataType(1.0) : TensorDataType(0.0);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0.0);
  }
};

/** Absolute value operator. */
template <typename TensorDataType>
struct abs_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::abs(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const TensorDataType zero = 0.;
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
    const TensorDataType zero = 0.;
    const TensorDataType one = 1.;
    if      (x > zero) { return one;  }
    else if (x < zero) { return -one; }
    else               { return zero; }
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0.0);
  }
};

/** Round operator. */
template <typename TensorDataType>
struct round_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::round(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0.0);
  }
};

/** Ceiling operator. */
template <typename TensorDataType>
struct ceil_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::ceil(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0.0);
  }
};

/** Floor operator. */
template <typename TensorDataType>
struct floor_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::floor(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return TensorDataType(0.0);
  }
};

/** Reciprocal operator. */
template <typename TensorDataType>
struct reciprocal_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return TensorDataType(1.) / x;
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    if (dy == TensorDataType(0.0)) { return TensorDataType(0.0); }
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
    return TensorDataType(2.) * x * dy;
  }
};


/** Square root operator. */
template <typename TensorDataType>
struct sqrt_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::sqrt(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (TensorDataType(2.) * cuda::sqrt(x));
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
    return - dy / (TensorDataType(2.) * x * s);
  }
};

/** Safe reciprocal operator.
 *  If a standard reciprocal produces an infinity or NaN, zero is
 *  output instead.
 */
template <typename TensorDataType>
struct safe_reciprocal_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    const auto& y = TensorDataType(1.) / x;
    if (cuda::isfinite(y)) { return y; }
    else             { return TensorDataType(0.0); }
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& y = TensorDataType(1.) / x;
    if (cuda::isfinite(y)) { return - dy * y*y; }
    else             { return TensorDataType(0.0); }
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
    return dy / (x + TensorDataType(1.0));
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
    return -dy / cuda::sqrt(TensorDataType(1.0) - x*x);
  }
};

/** Arcsine operator. */
template <typename TensorDataType>
struct asin_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::asin(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / cuda::sqrt(TensorDataType(1.0) - x*x);
  }
};

/** Arctangent operator. */
template <typename TensorDataType>
struct atan_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::atan(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (TensorDataType(1.0) + x*x);
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
    return -dy / (cuda::sqrt(x - TensorDataType(1.0)) * cuda::sqrt(x + TensorDataType(1.0)));
  }
};

/** Hyperbolic arcsine operator. */
template <typename TensorDataType>
struct asinh_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::asinh(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / cuda::sqrt(TensorDataType(1.0) + x*x);
  }
};

/** Hyperbolic arctangent operator. */
template <typename TensorDataType>
struct atanh_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::atanh(x);
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (TensorDataType(1.0) - x*x);
  }
};

} // namespace

// Template instantiation
#define DEFINE_COMPUTE_OPS(layer, op)                                   \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::fp_compute() {            \
    cuda::apply_entrywise_unary_operator<op>(                           \
      this->get_prev_activations(),                                     \
      this->get_activations());                                         \
  }                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::bp_compute() {            \
    cuda::apply_entrywise_binary_operator<op>(                          \
      this->get_prev_activations(),                                     \
      this->get_prev_error_signals(),                                   \
      this->get_error_signals());                                       \
  }

DEFINE_COMPUTE_OPS(logical_not_layer, logical_not_op)
DEFINE_COMPUTE_OPS(abs_layer, abs_op)
DEFINE_COMPUTE_OPS(negative_layer, negative_op)
DEFINE_COMPUTE_OPS(sign_layer, sign_op)
DEFINE_COMPUTE_OPS(round_layer, round_op)
DEFINE_COMPUTE_OPS(ceil_layer, ceil_op)
DEFINE_COMPUTE_OPS(floor_layer, floor_op)
DEFINE_COMPUTE_OPS(reciprocal_layer, reciprocal_op)
DEFINE_COMPUTE_OPS(square_layer, square_op)
DEFINE_COMPUTE_OPS(sqrt_layer, sqrt_op)
DEFINE_COMPUTE_OPS(rsqrt_layer, rsqrt_op)
DEFINE_COMPUTE_OPS(safe_reciprocal_layer, safe_reciprocal_op)
DEFINE_COMPUTE_OPS(exp_layer, exp_op)
DEFINE_COMPUTE_OPS(expm1_layer, expm1_op)
DEFINE_COMPUTE_OPS(log_layer, log_op)
DEFINE_COMPUTE_OPS(log1p_layer, log1p_op)
DEFINE_COMPUTE_OPS(cos_layer, cos_op)
DEFINE_COMPUTE_OPS(sin_layer, sin_op)
DEFINE_COMPUTE_OPS(tan_layer, tan_op)
DEFINE_COMPUTE_OPS(acos_layer, acos_op)
DEFINE_COMPUTE_OPS(asin_layer, asin_op)
DEFINE_COMPUTE_OPS(atan_layer, atan_op)
DEFINE_COMPUTE_OPS(cosh_layer, cosh_op)
DEFINE_COMPUTE_OPS(sinh_layer, sinh_op)
DEFINE_COMPUTE_OPS(tanh_layer, tanh_op)
DEFINE_COMPUTE_OPS(acosh_layer, acosh_op)
DEFINE_COMPUTE_OPS(asinh_layer, asinh_op)
DEFINE_COMPUTE_OPS(atanh_layer, atanh_op)

#define PROTO(T) \
  UNARY_ETI_INST_MACRO_DEV_DT(logical_not_layer, T, El::Device::GPU); \
  UNARY_ETI_INST_MACRO_DEV_DT(abs_layer, T, El::Device::GPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(negative_layer, T, El::Device::GPU);    \
  UNARY_ETI_INST_MACRO_DEV_DT(sign_layer, T, El::Device::GPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(round_layer, T, El::Device::GPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(ceil_layer, T, El::Device::GPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(floor_layer, T, El::Device::GPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(reciprocal_layer, T, El::Device::GPU);  \
  UNARY_ETI_INST_MACRO_DEV_DT(square_layer, T, El::Device::GPU);      \
  UNARY_ETI_INST_MACRO_DEV_DT(sqrt_layer, T, El::Device::GPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(rsqrt_layer, T, El::Device::GPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(safe_reciprocal_layer, T, El::Device::GPU); \
  UNARY_ETI_INST_MACRO_DEV_DT(exp_layer, T, El::Device::GPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(expm1_layer, T, El::Device::GPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(log_layer, T, El::Device::GPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(log1p_layer, T, El::Device::GPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(cos_layer, T, El::Device::GPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(sin_layer, T, El::Device::GPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(tan_layer, T, El::Device::GPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(acos_layer, T, El::Device::GPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(asin_layer, T, El::Device::GPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(atan_layer, T, El::Device::GPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(cosh_layer, T, El::Device::GPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(sinh_layer, T, El::Device::GPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(tanh_layer, T, El::Device::GPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(acosh_layer, T, El::Device::GPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(asinh_layer, T, El::Device::GPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(atanh_layer, T, El::Device::GPU)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
