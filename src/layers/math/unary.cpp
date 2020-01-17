///////////////////////////////////////////////////////////////////////////////
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
#include "lbann/utils/entrywise_operator.hpp"

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
  inline TensorDataType operator()(const TensorDataType& x) const {
    const auto& b = x != El::TypeTraits<TensorDataType>::Zero() && !std::isnan(x);
    return !b ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Absolute value operator. */
template <typename TensorDataType>
struct abs_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return x >= El::TypeTraits<TensorDataType>::Zero() ? x : -x;
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    if      (x > El::TypeTraits<TensorDataType>::Zero()) { return dy;   }
    else if (x < El::TypeTraits<TensorDataType>::Zero()) { return -dy;  }
    else               { return El::TypeTraits<TensorDataType>::Zero(); }
  }
};

/** Negative operator. */
template <typename TensorDataType>
struct negative_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return -x;
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy;
  }
};

/** Sign operator. */
template <typename TensorDataType>
struct sign_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    if      (x > El::TypeTraits<TensorDataType>::Zero()) { return El::TypeTraits<TensorDataType>::One();  }
    else if (x < El::TypeTraits<TensorDataType>::Zero()) { return -El::TypeTraits<TensorDataType>::One(); }
    else               { return El::TypeTraits<TensorDataType>::Zero(); }
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Round operator. */
template <typename TensorDataType>
struct round_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    using std::round;
    return round(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Ceiling operator. */
template <typename TensorDataType>
struct ceil_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    using std::ceil;
    return ceil(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Floor operator. */
template <typename TensorDataType>
struct floor_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    using std::floor;
    return floor(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Reciprocal operator.
 *  If a standard reciprocal produces an infinity or NaN, El::TypeTraits<TensorDataType>::Zero() is
 *  output instead.
 */
template <typename TensorDataType>
struct reciprocal_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::To<TensorDataType>(1) / x;
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    if (dy == El::TypeTraits<TensorDataType>::Zero()) { return El::TypeTraits<TensorDataType>::Zero(); }
    else            { return - dy / (x*x); }
  }
};

/** Square operator. */
template <typename TensorDataType>
struct square_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return x*x;
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::To<TensorDataType>(2)*x * dy;
  }
};


/** Square root operator. */
template <typename TensorDataType>
struct sqrt_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Sqrt(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (El::To<TensorDataType>(2) * El::Sqrt(x));
  }
};

/** Reciprocal square root operator. */
template <typename TensorDataType>
struct rsqrt_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::To<TensorDataType>(1) / El::Sqrt(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& s = El::Sqrt(x);
    return - dy / (El::To<TensorDataType>(2) * x * s);
  }
};

/** Safe reciprocal operator. */
template <typename TensorDataType>
struct safe_reciprocal_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    const auto& y = El::To<TensorDataType>(1) / x;
    if (std::isfinite(y)) { return y; }
    else                  { return El::TypeTraits<TensorDataType>::Zero(); }
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& y = El::To<TensorDataType>(1) / x;
    if (std::isfinite(y)) { return - dy * y*y; }
    else                  { return El::TypeTraits<TensorDataType>::Zero(); }
  }
};

/** Exponential operator. */
template <typename TensorDataType>
struct exp_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Exp(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * El::Exp(x);
  }
};

/** Exponential minus one operator. */
template <typename TensorDataType>
struct expm1_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    using std::expm1;
    return expm1(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * El::Exp(x);
  }
};

/** Natural logarithm operator. */
template <typename TensorDataType>
struct log_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Log(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / x;
  }
};

/** Natural logarithm one plus operator. */
template <typename TensorDataType>
struct log1p_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    using std::log1p;
    return log1p(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (x + El::TypeTraits<TensorDataType>::One());
  }
};

/** Cosine operator. */
template <typename TensorDataType>
struct cos_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Cos(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy * El::Sin(x);
  }
};

/** Sine operator. */
template <typename TensorDataType>
struct sin_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Sin(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * El::Cos(x);
  }
};

/** Tangent operator. */
template <typename TensorDataType>
struct tan_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Tan(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& c = El::Cos(x);
    return dy / (c*c);
  }
};

/** Arccosine operator. */
template <typename TensorDataType>
struct acos_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Acos(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy / El::Sqrt(El::TypeTraits<TensorDataType>::One() - x*x);
  }
};

/** Arcsine operator. */
template <typename TensorDataType>
struct asin_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Asin(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / El::Sqrt(El::TypeTraits<TensorDataType>::One() - x*x);
  }
};

/** Arctangent operator. */
template <typename TensorDataType>
struct atan_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Atan(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (El::TypeTraits<TensorDataType>::One() + x*x);
  }
};

/** Hyperbolic cosine operator. */
template <typename TensorDataType>
struct cosh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Cosh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * El::Sinh(x);
  }
};

/** Hyperbolic sine operator. */
template <typename TensorDataType>
struct sinh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Sinh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * El::Cosh(x);
  }
};

/** Hyperbolic tangent operator. */
template <typename TensorDataType>
struct tanh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Tanh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& c = El::Cosh(x);
    return dy / (c*c);
  }
};

/** Hyperbolic arccosine operator. */
template <typename TensorDataType>
struct acosh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Acosh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy / (El::Sqrt(x - El::TypeTraits<TensorDataType>::One()) * El::Sqrt(x + El::TypeTraits<TensorDataType>::One()));
  }
};

/** Hyperbolic arcsine operator. */
template <typename TensorDataType>
struct asinh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Asinh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / El::Sqrt(El::TypeTraits<TensorDataType>::One() + x*x);
  }
};

/** Hyperbolic arctangent operator. */
template <typename TensorDataType>
struct atanh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Atanh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (El::TypeTraits<TensorDataType>::One() - x*x);
  }
};

} // namespace

// Template instantiation
#define DEFINE_COMPUTE_OPS(layer, op)                                   \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::fp_compute() {            \
      apply_entrywise_unary_operator<op>(                               \
        this->get_prev_activations(),                                   \
    this->get_activations());                                           \
  }                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::bp_compute() {            \
    apply_entrywise_binary_operator<op>(                                \
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
  UNARY_ETI_INST_MACRO_DEV_DT(logical_not_layer, T, El::Device::CPU); \
  UNARY_ETI_INST_MACRO_DEV_DT(abs_layer, T, El::Device::CPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(negative_layer, T, El::Device::CPU);    \
  UNARY_ETI_INST_MACRO_DEV_DT(sign_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(round_layer, T, El::Device::CPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(ceil_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(floor_layer, T, El::Device::CPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(reciprocal_layer, T, El::Device::CPU);  \
  UNARY_ETI_INST_MACRO_DEV_DT(square_layer, T, El::Device::CPU);      \
  UNARY_ETI_INST_MACRO_DEV_DT(sqrt_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(rsqrt_layer, T, El::Device::CPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(safe_reciprocal_layer, T, El::Device::CPU); \
  UNARY_ETI_INST_MACRO_DEV_DT(exp_layer, T, El::Device::CPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(expm1_layer, T, El::Device::CPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(log_layer, T, El::Device::CPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(log1p_layer, T, El::Device::CPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(cos_layer, T, El::Device::CPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(sin_layer, T, El::Device::CPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(tan_layer, T, El::Device::CPU);         \
  UNARY_ETI_INST_MACRO_DEV_DT(acos_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(asin_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(atan_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(cosh_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(sinh_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(tanh_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(acosh_layer, T, El::Device::CPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(asinh_layer, T, El::Device::CPU);       \
  UNARY_ETI_INST_MACRO_DEV_DT(atanh_layer, T, El::Device::CPU)

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
