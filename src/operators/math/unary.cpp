///////////////////////////////////////////////////////////////////////////////
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

#include "lbann/operators/math/unary.hpp"

#include "common.hpp"

namespace lbann {
namespace {

// Operator implementations objects for entry-wise unary operators
//
// Note: Unary apply() corresponds to forward prop step
// (\f$ y = f(x) \f$) and binary apply() corresponds to
// back prop step
// (\f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$).

// Logical not operator.
template <typename DataT>
struct LogicalNotOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    const auto& b = x != El::TypeTraits<DataT>::Zero() && !std::isnan(x);
    return !b ? El::TypeTraits<DataT>::One() : El::TypeTraits<DataT>::Zero();
  }
  DataT operator()(DataT const& x, DataT const& /*dy*/) const noexcept
  {
    return El::TypeTraits<DataT>::Zero();
  }
};

// Negative operator.
template <typename DataT>
struct NegativeOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return -x; }
  DataT operator()(DataT const& /*x*/, DataT const& dy) const noexcept
  {
    return -dy;
  }
};

// Sign operator.
template <typename DataT>
struct SignOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    if (x > El::TypeTraits<DataT>::Zero()) {
      return El::TypeTraits<DataT>::One();
    }
    else if (x < El::TypeTraits<DataT>::Zero()) {
      return -El::TypeTraits<DataT>::One();
    }
    else {
      return El::TypeTraits<DataT>::Zero();
    }
  }
  DataT operator()(DataT const& /*x*/, DataT const& /*dy*/) const noexcept
  {
    return El::TypeTraits<DataT>::Zero();
  }
};

// Round operator.
template <typename DataT>
struct RoundOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    using std::round;
    return round(x);
  }
  DataT operator()(DataT const& /*x*/, DataT const& /*dy*/) const noexcept
  {
    return El::TypeTraits<DataT>::Zero();
  }
};

// Ceiling operator.
template <typename DataT>
struct CeilOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    using std::ceil;
    return ceil(x);
  }
  DataT operator()(DataT const& /*x*/, DataT const& /*dy*/) const noexcept
  {
    return El::TypeTraits<DataT>::Zero();
  }
};

// Floor operator.
template <typename DataT>
struct FloorOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    using std::floor;
    return floor(x);
  }
  DataT operator()(DataT const& /*x*/, DataT const& /*dy*/) const noexcept
  {
    return El::TypeTraits<DataT>::Zero();
  }
};

/** Reciprocal operator.
 *  If a standard reciprocal produces an infinity or NaN,
 * El::TypeTraits<DataT>::Zero() is output instead.
 */
template <typename DataT>
struct ReciprocalOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    return El::To<DataT>(1) / x;
  }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    if (dy == El::TypeTraits<DataT>::Zero()) {
      return El::TypeTraits<DataT>::Zero();
    }
    else {
      return -dy / (x * x);
    }
  }
};

// Square operator.
template <typename DataT>
struct SquareOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return x * x; }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return El::To<DataT>(2) * x * dy;
  }
};

// Square root operator.
template <typename DataT>
struct SqrtOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Sqrt(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy / (El::To<DataT>(2) * El::Sqrt(x));
  }
};

// Reciprocal square root operator.
template <typename DataT>
struct RsqrtOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    return El::To<DataT>(1) / El::Sqrt(x);
  }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    const auto& s = El::Sqrt(x);
    return -dy / (El::To<DataT>(2) * x * s);
  }
};

// Safe reciprocal operator.
template <typename DataT>
struct SafeReciprocalOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    const auto& y = El::To<DataT>(1) / x;
    if (std::isfinite(y)) {
      return y;
    }
    else {
      return El::TypeTraits<DataT>::Zero();
    }
  }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    const auto& y = El::To<DataT>(1) / x;
    if (std::isfinite(y)) {
      return -dy * y * y;
    }
    else {
      return El::TypeTraits<DataT>::Zero();
    }
  }
};

// Exponential operator.
template <typename DataT>
struct ExpOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Exp(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy * El::Exp(x);
  }
};

// Exponential minus one operator.
template <typename DataT>
struct Expm1OpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    using std::expm1;
    return expm1(x);
  }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy * El::Exp(x);
  }
};

// Natural logarithm operator.
template <typename DataT>
struct LogOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Log(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy / x;
  }
};

// Natural logarithm one plus operator.
template <typename DataT>
struct Log1pOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    using std::log1p;
    return log1p(x);
  }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy / (x + El::TypeTraits<DataT>::One());
  }
};

// Cosine operator.
template <typename DataT>
struct CosOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Cos(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return -dy * El::Sin(x);
  }
};

// Sine operator.
template <typename DataT>
struct SinOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Sin(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy * El::Cos(x);
  }
};

// Tangent operator.
template <typename DataT>
struct TanOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Tan(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    const auto& c = El::Cos(x);
    return dy / (c * c);
  }
};

// Arccosine operator.
template <typename DataT>
struct AcosOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Acos(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return -dy / El::Sqrt(El::TypeTraits<DataT>::One() - x * x);
  }
};

// Arcsine operator.
template <typename DataT>
struct AsinOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Asin(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy / El::Sqrt(El::TypeTraits<DataT>::One() - x * x);
  }
};

// Arctangent operator.
template <typename DataT>
struct AtanOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Atan(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy / (El::TypeTraits<DataT>::One() + x * x);
  }
};

// Hyperbolic cosine operator.
template <typename DataT>
struct CoshOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Cosh(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy * El::Sinh(x);
  }
};

// Hyperbolic sine operator.
template <typename DataT>
struct SinhOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Sinh(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy * El::Cosh(x);
  }
};

// Hyperbolic tangent operator.
template <typename DataT>
struct TanhOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Tanh(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    const auto& c = El::Cosh(x);
    return dy / (c * c);
  }
};

// Hyperbolic arccosine operator.
template <typename DataT>
struct AcoshOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Acosh(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return -dy / (El::Sqrt(x - El::TypeTraits<DataT>::One()) *
                  El::Sqrt(x + El::TypeTraits<DataT>::One()));
  }
};

// Hyperbolic arcsine operator.
template <typename DataT>
struct AsinhOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Asinh(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy / El::Sqrt(El::TypeTraits<DataT>::One() + x * x);
  }
};

// Hyperbolic arctangent operator.
template <typename DataT>
struct AtanhOpImpl
{
  DataT operator()(DataT const& x) const noexcept { return El::Atanh(x); }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy / (El::TypeTraits<DataT>::One() - x * x);
  }
};

// Error function operator.
template <typename DataT>
struct ErfOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    return El::To<DataT>(std::erf(El::To<double>(x)));
  }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    const auto two_rsqrt_pi = El::To<DataT>(1.12837916709551257389);
    return dy * two_rsqrt_pi * El::Exp(-x * x);
  }
};

// Inverse error function operator.
template <typename DataT>
struct ErfInvOpImpl
{

  DataT operator()(DataT const& x) const noexcept
  {

    // Trivial cases
    const DataT inf = std::numeric_limits<DataT>::infinity();
    if (x <= -El::TypeTraits<DataT>::One()) {
      return -inf;
    }
    if (x >= El::TypeTraits<DataT>::One()) {
      return inf;
    }

    // Apply Newton's method
    const double x_ = El::To<double>(x);
    double y = x_;
    constexpr double half_sqrt_pi = 0.88622692545275801364;
    constexpr double eps = std::numeric_limits<double>::epsilon();
    constexpr int max_iters = 50;
    for (int iter = 0; iter < max_iters; ++iter) {
      const double err = std::erf(y) - x_;
      if (std::isinf(y) || std::abs(err) < eps) {
        break;
      }
      y -= err * half_sqrt_pi * std::exp(y * y);
    }
    return El::To<DataT>(y);
  }

  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    if (El::Abs(x) >= El::TypeTraits<DataT>::One()) {
      return El::TypeTraits<DataT>::Zero();
    }
    else {
      const auto half_sqrt_pi = El::To<DataT>(0.88622692545275801364);
      const auto y = this->operator()(x);
      return dy * half_sqrt_pi * El::Exp(y * y);
    }
  }
};

// GELU operator
template <typename DataT>
struct GeluOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    const double rsqrt_two = 0.7071067811865475;
    return (x / 2) *
           (1 + El::To<DataT>(std::erf(El::To<double>(x) * rsqrt_two)));
  }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    const double rsqrt_two = 0.7071067811865475;
    const auto sqrt_two_pi = El::To<DataT>(2.5066282746310002);
    const auto h = El::To<DataT>(0.5);

    auto term1 = h * El::To<DataT>(std::erf(El::To<double>(x) * rsqrt_two));
    auto term2 = El::Exp(-x * x / 2) * x * sqrt_two_pi;
    return dy * (h + term1 + term2);
  }
};

// GELU operator (hyperbolic tangent approximation)
template <typename DataT>
struct GeluNewOpImpl
{
  DataT operator()(DataT const& x) const noexcept
  {
    // Coefficients as they appear in the BERT and GPT codebases
    const auto sqrt_two_over_pi = El::To<DataT>(0.7978845608028654);
    const auto coeff = El::To<DataT>(0.044715);

    auto hx = x / 2;
    return hx * (1 + El::Tanh(sqrt_two_over_pi * (x + coeff * x * x * x)));
  }
  DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    const auto c1 = El::To<DataT>(0.797885);
    const auto c2 = El::To<DataT>(0.107032);
    const auto c3 = El::To<DataT>(0.0356774);
    auto x3 = x * x * x;
    auto c1x = c1 * x;
    auto c3x3 = c3 * x3;
    auto sech = El::To<DataT>(1) / El::Cosh(c1x + c3x3);
    auto dx = (1 + (c1x + c2 * x3) * sech * sech + El::Tanh(c1x + c3x3));

    return dx * dy / 2;
  }
};

} // namespace

// Template instantiation
#define DEFINE_COMPUTE_OPS(OP_NAME)                                            \
  template <typename DataT, El::Device Device>                                 \
  void OP_NAME##Operator<DataT, Device>::fp_compute_local(                     \
    std::vector<ConstLocalInputTensorType> inputs,                             \
    std::vector<LocalOutputTensorType> outputs) const                          \
  {                                                                            \
    LBANN_ASSERT_DEBUG(inputs.size() == 1);                                    \
    LBANN_ASSERT_DEBUG(outputs.size() == 1);                                   \
    auto const& input = inputs.front().data();                                 \
    auto& output = outputs.front().data();                                     \
    El::EntrywiseMap(                                                          \
      input,                                                                   \
      output,                                                                  \
      std::function<DataT(DataT const&)>(OP_NAME##OpImpl<DataT>{}));           \
  }                                                                            \
  template <typename DataT, El::Device Device>                                 \
  void OP_NAME##Operator<DataT, Device>::bp_compute_local(                     \
    std::vector<ConstLocalInputTensorType> inputs,                             \
    std::vector<ConstLocalOutputTensorType> grads_wrt_outputs,                 \
    std::vector<LocalInputTensorType> grads_wrt_inputs) const                  \
  {                                                                            \
    LBANN_ASSERT_DEBUG(inputs.size() == 1);                                    \
    LBANN_ASSERT_DEBUG(grads_wrt_outputs.size() == 1);                         \
    LBANN_ASSERT_DEBUG(grads_wrt_inputs.size() == 1);                          \
    auto const& input = inputs.front().data();                                 \
    auto const& grad_wrt_output = grads_wrt_outputs.front().data();            \
    auto& grad_wrt_input = grads_wrt_inputs.front().data();                    \
    internal::EntrywiseZipInto(input,                                          \
                               grad_wrt_output,                                \
                               grad_wrt_input,                                 \
                               OP_NAME##OpImpl<DataT>{});                      \
  }

DEFINE_COMPUTE_OPS(Acos)
DEFINE_COMPUTE_OPS(Acosh)
DEFINE_COMPUTE_OPS(Asin)
DEFINE_COMPUTE_OPS(Asinh)
DEFINE_COMPUTE_OPS(Atan)
DEFINE_COMPUTE_OPS(Atanh)
DEFINE_COMPUTE_OPS(Ceil)
DEFINE_COMPUTE_OPS(Cos)
DEFINE_COMPUTE_OPS(Cosh)
DEFINE_COMPUTE_OPS(Erf)
DEFINE_COMPUTE_OPS(ErfInv)
DEFINE_COMPUTE_OPS(Exp)
DEFINE_COMPUTE_OPS(Expm1)
DEFINE_COMPUTE_OPS(Floor)
DEFINE_COMPUTE_OPS(Gelu)
DEFINE_COMPUTE_OPS(GeluNew)
DEFINE_COMPUTE_OPS(Log)
DEFINE_COMPUTE_OPS(Log1p)
DEFINE_COMPUTE_OPS(LogicalNot)
DEFINE_COMPUTE_OPS(Negative)
DEFINE_COMPUTE_OPS(Reciprocal)
DEFINE_COMPUTE_OPS(Round)
DEFINE_COMPUTE_OPS(Rsqrt)
DEFINE_COMPUTE_OPS(SafeReciprocal)
DEFINE_COMPUTE_OPS(Sign)
DEFINE_COMPUTE_OPS(Sin)
DEFINE_COMPUTE_OPS(Sinh)
DEFINE_COMPUTE_OPS(Sqrt)
DEFINE_COMPUTE_OPS(Square)
DEFINE_COMPUTE_OPS(Tan)
DEFINE_COMPUTE_OPS(Tanh)

#define PROTO(T)                                                               \
  template class AcosOperator<T, El::Device::CPU>;                             \
  template class AcoshOperator<T, El::Device::CPU>;                            \
  template class AsinOperator<T, El::Device::CPU>;                             \
  template class AsinhOperator<T, El::Device::CPU>;                            \
  template class AtanOperator<T, El::Device::CPU>;                             \
  template class AtanhOperator<T, El::Device::CPU>;                            \
  template class CeilOperator<T, El::Device::CPU>;                             \
  template class CosOperator<T, El::Device::CPU>;                              \
  template class CoshOperator<T, El::Device::CPU>;                             \
  template class ErfInvOperator<T, El::Device::CPU>;                           \
  template class ErfOperator<T, El::Device::CPU>;                              \
  template class ExpOperator<T, El::Device::CPU>;                              \
  template class Expm1Operator<T, El::Device::CPU>;                            \
  template class FloorOperator<T, El::Device::CPU>;                            \
  template class GeluOperator<T, El::Device::CPU>;                             \
  template class GeluNewOperator<T, El::Device::CPU>;                          \
  template class Log1pOperator<T, El::Device::CPU>;                            \
  template class LogOperator<T, El::Device::CPU>;                              \
  template class LogicalNotOperator<T, El::Device::CPU>;                       \
  template class NegativeOperator<T, El::Device::CPU>;                         \
  template class ReciprocalOperator<T, El::Device::CPU>;                       \
  template class RoundOperator<T, El::Device::CPU>;                            \
  template class RsqrtOperator<T, El::Device::CPU>;                            \
  template class SafeReciprocalOperator<T, El::Device::CPU>;                   \
  template class SignOperator<T, El::Device::CPU>;                             \
  template class SinOperator<T, El::Device::CPU>;                              \
  template class SinhOperator<T, El::Device::CPU>;                             \
  template class SqrtOperator<T, El::Device::CPU>;                             \
  template class SquareOperator<T, El::Device::CPU>;                           \
  template class TanOperator<T, El::Device::CPU>;                              \
  template class TanhOperator<T, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
