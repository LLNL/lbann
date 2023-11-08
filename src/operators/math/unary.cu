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

#include "lbann/operators/math/unary.hpp"

#include "lbann/base.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "common.cuh"

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
template <typename DataT>
struct LogicalNotOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    auto const& b = x != DataT(0.0) && !gpu_lib::isnan(x);
    return !b ? DataT(1.0) : DataT(0.0);
  }
  inline __device__ DataT operator()(DataT const& /*x*/,
                                     DataT const& /*dy*/) const
  {
    return DataT(0.0);
  }
};

/** Negative operator. */
template <typename DataT>
struct NegativeOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const { return -x; }
  inline __device__ DataT operator()(DataT const& /*x*/, DataT const& dy) const
  {
    return -dy;
  }
};

/** Sign operator. */
template <typename DataT>
struct SignOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    DataT const zero = 0.;
    DataT const one = 1.;
    if (x > zero) {
      return one;
    }
    else if (x < zero) {
      return -one;
    }
    else {
      return zero;
    }
  }
  inline __device__ DataT operator()(DataT const& /*x*/,
                                     DataT const& /*dy*/) const
  {
    return DataT(0.0);
  }
};

/** Round operator. */
template <typename DataT>
struct RoundOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::round(x);
  }
  inline __device__ DataT operator()(DataT const& /*x*/,
                                     DataT const& /*dy*/) const
  {
    return DataT(0.0);
  }
};

/** Ceiling operator. */
template <typename DataT>
struct CeilOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::ceil(x);
  }
  inline __device__ DataT operator()(DataT const& /*x*/,
                                     DataT const& /*dy*/) const
  {
    return DataT(0.0);
  }
};

/** Floor operator. */
template <typename DataT>
struct FloorOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::floor(x);
  }
  inline __device__ DataT operator()(DataT const& /*x*/,
                                     DataT const& /*dy*/) const
  {
    return DataT(0.0);
  }
};

/** Reciprocal operator. */
template <typename DataT>
struct ReciprocalOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return DataT(1.) / x;
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    if (dy == DataT(0.0)) {
      return DataT(0.0);
    }
    else {
      return -dy / (x * x);
    }
  }
};

/** Square operator. */
template <typename DataT>
struct SquareOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const { return x * x; }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return DataT(2.) * x * dy;
  }
};

/** Square root operator. */
template <typename DataT>
struct SqrtOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::sqrt(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy / (DataT(2.) * gpu_lib::sqrt(x));
  }
};

/** Reciprocal square root operator. */
template <typename DataT>
struct RsqrtOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::rsqrt(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    auto const& s = gpu_lib::sqrt(x);
    return -dy / (DataT(2.) * x * s);
  }
};

/** Safe reciprocal operator.
 *  If a standard reciprocal produces an infinity or NaN, zero is
 *  output instead.
 */
template <typename DataT>
struct SafeReciprocalOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    auto const& y = DataT(1.) / x;
    if (gpu_lib::isfinite(y)) {
      return y;
    }
    else {
      return DataT(0.0);
    }
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    auto const& y = DataT(1.) / x;
    if (gpu_lib::isfinite(y)) {
      return -dy * y * y;
    }
    else {
      return DataT(0.0);
    }
  }
};

/** Exponential operator. */
template <typename DataT>
struct ExpOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::exp(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy * gpu_lib::exp(x);
  }
};

/** Exponential minus one operator. */
template <typename DataT>
struct Expm1OpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::expm1(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy * gpu_lib::exp(x);
  }
};

/** Natural logarithm operator. */
template <typename DataT>
struct LogOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::log(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy / x;
  }
};

/** Natural logarithm one plus operator. */
template <typename DataT>
struct Log1pOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::log1p(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy / (x + DataT(1.0));
  }
};

/** Cosine operator. */
template <typename DataT>
struct CosOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::cos(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return -dy * gpu_lib::sin(x);
  }
};

/** Sine operator. */
template <typename DataT>
struct SinOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::sin(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy * gpu_lib::cos(x);
  }
};

/** Tangent operator. */
template <typename DataT>
struct TanOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::tan(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    auto const& c = gpu_lib::cos(x);
    return dy / (c * c);
  }
};

/** Arccosine operator. */
template <typename DataT>
struct AcosOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::acos(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return -dy / gpu_lib::sqrt(DataT(1.0) - x * x);
  }
};

/** Arcsine operator. */
template <typename DataT>
struct AsinOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::asin(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy / gpu_lib::sqrt(DataT(1.0) - x * x);
  }
};

/** Arctangent operator. */
template <typename DataT>
struct AtanOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::atan(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy / (DataT(1.0) + x * x);
  }
};

/** Hyperbolic cosine operator. */
template <typename DataT>
struct CoshOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::cosh(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy * gpu_lib::sinh(x);
  }
};

/** Hyperbolic sine operator. */
template <typename DataT>
struct SinhOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::sinh(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy * gpu_lib::cosh(x);
  }
};

/** Hyperbolic tangent operator. */
template <typename DataT>
struct TanhOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::tanh(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    auto const& c = gpu_lib::cosh(x);
    return dy / (c * c);
  }
};

/** Hyperbolic arccosine operator. */
template <typename DataT>
struct AcoshOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::acosh(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return -dy /
           (gpu_lib::sqrt(x - DataT(1.0)) * gpu_lib::sqrt(x + DataT(1.0)));
  }
};

/** Hyperbolic arcsine operator. */
template <typename DataT>
struct AsinhOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::asinh(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy / gpu_lib::sqrt(DataT(1.0) + x * x);
  }
};

/** Hyperbolic arctangent operator. */
template <typename DataT>
struct AtanhOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::atanh(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    return dy / (DataT(1.0) - x * x);
  }
};

/** Error function operator. */
template <typename DataT>
struct ErfOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::erf(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    DataT const two_rsqrt_pi(1.12837916709551257389);
    return dy * two_rsqrt_pi * gpu_lib::exp(-x * x);
  }
};

/** Inverse error function operator. */
template <typename DataT>
struct ErfInvOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    return gpu_lib::erfinv(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    DataT const half_sqrt_pi(0.88622692545275801364);
    auto const& y = gpu_lib::erfinv(x);
    return dy * half_sqrt_pi * gpu_lib::exp(y * y);
  }
};

// GELU operator (hyperbolic tangent approximation)
template <typename DataT>
struct GeluOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const
  {
    // Coefficients as they appear in the BERT and GPT codebases
    DataT const sqrt_two_over_pi(0.7978845608028654);
    DataT const coeff(0.044715);

    DataT hx = x * DataT(0.5);
    return hx * (DataT(1) +
                 gpu_lib::tanh(sqrt_two_over_pi * (x + coeff * x * x * x)));
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const
  {
    DataT const c1(0.797885);
    DataT const c2(0.107032);
    DataT const c3(0.0356774);
    DataT x3 = x * x * x;
    DataT c1x = c1 * x;
    DataT c3x3 = c3 * x3;
    DataT sech = DataT(1) / gpu_lib::cosh(c1x + c3x3);
    DataT dx =
      (DataT(1) + (c1x + c2 * x3) * sech * sech + gpu_lib::tanh(c1x + c3x3));

    return dx * dy * DataT(0.5);
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
    El::EntrywiseMap(input, output, OP_NAME##OpImpl<DataT>{});                 \
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
  template class AcosOperator<T, El::Device::GPU>;                             \
  template class AcoshOperator<T, El::Device::GPU>;                            \
  template class AsinOperator<T, El::Device::GPU>;                             \
  template class AsinhOperator<T, El::Device::GPU>;                            \
  template class AtanOperator<T, El::Device::GPU>;                             \
  template class AtanhOperator<T, El::Device::GPU>;                            \
  template class CeilOperator<T, El::Device::GPU>;                             \
  template class CosOperator<T, El::Device::GPU>;                              \
  template class CoshOperator<T, El::Device::GPU>;                             \
  template class ErfInvOperator<T, El::Device::GPU>;                           \
  template class ErfOperator<T, El::Device::GPU>;                              \
  template class ExpOperator<T, El::Device::GPU>;                              \
  template class Expm1Operator<T, El::Device::GPU>;                            \
  template class FloorOperator<T, El::Device::GPU>;                            \
  template class GeluOperator<T, El::Device::GPU>;                             \
  template class Log1pOperator<T, El::Device::GPU>;                            \
  template class LogOperator<T, El::Device::GPU>;                              \
  template class LogicalNotOperator<T, El::Device::GPU>;                       \
  template class NegativeOperator<T, El::Device::GPU>;                         \
  template class ReciprocalOperator<T, El::Device::GPU>;                       \
  template class RoundOperator<T, El::Device::GPU>;                            \
  template class RsqrtOperator<T, El::Device::GPU>;                            \
  template class SafeReciprocalOperator<T, El::Device::GPU>;                   \
  template class SignOperator<T, El::Device::GPU>;                             \
  template class SinOperator<T, El::Device::GPU>;                              \
  template class SinhOperator<T, El::Device::GPU>;                             \
  template class SqrtOperator<T, El::Device::GPU>;                             \
  template class SquareOperator<T, El::Device::GPU>;                           \
  template class TanOperator<T, El::Device::GPU>;                              \
  template class TanhOperator<T, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
