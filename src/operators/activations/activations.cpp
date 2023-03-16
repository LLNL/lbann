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

#include "lbann/operators/activations/activations.hpp"
#include "../math/common.hpp"

namespace lbann {

namespace {

// =========================================================
// Operator objects for entry-wise unary layers
// =========================================================
// Note: Unary operator corresponds to forward prop step
// (\f$ y = f(x) \f$) and binary operator corresponds to
// back prop step
// (\f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$).

/** Log sigmoid operator. */
template <typename DataT>
struct LogSigmoidOpImpl
{
  inline DataT operator()(DataT const& x) const noexcept
  {
    using std::log1p;
    if (x >= El::TypeTraits<DataT>::Zero()) {
      return -log1p(El::Exp(-x));
    }
    else {
      return x - log1p(El::Exp(x));
    }
  }
  inline DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy / (El::TypeTraits<DataT>::One() + El::Exp(x));
  }
};

/** SELU operator. */
template <typename DataT>
struct SeluOpImpl
{
  inline DataT operator()(DataT const& x) const noexcept
  {
    using std::expm1;
    static auto const alpha = DataT(1.6732632423543772848170429916717);
    static auto const scale = DataT(1.0507009873554804934193349852946);
    static auto const zero = DataT(0.);
    return (x > zero ? scale * x : scale * alpha * expm1(x));
  }
  inline DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    static auto const alpha = DataT(1.6732632423543772848170429916717);
    static auto const scale = DataT(1.0507009873554804934193349852946);
    static auto const zero = DataT(0.);
    return (x > zero ? dy * scale : dy * scale * alpha * El::Exp(x));
  }
};

/** Sigmoid operator. */
template <typename DataT>
struct SigmoidOpImpl
{
  DataT eps = std::numeric_limits<DataT>::epsilon();
  inline DataT operator()(DataT const& x) const noexcept
  {
    static auto const one = El::TypeTraits<DataT>::One();
    auto const& y = one / (one + El::Exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps) {
      return eps;
    }
    else if (y >= one - eps) {
      return one - eps;
    }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return y;
  }
  inline DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    static auto const one = El::TypeTraits<DataT>::One();
    auto const& y = one / (one + El::Exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps || y >= one - eps) {
      return El::TypeTraits<DataT>::Zero();
    }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return dy * y * (one - y);
  }
};

/** Softplus operator. */
template <typename DataT>
struct SoftplusOpImpl
{
  inline DataT operator()(DataT const& x) const noexcept
  {
    using std::log1p;
    if (x > El::TypeTraits<DataT>::Zero()) {
      return log1p(El::Exp(-x)) + x;
    }
    else {
      return log1p(El::Exp(x));
    }
  }
  inline DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    return dy / (El::TypeTraits<DataT>::One() + El::Exp(-x));
  }
};

/** Softsign operator. */
template <typename DataT>
struct SoftsignOpImpl
{
  inline DataT operator()(DataT const& x) const noexcept
  {
    using std::fabs;
    return x / (El::TypeTraits<DataT>::One() + fabs(x));
  }
  inline DataT operator()(DataT const& x, DataT const& dy) const noexcept
  {
    using std::fabs;
    auto const& denom = El::TypeTraits<DataT>::One() + fabs(x);
    return dy / (denom * denom);
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

DEFINE_COMPUTE_OPS(LogSigmoid)
DEFINE_COMPUTE_OPS(Selu)
DEFINE_COMPUTE_OPS(Sigmoid)
DEFINE_COMPUTE_OPS(Softplus)
DEFINE_COMPUTE_OPS(Softsign)

#define PROTO(T)                                                               \
  template class LogSigmoidOperator<T, El::Device::CPU>;                       \
  template class SeluOperator<T, El::Device::CPU>;                             \
  template class SigmoidOperator<T, El::Device::CPU>;                          \
  template class SoftplusOperator<T, El::Device::CPU>;                         \
  template class SoftsignOperator<T, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
