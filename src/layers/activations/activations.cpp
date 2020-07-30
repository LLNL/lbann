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

#define LBANN_ACTIVATIONS_LAYER_INSTANTIATE
#include "lbann/layers/activations/activations.hpp"
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

/** Log sigmoid operator. */
template <typename TensorDataType>
struct log_sigmoid_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    using std::log1p;
    if (x >= El::TypeTraits<TensorDataType>::Zero()) {
      return -log1p(El::Exp(-x));
    } else {
      return x - log1p(El::Exp(x));
    }
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (El::TypeTraits<TensorDataType>::One() + El::Exp(x));
  }
};

/** SELU operator. */
template <typename TensorDataType>
struct selu_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    using std::expm1;
    static const auto alpha = TensorDataType(1.6732632423543772848170429916717);
    static const auto scale = TensorDataType(1.0507009873554804934193349852946);
    static const auto zero = TensorDataType(0.);
    return (x > zero ?
            scale * x :
            scale * alpha * expm1(x));
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    static const auto alpha = TensorDataType(1.6732632423543772848170429916717);
    static const auto scale = TensorDataType(1.0507009873554804934193349852946);
    static const auto zero = TensorDataType(0.);
    return (x > zero ?
            dy * scale :
            dy * scale * alpha * El::Exp(x));
  }
};

/** Sigmoid operator. */
template <typename TensorDataType>
struct sigmoid_op {
  TensorDataType eps = std::numeric_limits<TensorDataType>::epsilon();
  inline TensorDataType operator()(const TensorDataType& x) const {
    static const auto one = El::TypeTraits<TensorDataType>::One();
    const auto& y = one / (one + El::Exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps)            { return eps; }
    else if (y >= one - eps) { return one - eps; }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return y;
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    static const auto one = El::TypeTraits<TensorDataType>::One();
    const auto& y = one / (one + El::Exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps || y >= one - eps) { return El::TypeTraits<TensorDataType>::Zero(); }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return dy * y * (one - y);
  }
};

/** Softplus operator. */
template <typename TensorDataType>
struct softplus_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    using std::log1p;
    if (x > El::TypeTraits<TensorDataType>::Zero()) {
      return log1p(El::Exp(-x)) + x;
    } else {
      return log1p(El::Exp(x));
    }
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (El::TypeTraits<TensorDataType>::One() + El::Exp(-x));
  }
};

/** Softsign operator. */
template <typename TensorDataType>
struct softsign_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    using std::fabs;
    return x / (El::TypeTraits<TensorDataType>::One() + fabs(x));
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    using std::fabs;
    const auto& denom = El::TypeTraits<TensorDataType>::One() + fabs(x);
    return dy / (denom * denom);
  }
};

} // namespace

// Template instantiation
#define DEFINE_COMPUTE_OPS(layer, op)                                   \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::fp_compute() {            \
    apply_entrywise_unary_operator<op>(                                 \
      this->get_prev_activations(),                                     \
      this->get_activations());                                         \
  }                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::bp_compute() {            \
    apply_entrywise_binary_operator<op>(                                \
      this->get_prev_activations(),                                     \
      this->get_prev_error_signals(),                                   \
      this->get_error_signals());                                       \
  }

DEFINE_COMPUTE_OPS(log_sigmoid_layer, log_sigmoid_op)
DEFINE_COMPUTE_OPS(selu_layer, selu_op)
DEFINE_COMPUTE_OPS(sigmoid_layer, sigmoid_op)
DEFINE_COMPUTE_OPS(softplus_layer, softplus_op)
DEFINE_COMPUTE_OPS(softsign_layer, softsign_op)

#define PROTO(T) \
  UNARY_ETI_INST_MACRO_DEV_DT(log_sigmoid_layer, T, El::Device::CPU); \
  UNARY_ETI_INST_MACRO_DEV_DT(selu_layer, T, El::Device::CPU);        \
  UNARY_ETI_INST_MACRO_DEV_DT(sigmoid_layer, T, El::Device::CPU);     \
  UNARY_ETI_INST_MACRO_DEV_DT(softplus_layer, T, El::Device::CPU);    \
  UNARY_ETI_INST_MACRO_DEV_DT(softsign_layer, T, El::Device::CPU)

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
