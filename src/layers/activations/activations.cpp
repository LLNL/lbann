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

#include "lbann/layers/activations/activations.hpp"
#include "lbann/utils/entrywise_operator.hpp"

namespace lbann {

namespace {

// Helpful constants
constexpr DataType zero = 0;
constexpr DataType one = 1;
constexpr DataType eps = std::numeric_limits<DataType>::epsilon();

// =========================================================
// Operator objects for entry-wise unary layers
// =========================================================
// Note: Unary operator corresponds to forward prop step
// (\f$ y = f(x) \f$) and binary operator corresponds to
// back prop step
// (\f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$).

/** Log sigmoid operator. */
struct log_sigmoid_op {
  inline DataType operator()(const DataType& x) const {
    if (x >= zero) {
      return -std::log1p(std::exp(-x));
    } else {
      return x - std::log1p(std::exp(x));
    }
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (one + std::exp(x));
  }
};

/** ReLU operator. */
struct relu_op {
  inline DataType operator()(const DataType& x) const {
    return std::max(x, zero);
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return x > zero ? dy : zero;
  }
};

/** SELU operator. */
struct selu_op {
  inline DataType operator()(const DataType& x) const {
    return (x > zero ?
            scale * x :
            scale * alpha * std::expm1(x));
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return (x > zero ?
            dy * scale :
            dy * scale * alpha * std::exp(x));
  }
private:
  static constexpr DataType alpha = 1.6732632423543772848170429916717;
  static constexpr DataType scale = 1.0507009873554804934193349852946;
};

/** Sigmoid operator. */
struct sigmoid_op {
  inline DataType operator()(const DataType& x) const {
    const auto& y = 1 / (one + std::exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps)            { return eps; }
    else if (y >= one - eps) { return one - eps; }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return y;
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& y = 1 / (one + std::exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps || y >= one - eps) { return zero; }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return dy * y * (one - y);
  }
};

/** Softplus operator. */
struct softplus_op {
  inline DataType operator()(const DataType& x) const {
    if (x > zero) {
      return std::log1p(std::exp(-x)) + x;
    } else {
      return std::log1p(std::exp(x));
    }
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (one + std::exp(-x));
  }
};

/** Softsign operator. */
struct softsign_op {
  inline DataType operator()(const DataType& x) const {
    return x / (one + std::fabs(x));
  }
  inline DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& denom = one + std::fabs(x);
    return dy / (denom * denom);
  }
};

} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                          \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::CPU>              \
  ::fp_compute() {                                                      \
    apply_entrywise_unary_operator<op>(get_prev_activations(),          \
                                       get_activations());              \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::CPU>              \
  ::bp_compute() {                                                      \
    apply_entrywise_binary_operator<op>(get_prev_activations(),         \
                                        get_prev_error_signals(),       \
                                        get_error_signals());           \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::CPU>               \
  ::fp_compute() {                                                      \
    apply_entrywise_unary_operator<op>(get_prev_activations(),          \
                                       get_activations());              \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::CPU>               \
  ::bp_compute() {                                                      \
    apply_entrywise_binary_operator<op>(get_prev_activations(),         \
                                        get_prev_error_signals(),       \
                                        get_error_signals());           \
  }
  INSTANTIATE(log_sigmoid_layer, log_sigmoid_op)
#ifndef LBANN_HAS_DISTCONV
  INSTANTIATE(relu_layer, relu_op)
#endif
  INSTANTIATE(selu_layer, selu_op)
  INSTANTIATE(sigmoid_layer, sigmoid_op)
  INSTANTIATE(softplus_layer, softplus_op)
  INSTANTIATE(softsign_layer, softsign_op)

} // namespace lbann
