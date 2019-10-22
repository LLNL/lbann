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

/** Log sigmoid operator. */
template <typename TensorDataType>
struct log_sigmoid_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    if (x >= TensorDataType(0)) {
      return -cuda::log1p(cuda::exp(-x));
    } else {
      return x - cuda::log1p(cuda::exp(x));
    }
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (TensorDataType(1) + cuda::exp(x));
  }
};

/** ReLU operator. */
template <typename TensorDataType>
struct relu_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return cuda::max(x, TensorDataType(0));
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return x > TensorDataType(0) ? dy : TensorDataType(0);
  }
};

/** SELU operator. */
template <typename TensorDataType>
struct selu_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return (x > TensorDataType(0) ?
            scale * x :
            scale * alpha * cuda::expm1(x));
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return (x > TensorDataType(0) ?
            dy * scale :
            dy * scale * alpha * cuda::exp(x));
  }
private:
  static constexpr TensorDataType alpha = 1.6732632423543772848170429916717;
  static constexpr TensorDataType scale = 1.0507009873554804934193349852946;
};

/** Sigmoid operator. */
template <typename TensorDataType>
struct sigmoid_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    constexpr TensorDataType one = 1;
    const auto& y = 1 / (one + cuda::exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    constexpr TensorDataType eps = cuda::epsilon<TensorDataType>();
    if (y <= eps) { return eps; }
    else if (y >= one - eps) { return one - eps; }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return y;
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    constexpr TensorDataType one = 1;
    const auto& y = 1 / (one + cuda::exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    constexpr TensorDataType eps = cuda::epsilon<TensorDataType>();
    if (y <= eps || y >= TensorDataType(1) - eps) { return TensorDataType(0); }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return dy * y * (TensorDataType(1) - y);
  }
};

/** Softplus operator. */
template <typename TensorDataType>
struct softplus_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    if (x > TensorDataType(0)) {
      return cuda::log1p(cuda::exp(-x)) + x;
    } else {
      return cuda::log1p(cuda::exp(x));
    }
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (TensorDataType(1) + cuda::exp(-x));
  }
};

/** Softsign operator. */
template <typename TensorDataType>
struct softsign_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x) const {
    return x / (TensorDataType(1) + cuda::abs(x));
  }
  inline __device__ TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& denom = TensorDataType(1) + cuda::abs(x);
    return dy / (denom * denom);
  }
};

} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                                                   \
  template <typename TensorDataType>                                                             \
  void fp_compute_inst(layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::GPU>& l) { \
    cuda::apply_entrywise_unary_operator<op<TensorDataType>>(l.get_prev_activations(),           \
                                                             l.get_activations());               \
  }                                                                                              \
  template <typename TensorDataType>                                                             \
  void bp_compute_inst(layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::GPU>& l) { \
    cuda::apply_entrywise_binary_operator<op<TensorDataType>>(l.get_prev_activations(),          \
                                                              l.get_prev_error_signals(),        \
                                                              l.get_error_signals());            \
  }                                                                                              \
  template <typename TensorDataType>                                                             \
  void fp_compute_impl(layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l) {  \
    cuda::apply_entrywise_unary_operator<op<TensorDataType>>(l.get_prev_activations(),           \
                                                             l.get_activations());               \
  }                                                                                              \
  template <typename TensorDataType>                                                             \
  void bp_compute_impl(layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l) {  \
    cuda::apply_entrywise_binary_operator<op<TensorDataType>>(l.get_prev_activations(),          \
                                                              l.get_prev_error_signals(),        \
                                                              l.get_error_signals());            \
  }                                                                                              \
  UNARY_ETI_INST_MACRO_DEV(layer, El::Device::GPU)

INSTANTIATE(log_sigmoid_layer, log_sigmoid_op);
INSTANTIATE(relu_layer, relu_op);
INSTANTIATE(selu_layer, selu_op);
INSTANTIATE(sigmoid_layer, sigmoid_op);
INSTANTIATE(softplus_layer, softplus_op);
INSTANTIATE(softsign_layer, softsign_op);

} // namespace lbann
