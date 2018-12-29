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
struct log_sigmoid_op {
  inline __device__ DataType operator()(const DataType& x) const {
    if (x >= DataType(0)) {
      return -cuda::log1p(cuda::exp(-x));
    } else {
      return x - cuda::log1p(cuda::exp(x));
    }
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (DataType(1) + cuda::exp(x));
  }
};

/** ReLU operator. */
struct relu_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return cuda::max(x, DataType(0));
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return x > DataType(0) ? dy : DataType(0);
  }
};

/** SELU operator. */
struct selu_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return (x > DataType(0) ?
            scale * x :
            scale * alpha * cuda::expm1(x));
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return (x > DataType(0) ?
            dy * scale :
            dy * scale * alpha * cuda::exp(x));
  }
private:
  static constexpr DataType alpha = 1.6732632423543772848170429916717;
  static constexpr DataType scale = 1.0507009873554804934193349852946;
};

/** Sigmoid operator. */
struct sigmoid_op {
  inline __device__ DataType operator()(const DataType& x) const {
    constexpr DataType one = 1;
    const auto& y = 1 / (one + cuda::exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    constexpr DataType eps = cuda::epsilon<DataType>();
    if (y <= eps) { return eps; }
    else if (y >= one - eps) { return one - eps; }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return y;
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    constexpr DataType one = 1;
    const auto& y = 1 / (one + cuda::exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    constexpr DataType eps = cuda::epsilon<DataType>();
    if (y <= eps || y >= DataType(1) - eps) { return DataType(0); }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return dy * y * (DataType(1) - y);
  }
};

/** Softplus operator. */
struct softplus_op {
  inline __device__ DataType operator()(const DataType& x) const {
    if (x > DataType(0)) {
      return cuda::log1p(cuda::exp(-x)) + x;
    } else {
      return cuda::log1p(cuda::exp(x));
    }
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    return dy / (DataType(1) + cuda::exp(-x));
  }
};

/** Softsign operator. */
struct softsign_op {
  inline __device__ DataType operator()(const DataType& x) const {
    return x / (DataType(1) + cuda::abs(x));
  }
  inline __device__ DataType operator()(const DataType& x, const DataType& dy) const {
    const auto& denom = DataType(1) + cuda::abs(x);
    return dy / (denom * denom);
  }
};

} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                          \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::GPU>              \
  ::fp_compute() {                                                      \
    cuda::apply_entrywise_unary_operator<op>(get_prev_activations(),    \
                                             get_activations());        \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::GPU>              \
  ::bp_compute() {                                                      \
    cuda::apply_entrywise_binary_operator<op>(get_prev_activations(),   \
                                              get_prev_error_signals(), \
                                              get_error_signals());     \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::GPU>               \
  ::fp_compute() {                                                      \
    cuda::apply_entrywise_unary_operator<op>(get_prev_activations(),    \
                                             get_activations());        \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::GPU>               \
  ::bp_compute() {                                                      \
    cuda::apply_entrywise_binary_operator<op>(get_prev_activations(),   \
                                              get_prev_error_signals(), \
                                              get_error_signals());     \
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
