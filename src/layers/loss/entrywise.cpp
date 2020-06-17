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

#define LBANN_ENTRYWISE_LAYER_INSTANTIATE
#include "lbann/layers/loss/entrywise.hpp"
#include "lbann/utils/entrywise_operator.hpp"
#include "lbann/utils/numerical_traits.hpp"

namespace lbann {

namespace {

/** Apply a binary backprop operator to CPU data.
 *  The input and output data must be on CPU and must have the same
 *  dimensions. Given a binary function \f$ y = f(x_1,x_2) \f$, the
 *  corresponding BinaryBackPropOperator is a 5-ary function with the
 *  arguments \f$ x_1 \f$, \f$ x_2 \f$, \f$ dL/dy \f$, \f$ dL/dx_1\f$,
 *  \f$ dL/dx_2 \f$. The last two arguments should be overwritten when
 *  the BinaryBackPropOperator is called.
 */
template <template <typename> class Op, typename TensorDataType>
void apply_binary_backprop_operator(const El::AbstractMatrix<TensorDataType>& x1,
                                    const El::AbstractMatrix<TensorDataType>& x2,
                                    const El::AbstractMatrix<TensorDataType>& dy,
                                    El::AbstractMatrix<TensorDataType>& dx1,
                                    El::AbstractMatrix<TensorDataType>& dx2) {
  using BinaryBackPropOperator = Op<TensorDataType>;
  if (x1.Contiguous() && x2.Contiguous() && dy.Contiguous()
      && dx1.Contiguous() && dx2.Contiguous()) {
    const auto* x1_buffer = x1.LockedBuffer();
    const auto* x2_buffer = x2.LockedBuffer();
    const auto* dy_buffer = dy.LockedBuffer();
    auto* dx1_buffer = dx1.Buffer();
    auto* dx2_buffer = dx2.Buffer();
    const size_t size = x1.Height() * x1.Width();
    LBANN_OMP_PARALLEL_FOR
    for (size_t i = 0; i < size; ++i) {
      BinaryBackPropOperator op;
      op(x1_buffer[i], x2_buffer[i], dy_buffer[i],
         dx1_buffer[i], dx2_buffer[i]);
    }
  } else {
    auto const width = x1.Width();
    auto const height = x1.Height();
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (El::Int col = 0; col < width; ++col) {
      for (El::Int row = 0; row < height; ++row) {
        BinaryBackPropOperator op;
        op(x1(row, col), x2(row, col), dy(row, col),
           dx1(row, col), dx2(row, col));
      }
    }
  }
}

// =========================================================
// Operator objects for entry-wise binary layers
// =========================================================
// Note: Binary operator corresponds to forward prop step
// (\f$ y = f(x_1,x_2) \f$) and 5-ary operator corresponds
// to back prop step
// (\f$ \frac{dL}{dx_i} = \frac{dL}{dy} \frac{df}{dx_i}(x_1,x_2) \f$).

/** Binary cross entropy operator. */
template <typename TensorDataType>
struct binary_cross_entropy_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                                   const TensorDataType& x2) const {
    static const auto zero = El::TypeTraits<TensorDataType>::Zero();
    static const auto one = El::TypeTraits<TensorDataType>::One();
    TensorDataType y = zero;
    if (x2 > zero) { y += -x2 * std::log(x1); }
    if (x2 < one)  { y += -(one-x2) * std::log(one-x1); }
    return y;
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    static const auto zero = El::TypeTraits<TensorDataType>::Zero();
    static const auto one = El::TypeTraits<TensorDataType>::One();
    dx2 = dx1 = zero;
    if (dy == zero) { return; }
    if (x2 > zero) {
      dx1 += -x2 / x1 * dy;
      dx2 += -std::log(x1) * dy;
    }
    if (x2 < one)  {
      dx1 += (one-x2) / (one-x1) * dy;
      dx2 += std::log(one-x1) * dy;
    }
  }
};

/** Sigmoid binary cross entropy operator.
 *  Equivalent to applying a sigmoid function to the first operand and
 *  then computing the binary cross entropy. Numerically stable
 *  implementation is taken from
 *  https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits.
 */
template <typename TensorDataType>
struct sigmoid_binary_cross_entropy_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                                   const TensorDataType& x2) const {
    using std::exp;
    using std::log1p;
    static const auto zero = El::TypeTraits<TensorDataType>::Zero();
    static const auto one = El::TypeTraits<TensorDataType>::One();
    const auto& z = std::max(zero, std::min(x2, one));
    if (x1 > zero) {
      return (one - z) * x1 + log1p(exp(-x1));
    } else {
      return - x1 * z + log1p(exp(x1));
    }
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    using std::exp;
    using std::log1p;
    static const auto zero = El::TypeTraits<TensorDataType>::Zero();
    static const auto one = El::TypeTraits<TensorDataType>::One();
    const auto& z = std::max(zero, std::min(x2, one));
    if (x1 > zero) {
      dx1 = -z + one / (one + exp(-x1));
    } else {
        dx1 = one - z - one / (one + exp(x1));
    }
    dx1 *= dy;
    dx2 = (x2 == z) ? -x1 * dy : zero;
  }
};

/** Boolean accuracy operator. */
template <typename TensorDataType>
struct boolean_accuracy_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                                   const TensorDataType& x2) const {
    const auto& b1 = x1 >= TensorDataType(0.5);
    const auto& b2 = x2 >= TensorDataType(0.5);
    return b1 == b2
        ? El::TypeTraits<TensorDataType>::One()
        : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx2 = dx1 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Boolean false negative operator. */
template <typename TensorDataType>
struct boolean_false_negative_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                                   const TensorDataType& x2) const {
    const auto& b1 = x1 >= TensorDataType(0.5);
    const auto& b2 = x2 >= TensorDataType(0.5);
    return (!b1 && b2) ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx2 = dx1 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Boolean false positive operator. */
template <typename TensorDataType>
struct boolean_false_positive_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                                   const TensorDataType& x2) const {
    const auto& b1 = x1 >= TensorDataType(0.5);
    const auto& b2 = x2 >= TensorDataType(0.5);
    return (b1 && !b2)
        ? El::TypeTraits<TensorDataType>::One()
        : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx2 = dx1 = El::TypeTraits<TensorDataType>::Zero();
  }
};

} // namespace

// Template instantiation
#define DEFINE_COMPUTE_OPS(layer, op)                                   \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::fp_compute() {            \
    apply_entrywise_binary_operator<op>(                                \
      this->get_prev_activations(0),                                    \
      this->get_prev_activations(1),                                    \
      this->get_activations());                                         \
  }                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::bp_compute() {            \
    apply_binary_backprop_operator<op>(                                 \
      this->get_local_prev_activations(0),                              \
      this->get_local_prev_activations(1),                              \
      this->get_local_prev_error_signals(),                             \
      this->get_local_error_signals(0),                                 \
      this->get_local_error_signals(1));                                \
  }

DEFINE_COMPUTE_OPS(binary_cross_entropy_layer, binary_cross_entropy_op)
DEFINE_COMPUTE_OPS(sigmoid_binary_cross_entropy_layer, sigmoid_binary_cross_entropy_op)
DEFINE_COMPUTE_OPS(boolean_accuracy_layer, boolean_accuracy_op)
DEFINE_COMPUTE_OPS(boolean_false_negative_layer, boolean_false_negative_op)
DEFINE_COMPUTE_OPS(boolean_false_positive_layer, boolean_false_positive_op)

#define PROTO(T) \
  BINARY_ETI_INST_MACRO_DEV_DT(binary_cross_entropy_layer, T, El::Device::CPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(sigmoid_binary_cross_entropy_layer, T, El::Device::CPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(boolean_accuracy_layer, T, El::Device::CPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(boolean_false_negative_layer, T, El::Device::CPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(boolean_false_positive_layer, T, El::Device::CPU)

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
