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

#include "lbann/layers/loss/entrywise.hpp"
#include "lbann/utils/entrywise_operator.hpp"

namespace lbann {

namespace {

// Helpful constants
constexpr DataType zero = 0;
constexpr DataType one = 1;

/** Apply a binary backprop operator to CPU data.
 *  The input and output data must be on CPU and must have the same
 *  dimensions. Given a binary function \f$ y = f(x_1,x_2) \f$, the
 *  corresponding BinaryBackPropOperator is a 5-ary function with the
 *  arguments \f$ x_1 \f$, \f$ x_2 \f$, \f$ dL/dy \f$, \f$ dL/dx_1\f$,
 *  \f$ dL/dx_2 \f$. The last two arguments should be overwritten when
 *  the BinaryBackPropOperator is called.
 */
template <typename BinaryBackPropOperator>
void apply_binary_backprop_operator(const AbsMat& x1,
                                    const AbsMat& x2,
                                    const AbsMat& dy,
                                    AbsMat& dx1,
                                    AbsMat& dx2) {
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
struct binary_cross_entropy_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    DataType y = zero;
    if (x2 > zero) { y += -x2 * std::log(x1); }
    if (x2 < one)  { y += -(one-x2) * std::log(one-x1); }
    return y;
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    dx1 = zero;
    dx2 = zero;
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
struct sigmoid_binary_cross_entropy_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    const auto& z = std::max(zero, std::min(x2, one));
    if (x1 > zero) {
      return (one - z) * x1 + std::log1p(std::exp(-x1));
    } else {
      return - x1 * z + std::log1p(std::exp(x1));
    }
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    const auto& z = std::max(zero, std::min(x2, one));
    if (x1 > zero) {
      dx1 = -z + 1 / (one + std::exp(-x1));
    } else {
      dx1 = one - z - 1 / (one + std::exp(x1));
    }
    dx1 *= dy;
    dx2 = (x2 == z) ? -x1 * dy : zero;
  }
};

/** Boolean accuracy operator. */
struct boolean_accuracy_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    const auto& b1 = x1 >= DataType(0.5);
    const auto& b2 = x2 >= DataType(0.5);
    return b1 == b2 ? one : zero;
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    dx1 = zero;
    dx2 = zero;
  }
};

/** Boolean false negative operator. */
struct boolean_false_negative_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    const auto& b1 = x1 >= DataType(0.5);
    const auto& b2 = x2 >= DataType(0.5);
    return (!b1 && b2) ? one : zero;
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    dx1 = zero;
    dx2 = zero;
  }
};

/** Boolean false positive operator. */
struct boolean_false_positive_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    const auto& b1 = x1 >= DataType(0.5);
    const auto& b2 = x2 >= DataType(0.5);
    return (b1 && !b2) ? one : zero;
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    dx1 = zero;
    dx2 = zero;
  }
};

} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                          \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::CPU>              \
         ::fp_compute() {                                               \
    apply_entrywise_binary_operator<op>(get_prev_activations(0),        \
                                        get_prev_activations(1),        \
                                        get_activations());             \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::CPU>              \
         ::bp_compute() {                                               \
    apply_binary_backprop_operator<op>(get_local_prev_activations(0),   \
                                       get_local_prev_activations(1),   \
                                       get_local_prev_error_signals(),  \
                                       get_local_error_signals(0),      \
                                       get_local_error_signals(1));     \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::CPU>               \
         ::fp_compute() {                                               \
    apply_entrywise_binary_operator<op>(get_prev_activations(0),        \
                                        get_prev_activations(1),        \
                                        get_activations());             \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::CPU>               \
  ::bp_compute() {                                                      \
    apply_binary_backprop_operator<op>(get_local_prev_activations(0),   \
                                       get_local_prev_activations(1),   \
                                       get_local_prev_error_signals(),  \
                                       get_local_error_signals(0),      \
                                       get_local_error_signals(1));     \
  }
  INSTANTIATE(binary_cross_entropy_layer, binary_cross_entropy_op)
  INSTANTIATE(sigmoid_binary_cross_entropy_layer, sigmoid_binary_cross_entropy_op)
  INSTANTIATE(boolean_accuracy_layer, boolean_accuracy_op)
  INSTANTIATE(boolean_false_negative_layer, boolean_false_negative_op)
  INSTANTIATE(boolean_false_positive_layer, boolean_false_positive_op)

} // namespace lbann
