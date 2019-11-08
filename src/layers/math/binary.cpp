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

#define LBANN_BINARY_LAYER_INSTANTIATE
#include "lbann/layers/math/binary.hpp"
#include "lbann/utils/entrywise_operator.hpp"

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
template <typename TensorDataType, typename BinaryBackPropOperator>
void apply_binary_backprop_operator(const El::AbstractMatrix<TensorDataType>& x1,
                                    const El::AbstractMatrix<TensorDataType>& x2,
                                    const El::AbstractMatrix<TensorDataType>& dy,
                                    El::AbstractMatrix<TensorDataType>& dx1,
                                    El::AbstractMatrix<TensorDataType>& dx2) {
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

/** Add operator. */
template <typename TensorDataType>
struct add_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 + x2;
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = dy;
    dx2 = dy;
  }
};

/** Subtract operator. */
template <typename TensorDataType>
struct subtract_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 - x2;
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = dy;
    dx2 = -dy;
  }
};

/** Multiply operator. */
template <typename TensorDataType>
struct multiply_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 * x2;
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = dy * x2;
    dx2 = dy * x1;
  }
};

/** Divide operator. */
template <typename TensorDataType>
struct divide_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 / x2;
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = dy / x2;
    dx2 = -dy * x1 / (x2*x2);
  }
};

/** Modulo operator. */
template <typename TensorDataType>
struct mod_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return std::fmod(x1, x2);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = dy;
    dx2 = -dy * std::floor(x1 / x2);
  }
};

/** Power operator. */
template <typename TensorDataType>
struct pow_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return std::pow(x1, x2);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {

    dx1 = dy * x2 * std::pow(x1, x2 - TensorDataType(1));
    dx2 = dy * std::log(x1) * std::pow(x1, x2);
  }
};

/** Safe divide operator.
 *  If a standard division produces an infinity or NaN, zero is output
 *  instead.
 */
template <typename TensorDataType>
struct safe_divide_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    const auto& y = x1 / x2;
    if (std::isfinite(y)) { return y; }
    else                  { return TensorDataType(0); }
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    const auto& y = x1 / x2;
    if (std::isfinite(y)) {
      dx1 = dy / x2;
      dx2 = -dy * x1 / (x2*x2);
    } else {
      dx1 = TensorDataType(0);
      dx2 = TensorDataType(0);
    }
  }
};

/** Squared difference operator. */
template <typename TensorDataType>
struct squared_difference_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    const auto& diff = x1 - x2;
    return diff * diff;
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = dy * 2*(x1-x2);
    dx2 = dy * 2*(x2-x1);
  }
};

/** Maximum operator. */
template <typename TensorDataType>
struct max_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return std::max(x1, x2);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    if (x1 > x2) {
      dx1 = dy;
      dx2 = TensorDataType(0);
    } else if (x2 > x1) {
      dx1 = TensorDataType(0);
      dx2 = dy;
    } else {
      dx1 = dy / 2;
      dx2 = dy / 2;
    }
  }
};

/** Minimum operator. */
template <typename TensorDataType>
struct min_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return std::min(x1, x2);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    if (x1 < x2) {
      dx1 = dy;
      dx2 = TensorDataType(0);
    } else if (x2 < x1) {
      dx1 = TensorDataType(0);
      dx2 = dy;
    } else {
      dx1 = dy / 2;
      dx2 = dy / 2;
    }
  }
};

/** Equal operator. */
template <typename TensorDataType>
struct equal_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 == x2 ? TensorDataType(1) : TensorDataType(0);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = TensorDataType(0);
    dx2 = TensorDataType(0);
  }
};

/** Not equal operator. */
template <typename TensorDataType>
struct not_equal_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 == x2 ? TensorDataType(0) : TensorDataType(1);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = TensorDataType(0);
    dx2 = TensorDataType(0);
  }
};

/** Less than operator. */
template <typename TensorDataType>
struct less_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 < x2 ? TensorDataType(1) : TensorDataType(0);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = TensorDataType(0);
    dx2 = TensorDataType(0);
  }
};

/** Less than or equal operator. */
template <typename TensorDataType>
struct less_equal_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 <= x2 ? TensorDataType(1) : TensorDataType(0);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = TensorDataType(0);
    dx2 = TensorDataType(0);
  }
};

/** Greater than operator. */
template <typename TensorDataType>
struct greater_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 > x2 ? TensorDataType(1) : TensorDataType(0);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = TensorDataType(0);
    dx2 = TensorDataType(0);
  }
};

/** Greater than or equal operator. */
template <typename TensorDataType>
struct greater_equal_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 >= x2 ? TensorDataType(1) : TensorDataType(0);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = TensorDataType(0);
    dx2 = TensorDataType(0);
  }
};

/** Logical and operator. */
template <typename TensorDataType>
struct logical_and_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    const auto& b1 = x1 != TensorDataType(0) && !std::isnan(x1);
    const auto& b2 = x2 != TensorDataType(0) && !std::isnan(x2);
    return (b1 && b2) ? TensorDataType(1) : TensorDataType(0);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = TensorDataType(0);
    dx2 = TensorDataType(0);
  }
};

/** Logical or operator. */
template <typename TensorDataType>
struct logical_or_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    const auto& b1 = x1 != TensorDataType(0) && !std::isnan(x1);
    const auto& b2 = x2 != TensorDataType(0) && !std::isnan(x2);
    return (b1 || b2) ? TensorDataType(1) : TensorDataType(0);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = TensorDataType(0);
    dx2 = TensorDataType(0);
  }
};

/** Logical xor operator. */
template <typename TensorDataType>
struct logical_xor_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    const auto& b1 = x1 != TensorDataType(0) && !std::isnan(x1);
    const auto& b2 = x2 != TensorDataType(0) && !std::isnan(x2);
    return (b1 || b2) && !(b1 && b2) ? TensorDataType(1) : TensorDataType(0);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = TensorDataType(0);
    dx2 = TensorDataType(0);
  }
};

} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                                                          \
  template <typename TensorDataType>                                                                    \
  void fp_compute_impl(layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::CPU>& l) {        \
    apply_entrywise_binary_operator<TensorDataType, op<TensorDataType>>(l.get_prev_activations(0),      \
                                                                        l.get_prev_activations(1),      \
                                                                        l.get_activations());           \
  }                                                                                                     \
  template <typename TensorDataType>                                                                    \
  void bp_compute_impl(layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::CPU>& l) {        \
    apply_binary_backprop_operator<TensorDataType, op<TensorDataType>>(l.get_local_prev_activations(0), \
                                                       l.get_local_prev_activations(1),                 \
                                                       l.get_local_prev_error_signals(),                \
                                                       l.get_local_error_signals(0),                    \
                                                       l.get_local_error_signals(1));                   \
  }                                                                                                     \
  template <typename TensorDataType>                                                                    \
  void fp_compute_impl(layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l) {         \
    apply_entrywise_binary_operator<TensorDataType, op<TensorDataType>>(l.get_prev_activations(0),      \
                                                                        l.get_prev_activations(1),      \
                                                                        l.get_activations());           \
  }                                                                                                     \
  template <typename TensorDataType>                                                                    \
  void bp_compute_impl(layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l) {         \
    apply_binary_backprop_operator<TensorDataType, op<TensorDataType>>(l.get_local_prev_activations(0), \
                                                       l.get_local_prev_activations(1),                 \
                                                       l.get_local_prev_error_signals(),                \
                                                       l.get_local_error_signals(0),                    \
                                                       l.get_local_error_signals(1));                   \
  }                                                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device>                             \
  void layer<TensorDataType, Layout, Device>::fp_compute() {                                            \
    fp_compute_impl<TensorDataType>(*this);                                                             \
  }                                                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device>                             \
  void layer<TensorDataType, Layout, Device>::bp_compute() {                                            \
    bp_compute_impl<TensorDataType>(*this);                                                             \
  }                                                                                                     \
  BINARY_ETI_INST_MACRO_DEV(layer, El::Device::CPU)

INSTANTIATE(add_layer, add_op);
INSTANTIATE(subtract_layer, subtract_op);
INSTANTIATE(multiply_layer, multiply_op);
INSTANTIATE(divide_layer, divide_op);
INSTANTIATE(mod_layer, mod_op);
INSTANTIATE(pow_layer, pow_op);
INSTANTIATE(safe_divide_layer, safe_divide_op);
INSTANTIATE(squared_difference_layer, squared_difference_op);
INSTANTIATE(max_layer, max_op);
INSTANTIATE(min_layer, min_op);
INSTANTIATE(equal_layer, equal_op);
INSTANTIATE(not_equal_layer, not_equal_op);
INSTANTIATE(less_layer, less_op);
INSTANTIATE(less_equal_layer, less_equal_op);
INSTANTIATE(greater_layer, greater_op);
INSTANTIATE(greater_equal_layer, greater_equal_op);
INSTANTIATE(logical_and_layer, logical_and_op);
INSTANTIATE(logical_or_layer, logical_or_op);
INSTANTIATE(logical_xor_layer, logical_xor_op);

} // namespace lbann
