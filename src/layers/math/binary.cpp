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

#include "lbann/layers/math/binary.hpp"
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
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
      BinaryBackPropOperator op;
      op(x1_buffer[i], x2_buffer[i], dy_buffer[i],
         dx1_buffer[i], dx2_buffer[i]);
    }
  } else {
#pragma omp parallel for collapse(2)
    for (El::Int col = 0; col < x1.Width(); ++col) {
      for (El::Int row = 0; row < x2.Height(); ++row) {
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
struct add_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 + x2;
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    dx1 = dy;
    dx2 = dy;
  }
};

/** Subtract operator. */
struct subtract_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 - x2;
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    dx1 = dy;
    dx2 = -dy;
  }
};
  
/** Multiply operator. */
struct multiply_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 * x2;
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    dx1 = dy * x2;
    dx2 = dy * x1;
  }
};

/** Divide operator. */
struct divide_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 / x2;
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    dx1 = dy / x2;
    dx2 = -dy * x1 / (x2*x2);
  }
};
  
/** Modulo operator. */
struct mod_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return std::fmod(x1, x2);
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    dx1 = dy;
    dx2 = -dy * std::floor(x1 / x2);
  }
};

/** Power operator. */
struct pow_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return std::pow(x1, x2);
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {

    dx1 = dy * x2 * std::pow(x1, x2 - one);
    dx2 = dy * std::log(x1) * std::pow(x1, x2);
  }
};

/** Maximum operator. */
struct max_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return std::max(x1, x2);
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    if (x1 > x2) {
      dx1 = dy;
      dx2 = zero;
    } else if (x2 > x1) {
      dx1 = zero;
      dx2 = dy;
    } else {
      dx1 = dy / 2;
      dx2 = dy / 2;
    }
  }
};

/** Minimum operator. */
struct min_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return std::min(x1, x2);
  }
  inline void operator()(const DataType& x1,
                         const DataType& x2,
                         const DataType& dy,
                         DataType& dx1,
                         DataType& dx2) const {
    if (x1 < x2) {
      dx1 = dy;
      dx2 = zero;
    } else if (x2 < x1) {
      dx1 = zero;
      dx2 = dy;
    } else {
      dx1 = dy / 2;
      dx2 = dy / 2;
    }
  }
};

/** Equal operator. */
struct equal_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 == x2 ? one : zero;
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

/** Not equal operator. */
struct not_equal_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 == x2 ? zero : one;
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

/** Less than operator. */
struct less_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 < x2 ? one : zero;
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

/** Less than or equal operator. */
struct less_equal_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 <= x2 ? one : zero;
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

/** Greater than operator. */
struct greater_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 > x2 ? one : zero;
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

/** Greater than or equal operator. */
struct greater_equal_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    return x1 >= x2 ? one : zero;
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

/** Logical and operator. */
struct and_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    const bool b1 = x1 != zero && !std::isnan(x1);
    const bool b2 = x2 != zero && !std::isnan(x2);
    return (b1 && b2) ? one : zero;
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

/** Logical or operator. */
struct or_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    const bool b1 = x1 != zero && !std::isnan(x1);
    const bool b2 = x2 != zero && !std::isnan(x2);
    return (b1 || b2) ? one : zero;
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

/** Logical xor operator. */
struct xor_op {
  inline DataType operator()(const DataType& x1,
                             const DataType& x2) const {
    const bool b1 = x1 != zero && !std::isnan(x1);
    const bool b2 = x2 != zero && !std::isnan(x2);
    return (b1 || b2) && !(b1 && b2) ? one : zero;
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
  INSTANTIATE(add_layer, add_op)
  INSTANTIATE(subtract_layer, subtract_op)
  INSTANTIATE(multiply_layer, multiply_op)
  INSTANTIATE(divide_layer, divide_op)
  INSTANTIATE(mod_layer, mod_op)
  INSTANTIATE(pow_layer, pow_op)
  INSTANTIATE(max_layer, max_op)
  INSTANTIATE(min_layer, min_op)
  INSTANTIATE(equal_layer, equal_op)
  INSTANTIATE(not_equal_layer, not_equal_op)
  INSTANTIATE(less_layer, less_op)
  INSTANTIATE(less_equal_layer, less_equal_op)
  INSTANTIATE(greater_layer, greater_op)
  INSTANTIATE(greater_equal_layer, greater_equal_op)
  INSTANTIATE(and_layer, and_op)
  INSTANTIATE(or_layer, or_op)
  INSTANTIATE(xor_layer, xor_op)
  
} // namespace lbann
