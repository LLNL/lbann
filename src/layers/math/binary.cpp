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
template <template <typename> class Op, typename TensorDataType>
void apply_binary_backprop_operator(
  const El::AbstractMatrix<TensorDataType>& x1,
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
    using std::fmod;
    return fmod(x1, x2);
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
    return El::Pow(x1, x2);
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {

    dx1 = dy * x2 * std::pow(x1, x2 - El::TypeTraits<TensorDataType>::One());
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
    else                  { return El::TypeTraits<TensorDataType>::Zero(); }
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
      dx1 = El::TypeTraits<TensorDataType>::Zero();
      dx2 = El::TypeTraits<TensorDataType>::Zero();
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
      dx2 = El::TypeTraits<TensorDataType>::Zero();
    } else if (x2 > x1) {
      dx1 = El::TypeTraits<TensorDataType>::Zero();
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
      dx2 = El::TypeTraits<TensorDataType>::Zero();
    } else if (x2 < x1) {
      dx1 = El::TypeTraits<TensorDataType>::Zero();
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
    return x1 == x2 ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = El::TypeTraits<TensorDataType>::Zero();
    dx2 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Not equal operator. */
template <typename TensorDataType>
struct not_equal_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 == x2 ? El::TypeTraits<TensorDataType>::Zero() : El::TypeTraits<TensorDataType>::One();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = El::TypeTraits<TensorDataType>::Zero();
    dx2 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Less than operator. */
template <typename TensorDataType>
struct less_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 < x2 ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = El::TypeTraits<TensorDataType>::Zero();
    dx2 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Less than or equal operator. */
template <typename TensorDataType>
struct less_equal_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 <= x2 ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = El::TypeTraits<TensorDataType>::Zero();
    dx2 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Greater than operator. */
template <typename TensorDataType>
struct greater_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 > x2 ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = El::TypeTraits<TensorDataType>::Zero();
    dx2 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Greater than or equal operator. */
template <typename TensorDataType>
struct greater_equal_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    return x1 >= x2 ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = El::TypeTraits<TensorDataType>::Zero();
    dx2 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Logical and operator. */
template <typename TensorDataType>
struct logical_and_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    const auto& b1 = x1 != El::TypeTraits<TensorDataType>::Zero() && !std::isnan(x1);
    const auto& b2 = x2 != El::TypeTraits<TensorDataType>::Zero() && !std::isnan(x2);
    return (b1 && b2) ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = El::TypeTraits<TensorDataType>::Zero();
    dx2 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Logical or operator. */
template <typename TensorDataType>
struct logical_or_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    const auto& b1 = x1 != El::TypeTraits<TensorDataType>::Zero() && !std::isnan(x1);
    const auto& b2 = x2 != El::TypeTraits<TensorDataType>::Zero() && !std::isnan(x2);
    return (b1 || b2) ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = El::TypeTraits<TensorDataType>::Zero();
    dx2 = El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Logical xor operator. */
template <typename TensorDataType>
struct logical_xor_op {
  inline TensorDataType operator()(const TensorDataType& x1,
                             const TensorDataType& x2) const {
    const auto& b1 = x1 != El::TypeTraits<TensorDataType>::Zero() && !std::isnan(x1);
    const auto& b2 = x2 != El::TypeTraits<TensorDataType>::Zero() && !std::isnan(x2);
    return (b1 || b2) && !(b1 && b2) ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline void operator()(const TensorDataType& x1,
                         const TensorDataType& x2,
                         const TensorDataType& dy,
                         TensorDataType& dx1,
                         TensorDataType& dx2) const {
    dx1 = El::TypeTraits<TensorDataType>::Zero();
    dx2 = El::TypeTraits<TensorDataType>::Zero();
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

DEFINE_COMPUTE_OPS(add_layer, add_op)
DEFINE_COMPUTE_OPS(subtract_layer, subtract_op)
DEFINE_COMPUTE_OPS(multiply_layer, multiply_op)
DEFINE_COMPUTE_OPS(divide_layer, divide_op)
DEFINE_COMPUTE_OPS(mod_layer, mod_op)
DEFINE_COMPUTE_OPS(pow_layer, pow_op)
DEFINE_COMPUTE_OPS(safe_divide_layer, safe_divide_op)
DEFINE_COMPUTE_OPS(squared_difference_layer, squared_difference_op)
DEFINE_COMPUTE_OPS(max_layer, max_op)
DEFINE_COMPUTE_OPS(min_layer, min_op)
DEFINE_COMPUTE_OPS(equal_layer, equal_op)
DEFINE_COMPUTE_OPS(not_equal_layer, not_equal_op)
DEFINE_COMPUTE_OPS(less_layer, less_op)
DEFINE_COMPUTE_OPS(less_equal_layer, less_equal_op)
DEFINE_COMPUTE_OPS(greater_layer, greater_op)
DEFINE_COMPUTE_OPS(greater_equal_layer, greater_equal_op)
DEFINE_COMPUTE_OPS(logical_and_layer, logical_and_op)
DEFINE_COMPUTE_OPS(logical_or_layer, logical_or_op)
DEFINE_COMPUTE_OPS(logical_xor_layer, logical_xor_op)

#define PROTO(T)                                                       \
  BINARY_ETI_INST_MACRO_DEV_DT(add_layer, T, El::Device::CPU);         \
  BINARY_ETI_INST_MACRO_DEV_DT(subtract_layer, T, El::Device::CPU);    \
  BINARY_ETI_INST_MACRO_DEV_DT(multiply_layer, T, El::Device::CPU);    \
  BINARY_ETI_INST_MACRO_DEV_DT(divide_layer, T, El::Device::CPU);      \
  BINARY_ETI_INST_MACRO_DEV_DT(mod_layer, T, El::Device::CPU);         \
  BINARY_ETI_INST_MACRO_DEV_DT(pow_layer, T, El::Device::CPU);         \
  BINARY_ETI_INST_MACRO_DEV_DT(safe_divide_layer, T, El::Device::CPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(squared_difference_layer, T, El::Device::CPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(max_layer, T, El::Device::CPU);         \
  BINARY_ETI_INST_MACRO_DEV_DT(min_layer, T, El::Device::CPU);         \
  BINARY_ETI_INST_MACRO_DEV_DT(equal_layer, T, El::Device::CPU);       \
  BINARY_ETI_INST_MACRO_DEV_DT(not_equal_layer, T, El::Device::CPU);   \
  BINARY_ETI_INST_MACRO_DEV_DT(less_layer, T, El::Device::CPU);        \
  BINARY_ETI_INST_MACRO_DEV_DT(less_equal_layer, T, El::Device::CPU);  \
  BINARY_ETI_INST_MACRO_DEV_DT(greater_layer, T, El::Device::CPU);     \
  BINARY_ETI_INST_MACRO_DEV_DT(greater_equal_layer, T, El::Device::CPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(logical_and_layer, T, El::Device::CPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(logical_or_layer, T, El::Device::CPU);  \
  BINARY_ETI_INST_MACRO_DEV_DT(logical_xor_layer, T, El::Device::CPU)

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
