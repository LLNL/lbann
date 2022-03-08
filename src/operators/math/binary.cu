////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#include "lbann/operators/math/binary.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "common.cuh"

namespace lbann {

namespace {

// =========================================================
// Operator objects for entry-wise binary layers
// =========================================================
// Note: Binary operator corresponds to forward prop step
// (\f$ y = f(x_1,x_2) \f$) and 5-ary operator corresponds
// to back prop step
// (\f$ \frac{dL}{dx_i} = \frac{dL}{dy} \frac{df}{dx_i}(x_1,x_2) \f$).

/** Add operator. */
template <typename DataT>
struct AddOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 + x2;
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = dy;
    dx2 = dy;
  }
};

/** Subtract operator. */
template <typename DataT>
struct SubtractOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 - x2;
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = dy;
    dx2 = -dy;
  }
};

/** Multiply operator. */
template <typename DataT>
struct MultiplyOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 * x2;
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = dy * x2;
    dx2 = dy * x1;
  }
};

/** Divide operator. */
template <typename DataT>
struct DivideOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 / x2;
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = dy / x2;
    dx2 = -dy * x1 / (x2*x2);
  }
};

/** Modulo operator. */
template <typename DataT>
struct ModOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return gpu_lib::mod(x1, x2);
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = dy;
    dx2 = -dy * gpu_lib::floor(x1 / x2);
  }
};

/** Power operator. */
template <typename DataT>
struct PowOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return gpu_lib::pow(x1, x2);
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = dy * x2 * gpu_lib::pow(x1, x2 - DataT(1.0));
    dx2 = dy * gpu_lib::log(x1) * gpu_lib::pow(x1, x2);
  }
};

/** Safe divide operator.
 *  If a standard division produces an infinity or NaN, zero is output
 *  instead.
 */
template <typename DataT>
struct SafeDivideOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    auto const& y = x1 / x2;
    if (gpu_lib::isfinite(y)) { return y; }
    else             { return DataT(0.0); }
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    auto const& y = x1 / x2;
    if (gpu_lib::isfinite(y)) {
      dx1 = dy / x2;
      dx2 = -dy * x1 / (x2*x2);
    }
    else {
      dx1 = DataT(0.0);
      dx2 = DataT(0.0);
    }
  }
};

/** Squared difference operator. */
template <typename DataT>
struct SquaredDifferenceOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    auto const& diff = x1 - x2;
    return diff * diff;
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = dy * DataT(2.) * (x1-x2);
    dx2 = dy * DataT(2.) * (x2-x1);
  }
};

/** Maximum operator. */
template <typename DataT>
struct MaxOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return gpu_lib::max(x1, x2);
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    if (x1 > x2) {
      dx1 = dy;
      dx2 = DataT(0.0);
    }
    else if (x2 > x1) {
      dx1 = DataT(0.0);
      dx2 = dy;
    }
    else {
      dx1 = dy / DataT(2.);
      dx2 = dy / DataT(2.);
    }
  }
};

/** Minimum operator. */
template <typename DataT>
struct MinOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return gpu_lib::min(x1, x2);
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    if (x1 < x2) {
      dx1 = dy;
      dx2 = DataT(0.0);
    }
    else if (x2 < x1) {
      dx1 = DataT(0.0);
      dx2 = dy;
    }
    else {
      dx1 = dy / DataT(2.);
      dx2 = dy / DataT(2.);
    }
  }
};

/** Equal operator. */
template <typename DataT>
struct EqualOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 == x2 ? DataT(1.0) : DataT(0.0);
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& /*dy*/,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = DataT(0.0);
    dx2 = DataT(0.0);
  }
};

/** Not equal operator. */
template <typename DataT>
struct NotEqualOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 == x2 ? DataT(0.0) : DataT(1.0);
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& /*dy*/,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = DataT(0.0);
    dx2 = DataT(0.0);
  }
};

/** Less than operator. */
template <typename DataT>
struct LessOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 < x2 ? DataT(1.0) : DataT(0.0);
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& /*dy*/,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = DataT(0.0);
    dx2 = DataT(0.0);
  }
};

/** Less than or equal operator. */
template <typename DataT>
struct LessEqualOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 <= x2 ? DataT(1.0) : DataT(0.0);
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& /*dy*/,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = DataT(0.0);
    dx2 = DataT(0.0);
  }
};

/** Greater than operator. */
template <typename DataT>
struct GreaterOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 > x2 ? DataT(1.0) : DataT(0.0);
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& /*dy*/,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = DataT(0.0);
    dx2 = DataT(0.0);
  }
};

/** Greater than or equal operator. */
template <typename DataT>
struct GreaterEqualOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    return x1 >= x2 ? DataT(1.0) : DataT(0.0);
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& /*dy*/,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = DataT(0.0);
    dx2 = DataT(0.0);
  }
};

/** Logical and operator. */
template <typename DataT>
struct LogicalAndOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    auto const& b1 = x1 != DataT(0.0) && !gpu_lib::isnan(x1);
    auto const& b2 = x2 != DataT(0.0) && !gpu_lib::isnan(x2);
    return (b1 && b2) ? DataT(1.0) : DataT(0.0);
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& /*dy*/,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = DataT(0.0);
    dx2 = DataT(0.0);
  }
};

/** Logical or operator. */
template <typename DataT>
struct LogicalOrOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    auto const& b1 = x1 != DataT(0.0) && !gpu_lib::isnan(x1);
    auto const& b2 = x2 != DataT(0.0) && !gpu_lib::isnan(x2);
    return (b1 || b2) ? DataT(1.0) : DataT(0.0);
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& /*dy*/,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = DataT(0.0);
    dx2 = DataT(0.0);
  }
};

/** Logical xor operator. */
template <typename DataT>
struct LogicalXorOpImpl {
  inline __device__ DataT operator()(DataT const& x1, DataT const& x2) const
  {
    auto const& b1 = x1 != DataT(0.0) && !gpu_lib::isnan(x1);
    auto const& b2 = x2 != DataT(0.0) && !gpu_lib::isnan(x2);
    return (b1 || b2) && !(b1 && b2) ? DataT(1.0) : DataT(0.0);
  }
  inline __device__ void operator()(DataT const& /*x1*/,
                                    DataT const& /*x2*/,
                                    DataT const& /*dy*/,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    dx1 = DataT(0.0);
    dx2 = DataT(0.0);
  }
};

} // namespace

  // Template instantiation
#define DEFINE_COMPUTE_OPS(OP_NAME)                                     \
  template <typename DataT, El::Device Device>                          \
  void OP_NAME##Operator<DataT, Device>::fp_compute_local(              \
    std::vector<ConstLocalInputTensorType> inputs,                      \
    std::vector<LocalOutputTensorType> outputs) const                   \
  {                                                                     \
    LBANN_ASSERT_DEBUG(inputs.size() == 2);                             \
    LBANN_ASSERT_DEBUG(outputs.size() == 1);                            \
    auto const& input0 = inputs[0].data();                              \
    auto const& input1 = inputs[1].data();                              \
    auto& output = outputs.front().data();                              \
    LBANN_ASSERT(input0.Height() == input1.Height());                   \
    LBANN_ASSERT(input0.Width() == input1.Width());                     \
    internal::EntrywiseZipInto(input0,                                  \
                               input1,                                  \
                               output,                                  \
                               OP_NAME##OpImpl<DataT>{});               \
  }                                                                     \
  template <typename DataT, El::Device Device>                          \
  void OP_NAME##Operator<DataT, Device>::bp_compute_local(              \
    std::vector<ConstLocalInputTensorType> inputs,                      \
    std::vector<ConstLocalOutputTensorType> grads_wrt_outputs,          \
    std::vector<LocalInputTensorType> grads_wrt_inputs) const           \
  {                                                                     \
    LBANN_ASSERT_DEBUG(inputs.size() == 2);                             \
    LBANN_ASSERT_DEBUG(grads_wrt_outputs.size() == 1);                  \
    LBANN_ASSERT_DEBUG(grads_wrt_inputs.size() == 2);                   \
    auto const& input0 = inputs[0].data();                              \
    auto const& input1 = inputs[1].data();                              \
    auto const& grad_wrt_output = grads_wrt_outputs.front().data();     \
    auto& grad_wrt_input0 = grads_wrt_inputs[0].data();                 \
    auto& grad_wrt_input1 = grads_wrt_inputs[1].data();                 \
    LBANN_ASSERT(grad_wrt_input0.Height() == grad_wrt_input1.Height()); \
    LBANN_ASSERT(grad_wrt_input0.Width() == grad_wrt_input1.Width());   \
    internal::apply_binary_backprop_operator(input0,                    \
                                             input1,                    \
                                             grad_wrt_output,           \
                                             grad_wrt_input0,           \
                                             grad_wrt_input1,           \
                                             OP_NAME##OpImpl<DataT>{}); \
  }

DEFINE_COMPUTE_OPS(Add)
DEFINE_COMPUTE_OPS(Divide)
DEFINE_COMPUTE_OPS(Equal)
DEFINE_COMPUTE_OPS(Greater)
DEFINE_COMPUTE_OPS(GreaterEqual)
DEFINE_COMPUTE_OPS(Less)
DEFINE_COMPUTE_OPS(LessEqual)
DEFINE_COMPUTE_OPS(LogicalAnd)
DEFINE_COMPUTE_OPS(LogicalOr)
DEFINE_COMPUTE_OPS(LogicalXor)
DEFINE_COMPUTE_OPS(Max)
DEFINE_COMPUTE_OPS(Min)
DEFINE_COMPUTE_OPS(Mod)
DEFINE_COMPUTE_OPS(Multiply)
DEFINE_COMPUTE_OPS(NotEqual)
DEFINE_COMPUTE_OPS(Pow)
DEFINE_COMPUTE_OPS(SafeDivide)
DEFINE_COMPUTE_OPS(SquaredDifference)
DEFINE_COMPUTE_OPS(Subtract)

#define PROTO(T)                                                     \
  template class AddOperator<T, El::Device::GPU>;                    \
  template class DivideOperator<T, El::Device::GPU>;                 \
  template class EqualOperator<T, El::Device::GPU>;                  \
  template class GreaterEqualOperator<T, El::Device::GPU>;           \
  template class GreaterOperator<T, El::Device::GPU>;                \
  template class LessEqualOperator<T, El::Device::GPU>;              \
  template class LessOperator<T, El::Device::GPU>;                   \
  template class LogicalAndOperator<T, El::Device::GPU>;             \
  template class LogicalOrOperator<T, El::Device::GPU>;              \
  template class LogicalXorOperator<T, El::Device::GPU>;             \
  template class MaxOperator<T, El::Device::GPU>;                    \
  template class MinOperator<T, El::Device::GPU>;                    \
  template class ModOperator<T, El::Device::GPU>;                    \
  template class MultiplyOperator<T, El::Device::GPU>;               \
  template class NotEqualOperator<T, El::Device::GPU>;               \
  template class PowOperator<T, El::Device::GPU>;                    \
  template class SafeDivideOperator<T, El::Device::GPU>;             \
  template class SquaredDifferenceOperator<T, El::Device::GPU>;      \
  template class SubtractOperator<T, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
