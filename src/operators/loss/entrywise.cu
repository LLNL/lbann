////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#include "lbann/operators/loss/entrywise.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "../math/common.cuh"

namespace lbann {

namespace {

// =========================================================
// Operator objects for entry-wise binary layers
// =========================================================
// Note: Binary operator corresponds to forward prop step
// (\f$ y = f(x_1,x_2) \f$) and 5-ary operator corresponds
// to back prop step
// (\f$ \frac{dL}{dx_i} = \frac{dL}{dy} \frac{df}{dx_i}(x_1,x_2) \f$).

/** Binary cross entropy operator. */
template <typename DataT>
struct BinaryCrossEntropyOpImpl
{
  inline __device__ DataT operator()(DataT const& x1,
                                     DataT const& x2) const
  {
    DataT const zero = 0.;
    DataT const one = 1.;
    DataT y = zero;
    if (x2 > zero) { y += -x2 * gpu_lib::log(x1); }
    if (x2 < one)  { y += -(one-x2) * gpu_lib::log(one-x1); }
    return y;
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    DataT const zero = 0.;
    DataT const one = 1.;
    dx1 = zero;
    dx2 = zero;
    if (dy == zero) { return; }
    if (x2 > zero) {
      dx1 += -x2 / x1 * dy;
      dx2 += -gpu_lib::log(x1) * dy;
    }
    if (x2 < one)  {
      dx1 += (one-x2) / (one-x1) * dy;
      dx2 += gpu_lib::log(one-x1) * dy;
    }
  }
};

/** Sigmoid binary cross entropy operator.
 *  Equivalent to applying a sigmoid function to the first operand and
 *  then computing the binary cross entropy. Numerically stable
 *  implementation is taken from
 *  https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits.
 */
template <typename DataT>
struct SigmoidBinaryCrossEntropyOpImpl
{
  inline __device__ DataT operator()(DataT const& x1,
                                     DataT const& x2) const
  {
    DataT const zero = 0.;
    DataT const one = 1.;
    auto const& z = gpu_lib::max(zero, gpu_lib::min(x2, one));
    if (x1 > zero) {
      return (one - z) * x1 + gpu_lib::log1p(gpu_lib::exp(-x1));
    } else {
      return - x1 * z + gpu_lib::log1p(gpu_lib::exp(x1));
    }
  }
  inline __device__ void operator()(DataT const& x1,
                                    DataT const& x2,
                                    DataT const& dy,
                                    DataT& dx1,
                                    DataT& dx2) const
  {
    DataT const zero = 0.;
    DataT const one = 1.;
    auto const& z = gpu_lib::max(zero, gpu_lib::min(x2, one));
    if (x1 > zero) {
      dx1 = -z + one / (one + gpu_lib::exp(-x1));
    } else {
      dx1 = one - z - one / (one + gpu_lib::exp(x1));
    }
    dx1 *= dy;
    dx2 = (x2 == z) ? -x1 * dy : zero;
  }
};

/** Boolean accuracy operator. */
template <typename DataT>
struct BooleanAccuracyOpImpl
{
  inline __device__ DataT operator()(DataT const& x1,
                                     DataT const& x2) const
  {
    auto const& b1 = x1 >= DataT(0.5);
    auto const& b2 = x2 >= DataT(0.5);
    return b1 == b2 ? DataT(1.0) : DataT(0.0);
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

/** Boolean false negative operator. */
template <typename DataT>
struct BooleanFalseNegativeOpImpl
{
  inline __device__ DataT operator()(DataT const& x1,
                                     DataT const& x2) const
  {
    auto const& b1 = x1 >= DataT(0.5);
    auto const& b2 = x2 >= DataT(0.5);
    return (!b1 && b2) ? DataT(1.0) : DataT(0.0);
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

/** Boolean false positive operator. */
template <typename DataT>
struct BooleanFalsePositiveOpImpl
{
  inline __device__ DataT operator()(DataT const& x1,
                                     DataT const& x2) const
  {
    auto const& b1 = x1 >= DataT(0.5);
    auto const& b2 = x2 >= DataT(0.5);
    return (b1 && !b2) ? DataT(1.0) : DataT(0.0);
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
    internal::apply_binary_backprop_operator(input0,                    \
                                             input1,                    \
                                             grad_wrt_output,           \
                                             grad_wrt_input0,           \
                                             grad_wrt_input1,           \
                                             OP_NAME##OpImpl<DataT>{}); \
  }

DEFINE_COMPUTE_OPS(BinaryCrossEntropy)
DEFINE_COMPUTE_OPS(SigmoidBinaryCrossEntropy)
DEFINE_COMPUTE_OPS(BooleanAccuracy)
DEFINE_COMPUTE_OPS(BooleanFalseNegative)
DEFINE_COMPUTE_OPS(BooleanFalsePositive)

#define PROTO(T)                                                        \
  template class BinaryCrossEntropyOperator<T, El::Device::GPU>;        \
  template class SigmoidBinaryCrossEntropyOperator<T, El::Device::GPU>; \
  template class BooleanAccuracyOperator<T, El::Device::GPU>;           \
  template class BooleanFalseNegativeOperator<T, El::Device::GPU>;      \
  template class BooleanFalsePositiveOperator<T, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
