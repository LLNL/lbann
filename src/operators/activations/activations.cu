////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/operators/activations/activations.hpp"

#include "lbann/base.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "../math/common.cuh"

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
template <typename DataT>
struct LogSigmoidOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const noexcept
  {
    if (x >= DataT(0.0)) {
      return -gpu_lib::log1p(gpu_lib::exp(-x));
    }
    else {
      return x - gpu_lib::log1p(gpu_lib::exp(x));
    }
  }
  inline __device__ DataT operator()(DataT const& x,
                                     DataT const& dy) const noexcept
  {
    return dy / (DataT(1.0) + gpu_lib::exp(x));
  }
};

/** SELU operator. */
template <typename DataT>
struct SeluOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const noexcept
  {
    DataT const alpha = 1.6732632423543772848170429916717;
    DataT const scale = 1.0507009873554804934193349852946;
    return (x > DataT(0.0) ? scale * x : scale * alpha * gpu_lib::expm1(x));
  }
  inline __device__ DataT operator()(DataT const& x,
                                     DataT const& dy) const noexcept
  {
    DataT const alpha = 1.6732632423543772848170429916717;
    DataT const scale = 1.0507009873554804934193349852946;
    return (x > DataT(0.0) ? dy * scale : dy * scale * alpha * gpu_lib::exp(x));
  }
};

/** Sigmoid operator. */
template <typename DataT>
struct SigmoidOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const noexcept
  {
    DataT const one = 1.;
    auto const& y = one / (one + gpu_lib::exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    auto const eps = gpu_lib::epsilon<DataT>();
    if (y <= eps) {
      return eps;
    }
    else if (y >= one - eps) {
      return one - eps;
    }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return y;
  }
  inline __device__ DataT operator()(DataT const& x,
                                     DataT const& dy) const noexcept
  {
    DataT const one = 1.;
    auto const& y = one / (one + gpu_lib::exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    auto const eps = gpu_lib::epsilon<DataT>();
    if (y <= eps || y >= one - eps) {
      return DataT(0.0);
    }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return dy * y * (one - y);
  }
};

/** Softplus operator. */
template <typename DataT>
struct SoftplusOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const noexcept
  {
    if (x > DataT(0.0)) {
      return gpu_lib::log1p(gpu_lib::exp(-x)) + x;
    }
    else {
      return gpu_lib::log1p(gpu_lib::exp(x));
    }
  }
  inline __device__ DataT operator()(DataT const& x,
                                     DataT const& dy) const noexcept
  {
    return dy / (DataT(1.0) + gpu_lib::exp(-x));
  }
};

/** Softsign operator. */
template <typename DataT>
struct SoftsignOpImpl
{
  inline __device__ DataT operator()(DataT const& x) const noexcept
  {
    return x / (DataT(1.0) + gpu_lib::abs(x));
  }
  inline __device__ DataT operator()(DataT const& x,
                                     DataT const& dy) const noexcept
  {
    auto const& denom = DataT(1.0) + gpu_lib::abs(x);
    return dy / (denom * denom);
  }
};

} // namespace

// Template instantiation
#define DEFINE_COMPUTE_OPS(OP_NAME)                                            \
  template <typename DataT, El::Device Device>                                 \
  void OP_NAME##Operator<DataT, Device>::fp_compute_local(                     \
    std::vector<ConstLocalInputTensorType> inputs,                             \
    std::vector<LocalOutputTensorType> outputs) const                          \
  {                                                                            \
    LBANN_ASSERT_DEBUG(inputs.size() == 1);                                    \
    LBANN_ASSERT_DEBUG(outputs.size() == 1);                                   \
    auto const& input = inputs.front().data();                                 \
    auto& output = outputs.front().data();                                     \
    El::EntrywiseMap(input, output, OP_NAME##OpImpl<DataT>{});                 \
  }                                                                            \
  template <typename DataT, El::Device Device>                                 \
  void OP_NAME##Operator<DataT, Device>::bp_compute_local(                     \
    std::vector<ConstLocalInputTensorType> inputs,                             \
    std::vector<ConstLocalOutputTensorType> grads_wrt_outputs,                 \
    std::vector<LocalInputTensorType> grads_wrt_inputs) const                  \
  {                                                                            \
    LBANN_ASSERT_DEBUG(inputs.size() == 1);                                    \
    LBANN_ASSERT_DEBUG(grads_wrt_outputs.size() == 1);                         \
    LBANN_ASSERT_DEBUG(grads_wrt_inputs.size() == 1);                          \
    auto const& input = inputs.front().data();                                 \
    auto const& grad_wrt_output = grads_wrt_outputs.front().data();            \
    auto& grad_wrt_input = grads_wrt_inputs.front().data();                    \
    internal::EntrywiseZipInto(input,                                          \
                               grad_wrt_output,                                \
                               grad_wrt_input,                                 \
                               OP_NAME##OpImpl<DataT>{});                      \
  }

DEFINE_COMPUTE_OPS(LogSigmoid)
DEFINE_COMPUTE_OPS(Selu)
DEFINE_COMPUTE_OPS(Sigmoid)
DEFINE_COMPUTE_OPS(Softplus)
DEFINE_COMPUTE_OPS(Softsign)

#define PROTO(T)                                                               \
  template class LogSigmoidOperator<T, El::Device::GPU>;                       \
  template class SeluOperator<T, El::Device::GPU>;                             \
  template class SigmoidOperator<T, El::Device::GPU>;                          \
  template class SoftplusOperator<T, El::Device::GPU>;                         \
  template class SoftsignOperator<T, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
