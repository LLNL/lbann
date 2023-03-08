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

#include "lbann/operators/math/binary_with_constant.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "common.cuh"

namespace {
// Device lambdas cannot be used in functions with private or
// protected access within their class. So we move them to functions
// that aren't in their class at all. "The only problem you can't
// solve with more indirection is too much indirection."
template <typename T>
void ApplyAddFP(T c,
                El::Matrix<T, El::Device::GPU> const& x,
                El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) { return x + c; });
}

template <typename T>
void ApplyScaleFP(T c,
                  El::Matrix<T, El::Device::GPU> const& x,
                  El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) { return x * c; });
}

template <typename T>
void ApplySubtractFP(T c,
                     El::Matrix<T, El::Device::GPU> const& x,
                     El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) { return x - c; });
}

template <typename T>
void ApplyCSubtractFP(T c,
                      El::Matrix<T, El::Device::GPU> const& x,
                      El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) { return c - x; });
}

template <typename T>
void ApplyCSubtractBP(El::Matrix<T, El::Device::GPU> const& x,
                      El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [] __device__(T const& x) { return -x; });
}

template <typename T>
void ApplyMaxFP(T c,
                El::Matrix<T, El::Device::GPU> const& x,
                El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) {
    return ::lbann::gpu_lib::max(c, x);
  });
}

template <typename T>
void ApplyMaxBP(T c,
                El::Matrix<T, El::Device::GPU> const& in,
                El::Matrix<T, El::Device::GPU> const& grad_wrt_out,
                El::Matrix<T, El::Device::GPU>& grad_wrt_in)
{
  ::lbann::internal::EntrywiseZipInto(
    in,
    grad_wrt_out,
    grad_wrt_in,
    [c] __device__(T const& x, T const& dy) {
      return (x < c ? (T)0. : (x > c ? dy : dy / (T)2.));
    });
}

template <typename T>
void ApplyMinFP(T c,
                El::Matrix<T, El::Device::GPU> const& x,
                El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) {
    return ::lbann::gpu_lib::min(c, x);
  });
}

template <typename T>
void ApplyMinBP(T c,
                El::Matrix<T, El::Device::GPU> const& in,
                El::Matrix<T, El::Device::GPU> const& grad_wrt_out,
                El::Matrix<T, El::Device::GPU>& grad_wrt_in)
{
  ::lbann::internal::EntrywiseZipInto(
    in,
    grad_wrt_out,
    grad_wrt_in,
    [c] __device__(T const& x, T const& dy) {
      return (x < c ? dy : (x > c ? (T)0. : dy / (T)2.));
    });
}

template <typename T>
void ApplyEqualFP(T c,
                  El::Matrix<T, El::Device::GPU> const& x,
                  El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) {
    return (c == x ? (T)1. : (T)0.);
  });
}

template <typename T>
void ApplyNotEqualFP(T c,
                     El::Matrix<T, El::Device::GPU> const& x,
                     El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) {
    return (c == x ? (T)0. : (T)1.);
  });
}

template <typename T>
void ApplyLessFP(T c,
                 El::Matrix<T, El::Device::GPU> const& x,
                 El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) {
    return (x < c ? (T)1. : (T)0.);
  });
}

template <typename T>
void ApplyGreaterFP(T c,
                    El::Matrix<T, El::Device::GPU> const& x,
                    El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) {
    return (c < x ? (T)1. : (T)0.);
  });
}

template <typename T>
void ApplyLessEqualFP(T c,
                      El::Matrix<T, El::Device::GPU> const& x,
                      El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) {
    return (x <= c ? (T)1. : (T)0.);
  });
}

template <typename T>
void ApplyGreaterEqualFP(T c,
                         El::Matrix<T, El::Device::GPU> const& x,
                         El::Matrix<T, El::Device::GPU>& y)
{
  El::EntrywiseMap(x, y, [c] __device__(T const& x) {
    return (c <= x ? (T)1. : (T)0.);
  });
}

} // namespace

namespace lbann {

template <typename DataT, El::Device D>
void AddConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyAddFP(this->m_constant, inputs.front().data(), outputs.front().data());
}

template <typename DataT, El::Device D>
void AddConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_outputs.size() == 1);
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::Copy(gradient_wrt_outputs.front().data(),
           gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void ScaleOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyScaleFP(this->m_constant, inputs.front().data(), outputs.front().data());
}

template <typename DataT, El::Device D>
void ScaleOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  this->fp_compute_local(std::move(gradient_wrt_outputs),
                         std::move(gradient_wrt_inputs));
}

template <typename DataT, El::Device D>
void SubtractConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplySubtractFP(this->m_constant,
                  inputs.front().data(),
                  outputs.front().data());
}

template <typename DataT, El::Device D>
void SubtractConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_outputs.size() == 1);
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::Copy(gradient_wrt_outputs.front().data(),
           gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void ConstantSubtractOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyCSubtractFP(this->m_constant,
                   inputs.front().data(),
                   outputs.front().data());
}

template <typename DataT, El::Device D>
void ConstantSubtractOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_outputs.size() == 1);
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  ApplyCSubtractBP(gradient_wrt_outputs.front().data(),
                   gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void MaxConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyMaxFP(this->m_constant, inputs.front().data(), outputs.front().data());
}

template <typename DataT, El::Device D>
void MaxConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(gradient_wrt_outputs.size() == 1);
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  ApplyMaxBP(this->m_constant,
             inputs.front().data(),
             gradient_wrt_outputs.front().data(),
             gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void MinConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyMinFP(this->m_constant, inputs.front().data(), outputs.front().data());
}

template <typename DataT, El::Device D>
void MinConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(gradient_wrt_outputs.size() == 1);
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  ApplyMinBP(this->m_constant,
             inputs.front().data(),
             gradient_wrt_outputs.front().data(),
             gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void EqualConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyEqualFP(this->m_constant, inputs.front().data(), outputs.front().data());
}

template <typename DataT, El::Device D>
void EqualConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> /*gradient_wrt_outputs*/,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::Zero(gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void NotEqualConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyNotEqualFP(this->m_constant,
                  inputs.front().data(),
                  outputs.front().data());
}

template <typename DataT, El::Device D>
void NotEqualConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> /*gradient_wrt_outputs*/,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::Zero(gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void LessConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyLessFP(this->m_constant, inputs.front().data(), outputs.front().data());
}

template <typename DataT, El::Device D>
void LessConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> /*gradient_wrt_outputs*/,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::Zero(gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void LessEqualConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyLessEqualFP(this->m_constant,
                   inputs.front().data(),
                   outputs.front().data());
}

template <typename DataT, El::Device D>
void LessEqualConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> /*gradient_wrt_outputs*/,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::Zero(gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void GreaterConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyGreaterFP(this->m_constant,
                 inputs.front().data(),
                 outputs.front().data());
}

template <typename DataT, El::Device D>
void GreaterConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> /*gradient_wrt_outputs*/,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::Zero(gradient_wrt_inputs.front().data());
}

template <typename DataT, El::Device D>
void GreaterEqualConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  ApplyGreaterEqualFP(this->m_constant,
                      inputs.front().data(),
                      outputs.front().data());
}

template <typename DataT, El::Device D>
void GreaterEqualConstantOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> /*gradient_wrt_outputs*/,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::Zero(gradient_wrt_inputs.front().data());
}

#define PROTO(T)                                                               \
  template class AddConstantOperator<T, El::Device::GPU>;                      \
  template class ScaleOperator<T, El::Device::GPU>;                            \
  template class SubtractConstantOperator<T, El::Device::GPU>;                 \
  template class ConstantSubtractOperator<T, El::Device::GPU>;                 \
  template class MaxConstantOperator<T, El::Device::GPU>;                      \
  template class MinConstantOperator<T, El::Device::GPU>;                      \
  template class EqualConstantOperator<T, El::Device::GPU>;                    \
  template class NotEqualConstantOperator<T, El::Device::GPU>;                 \
  template class LessConstantOperator<T, El::Device::GPU>;                     \
  template class LessEqualConstantOperator<T, El::Device::GPU>;                \
  template class GreaterConstantOperator<T, El::Device::GPU>;                  \
  template class GreaterEqualConstantOperator<T, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
