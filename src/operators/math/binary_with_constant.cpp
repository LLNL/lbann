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

#include "common.hpp"
#include "lbann_config.hpp"
#include <hydrogen/meta/TypeTraits.hpp>

namespace lbann {

template <typename DataT, El::Device D>
void AddConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>(
                     [this](DataT const& x) { return x + this->m_constant; }));
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
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>(
                     [this](DataT const& x) { return x * this->m_constant; }));
}

template <typename DataT, El::Device D>
void ScaleOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_outputs.size() == 1);
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::EntrywiseMap(gradient_wrt_outputs.front().data(),
                   gradient_wrt_inputs.front().data(),
                   std::function<DataT(DataT const&)>(
                     [this](DataT const& x) { return x * this->m_constant; }));
}

template <typename DataT, El::Device D>
void SubtractConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>(
                     [this](DataT const& x) { return x - this->m_constant; }));
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
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>(
                     [this](DataT const& x) { return this->m_constant - x; }));
}

template <typename DataT, El::Device D>
void ConstantSubtractOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> /*inputs*/,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_outputs.size() == 1);
  LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);
  El::EntrywiseMap(
    gradient_wrt_outputs.front().data(),
    gradient_wrt_inputs.front().data(),
    std::function<DataT(DataT const&)>([](DataT const& x) { return -x; }));
}

template <typename DataT, El::Device D>
void MaxConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>([this](DataT const& x) {
                     return std::max(this->m_constant, x);
                   }));
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
  internal::EntrywiseZipInto(inputs.front().data(),
                             gradient_wrt_outputs.front().data(),
                             gradient_wrt_inputs.front().data(),
                             [this](DataT const& x, DataT const& dy) {
                               auto const& c = this->m_constant;
                               return (x < c ? El::TypeTraits<DataT>::Zero()
                                             : (x > c ? dy : dy / 2));
                             });
}

template <typename DataT, El::Device D>
void MinConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>([this](DataT const& x) {
                     return std::min(this->m_constant, x);
                   }));
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
  internal::EntrywiseZipInto(
    inputs.front().data(),
    gradient_wrt_outputs.front().data(),
    gradient_wrt_inputs.front().data(),
    [this](DataT const& x, DataT const& dy) {
      auto const& c = this->m_constant;
      return (x < c ? dy : (x > c ? El::TypeTraits<DataT>::Zero() : dy / 2));
    });
}

template <typename DataT, El::Device D>
void EqualConstantOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>([this](DataT const& x) {
                     return (this->m_constant == x
                               ? El::TypeTraits<DataT>::One()
                               : El::TypeTraits<DataT>::Zero());
                   }));
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
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>([this](DataT const& x) {
                     return (this->m_constant == x
                               ? El::TypeTraits<DataT>::Zero()
                               : El::TypeTraits<DataT>::One());
                   }));
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
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>([this](DataT const& x) {
                     return (x < this->m_constant
                               ? El::TypeTraits<DataT>::One()
                               : El::TypeTraits<DataT>::Zero());
                   }));
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
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>([this](DataT const& x) {
                     return (x <= this->m_constant
                               ? El::TypeTraits<DataT>::One()
                               : El::TypeTraits<DataT>::Zero());
                   }));
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
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>([this](DataT const& x) {
                     return (this->m_constant < x
                               ? El::TypeTraits<DataT>::One()
                               : El::TypeTraits<DataT>::Zero());
                   }));
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
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>([this](DataT const& x) {
                     return (this->m_constant <= x
                               ? El::TypeTraits<DataT>::One()
                               : El::TypeTraits<DataT>::Zero());
                   }));
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
  template class AddConstantOperator<T, El::Device::CPU>;                      \
  template class ScaleOperator<T, El::Device::CPU>;                            \
  template class SubtractConstantOperator<T, El::Device::CPU>;                 \
  template class ConstantSubtractOperator<T, El::Device::CPU>;                 \
  template class MaxConstantOperator<T, El::Device::CPU>;                      \
  template class MinConstantOperator<T, El::Device::CPU>;                      \
  template class EqualConstantOperator<T, El::Device::CPU>;                    \
  template class NotEqualConstantOperator<T, El::Device::CPU>;                 \
  template class LessConstantOperator<T, El::Device::CPU>;                     \
  template class LessEqualConstantOperator<T, El::Device::CPU>;                \
  template class GreaterConstantOperator<T, El::Device::CPU>;                  \
  template class GreaterEqualConstantOperator<T, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
