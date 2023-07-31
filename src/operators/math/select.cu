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

#include "lbann/operators/math/select.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "common.cuh"

namespace lbann {

namespace {
template <typename DataT>
struct SelectImpl
{
  DataT m_value, m_epsilon;
  SelectImpl(DataT const& value, DataT const& epsilon)
    : m_value(value), m_epsilon(epsilon)
  {}

  inline __device__ DataT operator()(DataT const& predicate,
                                     DataT const& iftrue,
                                     DataT const& iffalse) const noexcept
  {
    bool istrue = fabs(predicate - m_value) < m_epsilon;
    return istrue ? iftrue : iffalse;
  }
  inline __device__ void operator()(DataT const& predicate,
                                    DataT const& iftrue,
                                    DataT const& iffalse,
                                    DataT const& dy,
                                    DataT& dpredicate,
                                    DataT& diftrue,
                                    DataT& diffalse) const noexcept
  {
    bool istrue = fabs(predicate - m_value) < m_epsilon;
    diftrue = istrue ? dy : DataT(0.0);
    diffalse = istrue ? DataT(0.0) : dy;
    dpredicate = DataT(0.0);
  }
};

template <typename DataT>
struct SelectImplIfTrue
{
  DataT m_value, m_epsilon, m_iftrue;
  SelectImplIfTrue(DataT const& value,
                   DataT const& epsilon,
                   DataT const& iftrue)
    : m_value(value), m_epsilon(epsilon), m_iftrue(iftrue)
  {}

  inline __device__ DataT operator()(DataT const& predicate,
                                     DataT const& iffalse) const noexcept
  {
    bool istrue = fabs(predicate - m_value) < m_epsilon;
    return istrue ? m_iftrue : iffalse;
  }
  inline __device__ void operator()(DataT const& predicate,
                                    DataT const& iffalse,
                                    DataT const& dy,
                                    DataT& dpredicate,
                                    DataT& diffalse) const noexcept
  {
    bool istrue = fabs(predicate - m_value) < m_epsilon;
    diffalse = istrue ? DataT(0.0) : dy;
    dpredicate = DataT(0.0);
  }
};

template <typename DataT>
struct SelectImplIfFalse
{
  DataT m_value, m_epsilon, m_iffalse;
  SelectImplIfFalse(DataT const& value,
                    DataT const& epsilon,
                    DataT const& iffalse)
    : m_value(value), m_epsilon(epsilon), m_iffalse(iffalse)
  {}

  inline __device__ DataT operator()(DataT const& predicate,
                                     DataT const& iftrue) const noexcept
  {
    bool istrue = fabs(predicate - m_value) < m_epsilon;
    return istrue ? iftrue : m_iffalse;
  }
  inline __device__ void operator()(DataT const& predicate,
                                    DataT const& iftrue,
                                    DataT const& dy,
                                    DataT& dpredicate,
                                    DataT& diftrue) const noexcept
  {
    bool istrue = fabs(predicate - m_value) < m_epsilon;
    diftrue = istrue ? dy : DataT(0.0);
    dpredicate = DataT(0.0);
  }
};

template <typename DataT>
struct SelectImplConstant
{
  DataT m_value, m_epsilon, m_iftrue, m_iffalse;
  SelectImplConstant(DataT const& value,
                     DataT const& epsilon,
                     DataT const& iftrue,
                     DataT const& iffalse)
    : m_value(value), m_epsilon(epsilon), m_iftrue(iftrue), m_iffalse(iffalse)
  {}

  inline __device__ DataT operator()(DataT const& predicate) const noexcept
  {
    bool istrue = fabs(predicate - m_value) < m_epsilon;
    return istrue ? m_iftrue : m_iffalse;
  }
  inline __device__ DataT operator()(DataT const& predicate,
                                     DataT const& dy) const noexcept
  {
    return DataT(0.0);
  }
};

} // namespace

template <typename DataT, El::Device D>
void SelectOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(outputs.size() == 1);

  // Select implementation version
  if (!m_constant_if_false && !m_constant_if_true) {
    LBANN_ASSERT_DEBUG(inputs.size() == 3);
    internal::EntrywiseZipInto(inputs[0].data(),
                               inputs[1].data(),
                               inputs[2].data(),
                               outputs.front().data(),
                               SelectImpl(m_value, m_epsilon));
  }
  else if (m_constant_if_false && !m_constant_if_true) {
    LBANN_ASSERT_DEBUG(inputs.size() == 2);
    internal::EntrywiseZipInto(
      inputs[0].data(),
      inputs[1].data(),
      outputs.front().data(),
      SelectImplIfFalse(m_value, m_epsilon, m_value_if_false));
  }
  else if (!m_constant_if_false && m_constant_if_true) {
    LBANN_ASSERT_DEBUG(inputs.size() == 2);
    internal::EntrywiseZipInto(
      inputs[0].data(),
      inputs[1].data(),
      outputs.front().data(),
      SelectImplIfTrue(m_value, m_epsilon, m_value_if_true));
  }
  else { // Both if-false and if-true are constants
    LBANN_ASSERT_DEBUG(inputs.size() == 1);
    El::EntrywiseMap(inputs.front().data(),
                     outputs.front().data(),
                     SelectImplConstant(m_value,
                                        m_epsilon,
                                        m_value_if_true,
                                        m_value_if_false));
  }
}

template <typename DataT, El::Device D>
void SelectOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(gradient_wrt_outputs.size() == 1);

  // Select implementation version
  if (!m_constant_if_false && !m_constant_if_true) {
    LBANN_ASSERT_DEBUG(inputs.size() == 3);
    LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 3);

    internal::apply_ternary_backprop_operator(
      inputs[0].data(),
      inputs[1].data(),
      inputs[2].data(),
      gradient_wrt_outputs.front().data(),
      gradient_wrt_inputs[0].data(),
      gradient_wrt_inputs[1].data(),
      gradient_wrt_inputs[2].data(),
      SelectImpl(m_value, m_epsilon));
  }
  else if (m_constant_if_false && !m_constant_if_true) {
    LBANN_ASSERT_DEBUG(inputs.size() == 2);
    LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 2);

    internal::apply_binary_backprop_operator(
      inputs[0].data(),
      inputs[1].data(),
      gradient_wrt_outputs.front().data(),
      gradient_wrt_inputs[0].data(),
      gradient_wrt_inputs[1].data(),
      SelectImplIfFalse(m_value, m_epsilon, m_value_if_false));
  }
  else if (!m_constant_if_false && m_constant_if_true) {
    LBANN_ASSERT_DEBUG(inputs.size() == 2);
    LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 2);

    internal::apply_binary_backprop_operator(
      inputs[0].data(),
      inputs[1].data(),
      gradient_wrt_outputs.front().data(),
      gradient_wrt_inputs[0].data(),
      gradient_wrt_inputs[1].data(),
      SelectImplIfTrue(m_value, m_epsilon, m_value_if_true));
  }
  else { // Both constant
    LBANN_ASSERT_DEBUG(inputs.size() == 1);
    LBANN_ASSERT_DEBUG(gradient_wrt_inputs.size() == 1);

    internal::EntrywiseZipInto(inputs.front().data(),
                               gradient_wrt_outputs.front().data(),
                               gradient_wrt_inputs.front().data(),
                               SelectImplConstant(m_value,
                                                  m_epsilon,
                                                  m_value_if_true,
                                                  m_value_if_false));
  }
}

#define PROTO(T) template class SelectOperator<T, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
