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

#define LBANN_CLAMP_OPERATOR_INSTANTIATE
#include "lbann/operators/math/clamp.hpp"

namespace lbann {

namespace {

/** Local forward prop computation. */
template <typename DataT>
void local_fp(DataT min,
              DataT max,
              El::Matrix<DataT, El::Device::CPU> const& input,
              El::Matrix<DataT, El::Device::CPU>& output)
{
  const auto& height = input.Height();
  const auto& width = input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = input(row, col);
      output(row, col) = std::max(min, std::min(max, x));
    }
  }
}

/** Local backprop computation. */
template <typename DataT>
void local_bp(DataT min,
              DataT max,
              El::Matrix<DataT, El::Device::CPU> const& input,
              El::Matrix<DataT, El::Device::CPU> const& gradient_wrt_output,
              El::Matrix<DataT, El::Device::CPU>& gradient_wrt_input)
{
  const auto& height = input.Height();
  const auto& width = input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = input(row, col);
      const auto& dy = gradient_wrt_output(row, col);
      auto& dx = gradient_wrt_input(row, col);
      dx = (x <= min || x >= max) ? El::TypeTraits<DataT>::Zero() : dy;
    }
  }
}

} // namespace

template <typename DataT, El::Device D>
void ClampOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
  local_fp(this->m_min,
           this->m_max,
           inputs.front().data(),
           outputs.front().data());
}

template <typename DataT, El::Device D>
void ClampOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT(inputs.size() == 1 && gradient_wrt_outputs.size() == 1 &&
               gradient_wrt_inputs.size() == 1);

  local_bp(this->m_min,
           this->m_max,
           inputs.front().data(),
           gradient_wrt_outputs.front().data(),
           gradient_wrt_inputs.front().data());
}

#define PROTO(T) template class ClampOperator<T, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
