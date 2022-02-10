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

#include "lbann/operators/math/clamp.hpp"
#include "lbann/utils/exception.hpp"

#include "common.hpp"

namespace lbann {

template <typename DataT, El::Device D>
void ClampOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  El::EntrywiseMap(inputs.front().data(),
                   outputs.front().data(),
                   std::function<DataT(DataT const&)>([this](DataT const& x) {
                     return std::max(m_min, std::min(m_max, x));
                   }));
}

template <typename DataT, El::Device D>
void ClampOperator<DataT, D>::bp_compute_local(
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
                             [this](auto const& x, auto const& dy) {
                               return (x <= m_min || x >= m_max)
                                        ? El::TypeTraits<DataT>::Zero()
                                        : dy;
                             });
}

#define PROTO(T) template class ClampOperator<T, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
