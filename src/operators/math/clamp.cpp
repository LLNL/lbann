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
template <typename TensorDataType>
void local_fp(TensorDataType min,
              TensorDataType max,
              const El::AbstractMatrix<TensorDataType>& input,
              El::AbstractMatrix<TensorDataType>& output) {
  const auto& height = input.Height();
  const auto& width = input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = input(row, col);
      auto& y = output(row, col);
      if (x <= min)      { y = min; }
      else if (x >= max) { y = max; }
      else              { y = x;   }
    }
  }
}

/** Local backprop computation. */
template <typename TensorDataType>
void local_bp(TensorDataType min,
              TensorDataType max,
              const El::AbstractMatrix<TensorDataType>& input,
              const El::AbstractMatrix<TensorDataType>& gradient_wrt_output,
              El::AbstractMatrix<TensorDataType>& gradient_wrt_input) {
  const auto& height = input.Height();
  const auto& width = input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = input(row, col);
      const auto& dy = gradient_wrt_output(row, col);
      auto& dx = gradient_wrt_input(row, col);
      dx = (x <= min || x >= max) ? El::TypeTraits<TensorDataType>::Zero() : dy;
    }
  }
}

} // namespace

template <typename TensorDataType>
void ClampOperator<TensorDataType>::fp_compute_local(std::vector<CPUMatrixType const*> inputs,
                                                     std::vector<CPUMatrixType*> outputs) const {
  if(inputs.size() != 1 || outputs.size() != 1) {
    LBANN_ERROR("Invalid argument list");
  }
  local_fp(this->m_min, this->m_max,
           *(inputs[0]),
           *(outputs[0]));
}

template <typename TensorDataType>
void ClampOperator<TensorDataType>::bp_compute_local(std::vector<CPUMatrixType const*> inputs,
                                                     std::vector<CPUMatrixType const*> gradient_wrt_outputs,
                                                     std::vector<CPUMatrixType*> gradient_wrt_inputs) const {
  if(inputs.size() != 1 || gradient_wrt_outputs.size() != 1 || gradient_wrt_inputs.size() != 1) {
    LBANN_ERROR("Invalid argument list");
  }
  local_bp(this->m_min, this->m_max,
           *(inputs[0]),
           *(gradient_wrt_outputs[0]),
           *(gradient_wrt_inputs[0]));
}

#define PROTO(T)                                     \
  template class ClampOperator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
