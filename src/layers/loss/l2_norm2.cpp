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

#define LBANN_L2_NORM2_LAYER_INSTANTIATE
#include "lbann/layers/loss/l2_norm2_impl.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
void local_fp_cpu(const El::AbstractMatrix<TensorDataType>& local_input,
                  El::AbstractMatrix<TensorDataType>& local_contribution)
{
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_input.Width(); ++col) {
    TensorDataType sum = El::TypeTraits<TensorDataType>::Zero();
    for (El::Int row = 0; row < local_input.Height(); ++row) {
      const auto& x = local_input(row, col);
      sum += x * x;
    }
    local_contribution(0, col) = sum;
  }
}

template <typename TensorDataType>
void local_bp_cpu(
  const El::AbstractMatrix<TensorDataType>& local_input,
  const El::AbstractMatrix<TensorDataType>& local_gradient_wrt_output,
  El::AbstractMatrix<TensorDataType>& local_gradient_wrt_input)
{
  auto const width = local_input.Width();
  auto const height = local_input.Height();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = local_input(row, col);
      const auto& dy = local_gradient_wrt_output(0, col);
      auto& dx = local_gradient_wrt_input(row, col);
      dx = 2 * x * dy;
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
<<<<<<< HEAD
void l2_norm2_layer<TensorDataType, T_layout, Dev>::local_fp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("l2_norm2_layer::fp_compute");
  local_fp_cpu(this->get_local_prev_activations(), this->m_workspace->Matrix());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void l2_norm2_layer<TensorDataType, T_layout, Dev>::local_bp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("l2_norm2_layer::bp_compute");
  local_bp_cpu(this->get_local_prev_activations(),
               this->m_workspace->LockedMatrix(),
               this->get_local_error_signals());
}

#define PROTO(T)                                                               \
  template class l2_norm2_layer<T,                                             \
                                data_layout::DATA_PARALLEL,                    \
                                El::Device::CPU>;                              \
  template class l2_norm2_layer<T, data_layout::MODEL_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
