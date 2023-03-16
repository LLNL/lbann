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

#define LBANN_ELU_LAYER_INSTANTIATE
#include "lbann/layers/activations/elu.hpp"

namespace lbann {

namespace {

/** Local forward prop computation. */
template <typename TensorDataType>
void local_fp(TensorDataType alpha,
              const El::AbstractMatrix<TensorDataType>& input,
              El::AbstractMatrix<TensorDataType>& output)
{
  const auto& height = input.Height();
  const auto& width = input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = input(row, col);
      auto& y = output(row, col);
      y = (x > El::TypeTraits<TensorDataType>::Zero()) ? x
                                                       : alpha * std::expm1(x);
    }
  }
}

/** Local backprop computation. */
template <typename TensorDataType>
void local_bp(TensorDataType alpha,
              const El::AbstractMatrix<TensorDataType>& input,
              const El::AbstractMatrix<TensorDataType>& gradient_wrt_output,
              El::AbstractMatrix<TensorDataType>& gradient_wrt_input)
{
  const auto& height = input.Height();
  const auto& width = input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = input(row, col);
      const auto& dy = gradient_wrt_output(row, col);
      auto& dx = gradient_wrt_input(row, col);
      dx = (x > El::TypeTraits<TensorDataType>::Zero())
             ? dy
             : dy * alpha * std::exp(x);
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void elu_layer<TensorDataType, Layout, Device>::fp_compute()
{
  local_fp(this->m_alpha,
           this->get_local_prev_activations(),
           this->get_local_activations());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void elu_layer<TensorDataType, Layout, Device>::bp_compute()
{
  local_bp(this->m_alpha,
           this->get_local_prev_activations(),
           this->get_local_prev_error_signals(),
           this->get_local_error_signals());
}

#define PROTO(T)                                                               \
  template class elu_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>;    \
  template class elu_layer<T, data_layout::MODEL_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
