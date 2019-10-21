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

#define LBANN_CLAMP_LAYER_INSTANTIATE
#include "lbann/layers/math/clamp.hpp"

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
      dx = (x <= min || x >= max) ? TensorDataType(0) : dy;
    }
  }
}

} // namespace

template <typename TensorDataType>
void fp_compute_impl(clamp_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l) {
  local_fp<TensorDataType>(l.m_min, l.m_max,
                           l.get_local_prev_activations(),
                           l.get_local_activations());
}
template <typename TensorDataType>
void bp_compute_impl(clamp_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l) {
  local_bp<TensorDataType>(l.m_min, l.m_max,
                           l.get_local_prev_activations(),
                           l.get_local_prev_error_signals(),
                           l.get_local_error_signals());
}
template <typename TensorDataType>
void fp_compute_impl(clamp_layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::CPU>& l) {
  local_fp<TensorDataType>(l.m_min, l.m_max,
                           l.get_local_prev_activations(),
                           l.get_local_activations());
}
template <typename TensorDataType>
void bp_compute_impl(clamp_layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::CPU>& l) {
  local_bp<TensorDataType>(l.m_min, l.m_max,
                           l.get_local_prev_activations(),
                           l.get_local_prev_error_signals(),
                           l.get_local_error_signals());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void clamp_layer<TensorDataType, Layout, Device>::fp_compute() {
  fp_compute_impl<TensorDataType>(*this);
}
template <typename TensorDataType, data_layout Layout, El::Device Device>
void clamp_layer<TensorDataType, Layout, Device>::bp_compute() {
  bp_compute_impl<TensorDataType>(*this);
}

template class clamp_layer<
  float, data_layout::DATA_PARALLEL, El::Device::CPU>;
template class clamp_layer<
  float, data_layout::MODEL_PARALLEL, El::Device::CPU>;

} // namespace lbann
