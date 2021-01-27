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

#define LBANN_SCATTER_LAYER_INSTANTIATE
#include "lbann/layers/transform/scatter.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void scatter_layer<TensorDataType, Layout, Device>::fp_compute() {

  // Local matrices
  const auto& local_values = this->get_local_prev_activations(0);
  const auto& local_indices = this->get_local_prev_activations(1);
  auto& local_output = this->get_local_activations();
  const size_t input_size = this->get_input_size(0);
  const El::Int output_size = this->get_output_size();
  const size_t local_mini_batch_size = local_values.Width();

  // Scatter into output matrix
  El::Zero(local_output);
  LBANN_OMP_PARALLEL_FOR
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const El::Int ind = static_cast<El::Int>(std::floor(local_indices(i,j)));
      if (0<=ind && ind<output_size) {
        local_output(ind,j) += local_values(i,j);
      }
    }
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void scatter_layer<TensorDataType, Layout, Device>::bp_compute() {

  // Local matrices
  const auto& local_indices = this->get_local_prev_activations(1);
  const auto& local_output_grad = this->get_local_prev_error_signals();
  auto& local_values_grad = this->get_local_error_signals(0);
  auto& local_indices_grad = this->get_local_error_signals(1);
  const size_t input_size = this->get_input_size(0);
  const El::Int output_size = this->get_output_size();
  const size_t local_mini_batch_size = local_indices.Width();

  // Zero out gradient w.r.t. indices
  El::Zero(local_indices_grad);

  // Gather into gradient w.r.t. values
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const El::Int ind = static_cast<El::Int>(std::floor(local_indices(i,j)));
      if (0<=ind && ind<output_size) {
        local_values_grad(i,j) = local_output_grad(ind,j);
      }
      else {
        local_values_grad(i,j) = El::TypeTraits<TensorDataType>::Zero();
      }
    }
  }

}

#define PROTO(T)                                        \
  template class scatter_layer<                         \
    T, data_layout::DATA_PARALLEL, El::Device::CPU>
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
