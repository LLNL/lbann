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

#define LBANN_GATHER_LAYER_INSTANTIATE
#include "lbann/layers/transform/gather.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gather_layer<TensorDataType, Layout, Device>::fp_compute() {

  // Local matrices
  const auto& local_values = this->get_local_prev_activations(0);
  const auto& local_indices = this->get_local_prev_activations(1);
  auto& local_output = this->get_local_activations();
  const El::Int values_size = this->get_input_size(0);
  const size_t output_size = this->get_output_size();
  const size_t local_mini_batch_size = local_values.Width();

  // Gather into output tensor
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<output_size; ++i) {
      const auto ind = static_cast<El::Int>(std::floor(local_indices(i,j)));
      if (0<=ind && ind<values_size) {
        local_output(i,j) = local_values(ind,j);
      }
      else {
        local_output(i,j) = El::TypeTraits<TensorDataType>::Zero();
      }
    }
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gather_layer<TensorDataType, Layout, Device>::bp_compute() {

  // Local matrices
  const auto& local_indices = this->get_local_prev_activations(1);
  const auto& local_output_grad = this->get_local_prev_error_signals();
  auto& local_values_grad = this->get_local_error_signals(0);
  auto& local_indices_grad = this->get_local_error_signals(1);
  const El::Int values_size = this->get_input_size(0);
  const size_t output_size = this->get_output_size();
  const size_t local_mini_batch_size = local_indices.Width();

  // Zero out gradient w.r.t. indices
  El::Zero(local_indices_grad);

  // Scatter into gradient w.r.t. values
  El::Zero(local_values_grad);
  LBANN_OMP_PARALLEL_FOR
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<output_size; ++i) {
      const auto ind = static_cast<El::Int>(std::floor(local_indices(i,j)));
      if (0<=ind && ind<values_size) {
        local_values_grad(ind,j) += local_output_grad(i,j);
      }
    }
  }

}

#define PROTO(T)                                        \
  template class gather_layer<                          \
    T, data_layout::DATA_PARALLEL, El::Device::CPU>
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
