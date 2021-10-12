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

#define LBANN_GATHER_LAYER_INSTANTIATE
#include "lbann/layers/transform/gather.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gather_layer<TensorDataType, Layout, Device>::fp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("gather_layer::fp_compute");

  // Local matrices
  const auto& local_values = this->get_local_prev_activations(0);
  const auto& local_indices = this->get_local_prev_activations(1);
  auto& local_output = this->get_local_activations();
  const size_t local_mini_batch_size = local_values.Width();

  // Get the dimensions  to check if values is 2D
  const auto& input_dims_ = this->get_input_dims();
  const auto& output_dims_ = this->get_output_dims();
  std::vector<size_t> input_dims(input_dims_.begin(), input_dims_.end());
  std::vector<size_t> output_dims(output_dims_.begin(), output_dims_.end());

  const bool is_2D = input_dims.size() > 1;
  const bool is_axis_0 = (is_2D && this->m_gather_axis == 0);
  // Set value size, output size and set number of rows appropriately if 2D
  const El::Int num_values_cols =
    is_2D ? input_dims[1] : this->get_input_size(0);
  const El::Int num_values_rows = is_2D ? input_dims[0] : 1;

  const size_t num_output_cols =
    is_2D ? output_dims[1] : this->get_output_size();
  const size_t num_output_rows = is_2D ? output_dims[0] : 1;

  const El::Int bounds = is_axis_0 ? num_values_rows : num_values_cols;

  // Gather into output tensor
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (size_t j = 0; j < local_mini_batch_size; ++j) {
    for (size_t k = 0; k < num_output_rows; ++k) {
      for (size_t i = 0; i < num_output_cols; ++i) {
        const auto& index_val = is_axis_0 ? k : i;
        const auto ind =
          static_cast<El::Int>(std::floor(local_indices(index_val, j)));

        if (0 <= ind && ind < bounds) {
          if (is_axis_0) {
            local_output(k * num_output_cols + i, j) =
              local_values(ind * num_values_cols + i, j);
          }
          else {
            local_output(k * num_output_cols + i, j) =
              local_values(k * num_values_cols + ind, j);
          }
        }
        else {
          local_output(k * num_output_cols + i, j) =
            El::TypeTraits<TensorDataType>::Zero();
        }
      }
    }
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gather_layer<TensorDataType, Layout, Device>::bp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("gather_layer::bp_compute");

  // Local matrices
  const auto& local_indices = this->get_local_prev_activations(1);
  const auto& local_output_grad = this->get_local_prev_error_signals();
  auto& local_values_grad = this->get_local_error_signals(0);
  auto& local_indices_grad = this->get_local_error_signals(1);

  // Get the dimensions  to check if values is 2D
  const auto& input_dims_ = this->get_input_dims();
  const auto& output_dims_ = this->get_output_dims();
  std::vector<size_t> input_dims(input_dims_.begin(), input_dims_.end());
  std::vector<size_t> output_dims(output_dims_.begin(), output_dims_.end());
  const size_t local_mini_batch_size = local_indices.Width();

  const bool is_2D = input_dims.size() > 1;
  const bool is_axis_0 = (is_2D && this->m_gather_axis == 0);

  // Set value size, output size and set number of rows appropriately if 2D
  const El::Int num_values_cols =
    is_2D ? input_dims[1] : this->get_input_size(0);
  const El::Int num_values_rows = is_2D ? output_dims[0] : 1;

  const size_t num_output_cols =
    is_2D ? output_dims[1] : this->get_output_size();
  const size_t num_output_rows = is_2D ? output_dims[0] : 1;

  const El::Int bounds = is_axis_0 ? num_values_rows : num_values_cols;

  // Zero out gradient w.r.t. indices
  El::Zero(local_indices_grad);

  // Scatter into gradient w.r.t. values
  El::Zero(local_values_grad);
  LBANN_OMP_PARALLEL_FOR
  for (size_t j = 0; j < local_mini_batch_size; ++j) {
    for (size_t k = 0; k < num_output_rows; ++k) {
      for (size_t i = 0; i < num_output_cols; ++i) {
        const auto& index_val = is_axis_0 ? k : i;
        const auto ind =
          static_cast<El::Int>(std::floor(local_indices(index_val, j)));
        if (0 <= ind && ind < bounds) {
          if (is_axis_0) {
            local_values_grad(ind * num_values_cols + i, j) +=
              local_output_grad(k * num_output_cols + i, j);
          }
          else {
            local_values_grad(k * num_values_cols + ind, j) +=
              local_output_grad(k * num_output_cols + i, j);
          }
        }
      }
    }
  }
}

#define PROTO(T)                                                               \
  template class gather_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
