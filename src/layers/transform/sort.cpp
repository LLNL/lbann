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

#include "lbann/layers/transform/sort.hpp"

namespace lbann {

template <>
void sort_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::fp_compute() {

  // Local matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  auto& local_indices = *m_indices;
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // Sort each matrix column
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    std::multimap<DataType, El::Int> sorted_list;
    for (El::Int row = 0; row < local_height; ++row) {
      sorted_list.emplace(local_input(row, col), row);
    }
    if (m_descending) {
      auto&& it = sorted_list.rbegin();
      for (El::Int row = 0; row < local_height; ++row, ++it) {
        local_output(row, col) = it->first;
        local_indices(row, col) = it->second;
      }
    } else {
      auto&& it = sorted_list.begin();
      for (El::Int row = 0; row < local_height; ++row, ++it) {
        local_output(row, col) = it->first;
        local_indices(row, col) = it->second;
      }
    }
  }

}

template <>
void sort_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::bp_compute() {

  // Local matrices
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  const auto& local_indices = *m_indices;
  const auto& local_height = local_gradient_wrt_input.Height();
  const auto& local_width = local_gradient_wrt_input.Width();

  // Scatter gradients based on sorted indices
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& dy = local_gradient_wrt_output(row, col);
      auto& dx = local_gradient_wrt_input(local_indices(row, col), col);
      dx = dy;
    }
  }

}

} // namespace lbann
