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

#define LBANN_TESSELLATE_LAYER_INSTANTIATE
#include "lbann/layers/transform/tessellate.hpp"

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void tessellate_layer<TensorDataType, T_layout, Dev>
::fp_compute_3d(const std::vector<int>& input_dims,
                const std::vector<int>& output_dims,
                const El::AbstractMatrix<TensorDataType>& input,
                El::AbstractDistMatrix<TensorDataType>& output) {
  auto& local_output = output.Matrix();
  const auto& local_height = local_output.Height();
  const auto& local_width = local_output.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (El::Int local_col = 0; local_col < local_width; ++local_col) {
      for (El::Int local_row = 0; local_row < local_height; ++local_row) {

        // Get output entry
        const auto& output_index = output.GlobalRow(local_row);
        const auto& output_pos2 = output_index % output_dims[2];
        const auto& output_pos1 = (output_index / output_dims[2]) % output_dims[1];
        const auto& output_pos0 = output_index / (output_dims[1] * output_dims[2]);
        auto& y = local_output(local_row, local_col);

        // Get corresponding input entry
        const auto& input_pos0 = output_pos0 % input_dims[0];
        const auto& input_pos1 = output_pos1 % input_dims[1];
        const auto& input_pos2 = output_pos2 % input_dims[2];
        const auto& input_index = (input_pos0 * input_dims[1] * input_dims[2]
                                   + input_pos1 * input_dims[2]
                                   + input_pos2);
        const auto& x = input(input_index, local_col);
        y = x;

      }
    }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void tessellate_layer<TensorDataType, T_layout, Dev>
::bp_compute_3d(const std::vector<int>& input_dims,
                const std::vector<int>& output_dims,
                const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
                El::AbstractMatrix<TensorDataType>& gradient_wrt_input) {

  // Local data
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  const auto& local_height = local_gradient_wrt_output.Height();
  const auto& local_width = local_gradient_wrt_output.Width();

  // Compute local contribution to error signal
  El::Zero(gradient_wrt_input);
  LBANN_OMP_PARALLEL_FOR
    for (El::Int local_col = 0; local_col < local_width; ++local_col) {
      for (El::Int local_row = 0; local_row < local_height; ++local_row) {

        // Get gradient w.r.t. output entry
        const auto& output_index = gradient_wrt_output.GlobalRow(local_row);
        const auto& output_pos2 = output_index % output_dims[2];
        const auto& output_pos1 = (output_index / output_dims[2]) % output_dims[1];
        const auto& output_pos0 = output_index / (output_dims[1] * output_dims[2]);
        const auto& dy = local_gradient_wrt_output(local_row, local_col);

        // Update corresponding gradient w.r.t. input entry
        const auto& input_pos0 = output_pos0 % input_dims[0];
        const auto& input_pos1 = output_pos1 % input_dims[1];
        const auto& input_pos2 = output_pos2 % input_dims[2];
        const auto& input_index = (input_pos0 * input_dims[1] * input_dims[2]
                                   + input_pos1 * input_dims[2]
                                   + input_pos2);
        auto& dx = gradient_wrt_input(input_index, local_col);
        dx += dy;

      }
    }

}

#define PROTO(T)                                      \
  template class tessellate_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>; \
  template class tessellate_layer<T, data_layout::MODEL_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
