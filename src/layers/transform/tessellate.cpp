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

#define LBANN_TESSELLATE_LAYER_INSTANTIATE
#include "lbann/layers/transform/tessellate.hpp"

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void tessellate_layer<TensorDataType, T_layout, Dev>::fp_compute_3d(
  const std::vector<int>& input_dims,
  const std::vector<int>& output_dims,
  const El::AbstractMatrix<TensorDataType>& input,
  El::AbstractDistMatrix<TensorDataType>& output)
{
  LBANN_CALIPER_MARK_SCOPE("tessellate_layer::fp_compute");

  // Input matrix
  const El::Int input_dim0 = input_dims[0];
  const El::Int input_dim1 = input_dims[1];
  const El::Int input_dim2 = input_dims[2];
  const TensorDataType* __restrict__ input_buffer = input.LockedBuffer();
  const El::Int input_ldim = input.LDim();

  // Output matrix
  const El::Int output_dim1 = output_dims[1];
  const El::Int output_dim2 = output_dims[2];
  const El::Int local_output_height = output.LocalHeight();
  const El::Int local_output_width = output.LocalWidth();
  TensorDataType* __restrict__ output_buffer = output.Buffer();
  const El::Int output_ldim = output.LDim();
  const El::Int output_col_shift = output.ColShift();
  const El::Int output_col_stride = output.ColStride();

  // Populate local entries in output matrix
  LBANN_OMP_PARALLEL_FOR
  for (El::Int local_col = 0; local_col < local_output_width; ++local_col) {
    for (El::Int local_row = 0; local_row < local_output_height; ++local_row) {

      // Get output entry
      const auto& output_index =
        (output_col_shift + local_row * output_col_stride);
      const auto& output_pos2 = output_index % output_dim2;
      const auto& output_pos1 = (output_index / output_dim2) % output_dim1;
      const auto& output_pos0 = output_index / (output_dim1 * output_dim2);
      auto& y = output_buffer[local_row + local_col * output_ldim];

      // Get corresponding input entry
      const auto& input_pos0 = output_pos0 % input_dim0;
      const auto& input_pos1 = output_pos1 % input_dim1;
      const auto& input_pos2 = output_pos2 % input_dim2;
      const auto& input_index = (input_pos0 * input_dim1 * input_dim2 +
                                 input_pos1 * input_dim2 + input_pos2);
      const auto& x = input_buffer[input_index + local_col * input_ldim];
      y = x;
    }
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void tessellate_layer<TensorDataType, T_layout, Dev>::bp_compute_3d(
  const std::vector<int>& input_dims,
  const std::vector<int>& output_dims,
  const El::AbstractDistMatrix<TensorDataType>& output_grad,
  El::AbstractMatrix<TensorDataType>& input_grad)
{
  LBANN_CALIPER_MARK_SCOPE("tessellate_layer::bp_compute");

  // Input grad matrix
  const El::Int input_dim0 = input_dims[0];
  const El::Int input_dim1 = input_dims[1];
  const El::Int input_dim2 = input_dims[2];
  TensorDataType* __restrict__ input_grad_buffer = input_grad.Buffer();
  const El::Int input_grad_ldim = input_grad.LDim();

  // Output grad matrix
  const El::Int output_dim1 = output_dims[1];
  const El::Int output_dim2 = output_dims[2];
  const El::Int local_output_grad_height = output_grad.LocalHeight();
  const El::Int local_output_grad_width = output_grad.LocalWidth();
  const TensorDataType* __restrict__ output_grad_buffer =
    output_grad.LockedBuffer();
  const El::Int output_grad_ldim = output_grad.LDim();
  const El::Int output_grad_col_shift = output_grad.ColShift();
  const El::Int output_grad_col_stride = output_grad.ColStride();

  // Compute local contribution to error signal
  El::Zero(input_grad);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int local_col = 0; local_col < local_output_grad_width;
       ++local_col) {
    for (El::Int local_row = 0; local_row < local_output_grad_height;
         ++local_row) {

      // Get gradient w.r.t. output entry
      const auto& output_index =
        (output_grad_col_shift + local_row * output_grad_col_stride);
      const auto& output_pos2 = output_index % output_dim2;
      const auto& output_pos1 = (output_index / output_dim2) % output_dim1;
      const auto& output_pos0 = output_index / (output_dim1 * output_dim2);
      const auto& dy =
        output_grad_buffer[local_row + local_col * output_grad_ldim];

      // Update corresponding gradient w.r.t. input entry
      const auto& input_pos0 = output_pos0 % input_dim0;
      const auto& input_pos1 = output_pos1 % input_dim1;
      const auto& input_pos2 = output_pos2 % input_dim2;
      const auto& input_index = (input_pos0 * input_dim1 * input_dim2 +
                                 input_pos1 * input_dim2 + input_pos2);
      auto& dx = input_grad_buffer[input_index + local_col * input_grad_ldim];
      dx += dy;
    }
  }
}

// Explicit template instantiation
#define PROTO(T)                                                               \
  template class tessellate_layer<T,                                           \
                                  data_layout::DATA_PARALLEL,                  \
                                  El::Device::CPU>;                            \
  template class tessellate_layer<T,                                           \
                                  data_layout::MODEL_PARALLEL,                 \
                                  El::Device::CPU>
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
