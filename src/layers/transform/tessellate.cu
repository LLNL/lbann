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
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
__global__ void fp_gpu_3d_kernel(
  El::Int input_dim0, El::Int input_dim1, El::Int input_dim2,
  El::Int output_dim0, El::Int output_dim1, El::Int output_dim2,
  El::Int local_output_height, El::Int local_output_width,
  const TensorDataType * __restrict__ input, El::Int input_ldim,
  TensorDataType * __restrict__ local_output, El::Int local_output_ldim,
  El::Int output_col_shift, El::Int output_col_stride) {

  // Indices
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;

  // Iterate through output entries
  const auto& local_output_size = local_output_height * local_output_width;
  for (El::Int pos = gidx; pos < local_output_size; pos += num_threads) {
    const auto& local_row = pos % local_output_height;
    const auto& local_col = pos / local_output_height;

    // Get output entry
    const auto& output_index = (output_col_shift
                                + local_row * output_col_stride);
    const auto& output_pos2 = output_index % output_dim2;
    const auto& output_pos1 = (output_index / output_dim2) % output_dim1;
    const auto& output_pos0 = output_index / (output_dim1 * output_dim2);
    auto& y = local_output[local_row + local_col * local_output_ldim];

    // Get corresponding input entry
    const auto& input_pos0 = output_pos0 % input_dim0;
    const auto& input_pos1 = output_pos1 % input_dim1;
    const auto& input_pos2 = output_pos2 % input_dim2;
    const auto& input_index = (input_pos0 * input_dim1 * input_dim2
                               + input_pos1 * input_dim2
                               + input_pos2);
    const auto& x = input[input_index + local_col * input_ldim];
    y = x;

  }

}

template <typename TensorDataType>
__global__ void bp_gpu_3d_kernel(
  El::Int input_dim0, El::Int input_dim1, El::Int input_dim2,
  El::Int output_dim0, El::Int output_dim1, El::Int output_dim2,
  El::Int local_output_height, El::Int local_output_width,
  const TensorDataType * __restrict__ local_gradient_wrt_output,
  El::Int local_gradient_wrt_output_ldim,
  El::Int gradient_wrt_output_col_shift,
  El::Int gradient_wrt_output_col_stride,
  TensorDataType * __restrict__ gradient_wrt_input,
  El::Int gradient_wrt_input_ldim) {

  // Indices
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;

  // Iterate through output entries
  const auto& local_output_size = local_output_height * local_output_width;
  for (El::Int pos = gidx; pos < local_output_size; pos += num_threads) {
    const auto& local_row = pos % local_output_height;
    const auto& local_col = pos / local_output_height;

    // Get gradient w.r.t. output entry
    const auto& output_index = (gradient_wrt_output_col_shift
                                + local_row * gradient_wrt_output_col_stride);
    const auto& output_pos2 = output_index % output_dim2;
    const auto& output_pos1 = (output_index / output_dim2) % output_dim1;
    const auto& output_pos0 = output_index / (output_dim1 * output_dim2);
    const auto& dy = local_gradient_wrt_output[local_row + local_col * local_gradient_wrt_output_ldim];

    // Update corresponding gradient w.r.t. input entry
    const auto& input_pos0 = output_pos0 % input_dim0;
    const auto& input_pos1 = output_pos1 % input_dim1;
    const auto& input_pos2 = output_pos2 % input_dim2;
    const auto& input_index = (input_pos0 * input_dim1 * input_dim2
                               + input_pos1 * input_dim2
                               + input_pos2);
    auto& dx = gradient_wrt_input[input_index + local_col * gradient_wrt_input_ldim];
    cuda::atomic_add(&dx, dy);

  }

}

}// namespace <anon>

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void tessellate_layer<TensorDataType, T_layout, Dev>
::fp_compute_3d(const std::vector<int>& input_dims,
                const std::vector<int>& output_dims,
                const El::AbstractMatrix<TensorDataType>& input,
                El::AbstractDistMatrix<TensorDataType>& output) {
  auto& local_output = output.Matrix();
  if (!local_output.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(output),
                                       gpu::get_sync_info(input));

    const auto& local_height = local_output.Height();
    const auto& local_width = local_output.Width();
    const auto& block_size = 256;
    const auto& grid_size =
      (local_height * local_width + block_size - 1) / block_size;
    hydrogen::gpu::LaunchKernel(
      fp_gpu_3d_kernel<TensorDataType>,
      grid_size, block_size, 0, multisync,
      input_dims[0], input_dims[1], input_dims[2],
      output_dims[0], output_dims[1], output_dims[2],
      local_height, local_width,
      input.LockedBuffer(), input.LDim(),
      local_output.Buffer(), local_output.LDim(),
      output.ColShift(), output.ColStride());
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void tessellate_layer<TensorDataType, T_layout, Dev>
::bp_compute_3d(const std::vector<int>& input_dims,
                const std::vector<int>& output_dims,
                const El::AbstractDistMatrix<TensorDataType>& gradient_wrt_output,
                El::AbstractMatrix<TensorDataType>& gradient_wrt_input) {
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  El::Zero(gradient_wrt_input);
  if (!local_gradient_wrt_output.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(gradient_wrt_input),
                                       gpu::get_sync_info(gradient_wrt_output));

    const auto& local_height = local_gradient_wrt_output.Height();
    const auto& local_width = local_gradient_wrt_output.Width();
    const auto& block_size = 256;
    const auto& grid_size =
      (local_height * local_width + block_size - 1) / block_size;
    hydrogen::gpu::LaunchKernel(
      bp_gpu_3d_kernel<TensorDataType>,
      grid_size, block_size, 0, multisync,
      input_dims[0], input_dims[1], input_dims[2],
      output_dims[0], output_dims[1], output_dims[2],
      local_height, local_width,
      local_gradient_wrt_output.LockedBuffer(),
      local_gradient_wrt_output.LDim(),
      gradient_wrt_output.ColShift(),
      gradient_wrt_output.ColStride(),
      gradient_wrt_input.Buffer(),
      gradient_wrt_input.LDim());
  }
}

#define PROTO(T)                                      \
  template class tessellate_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class tessellate_layer<T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
