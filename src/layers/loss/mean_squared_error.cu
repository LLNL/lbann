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

#define LBANN_MEAN_SQUARED_ERROR_LAYER_INSTANTIATE
#include "lbann/layers/loss/mean_squared_error.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

template <int block_size, typename TensorDataType>
__global__ void fp_kernel(int global_height,
                          int local_height, int local_width,
                          const TensorDataType* __restrict__ prediction,
                          int prediction_ldim,
                          const TensorDataType* __restrict__ ground_truth,
                          int ground_truth_ldim,
                          TensorDataType* __restrict__ contribution) {

  // Indices
  const int tid = threadIdx.x;
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;
  const int nthreadsx = blockDim.x * gridDim.x;

  // Compute local contribution for each matrix column
  for (int col = bidy; col < local_width; col += gridDim.y) {

    // Compute contributions for each thread
    TensorDataType private_contribution = TensorDataType(0.0);
    for (int row = gidx; row < local_height; row += nthreadsx) {
      const auto& err = (prediction[row + col * prediction_ldim]
                         - ground_truth[row + col * ground_truth_ldim]);
      private_contribution += err * err;
    }

    // Shared memory reduction to get contribution for each block
    /// @todo unroll loops
    __shared__ TensorDataType shared_contribution[block_size];
    shared_contribution[tid] = private_contribution;
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_contribution[tid] += shared_contribution[tid + stride];
      }
    }
    if (tid == 0) {
      shared_contribution[0] /= global_height;
      gpu_lib::atomic_add(&contribution[col], shared_contribution[0]);
    }

  }

}

template <typename TensorDataType>
void local_fp_gpu(El::Int height,
                  const El::AbstractMatrix<TensorDataType>& local_prediction,
                  const El::AbstractMatrix<TensorDataType>& local_ground_truth,
                  El::AbstractMatrix<TensorDataType>& local_contribution) {
  El::Zero(local_contribution);
  const auto& local_height = local_prediction.Height();
  const auto& local_width = local_prediction.Width();
  if (local_height > 0 && local_width > 0) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_contribution),
                        gpu::get_sync_info(local_ground_truth),
                        gpu::get_sync_info(local_prediction));
    const int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    hydrogen::gpu::LaunchKernel(
      fp_kernel<block_size, TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      height, local_height, local_width,
      local_prediction.LockedBuffer(), local_prediction.LDim(),
      local_ground_truth.LockedBuffer(), local_ground_truth.LDim(),
      local_contribution.Buffer());
  }
}

template <int block_size, typename TensorDataType>
__global__ void bp_kernel(int global_height,
                          int local_height, int local_width,
                          const TensorDataType* __restrict__ prediction,
                          int prediction_ldim,
                          const TensorDataType* __restrict__ ground_truth,
                          int ground_truth_ldim,
                          const TensorDataType* __restrict__ gradient_wrt_output,
                          TensorDataType* __restrict__ gradient_wrt_prediction,
                          int gradient_wrt_prediction_ldim,
                          TensorDataType* __restrict__ gradient_wrt_ground_truth,
                          int gradient_wrt_ground_truth_ldim) {

  // Indices
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;
  const int nthreadsx = blockDim.x * gridDim.x;

  // Compute gradients
  for (int col = bidy; col < local_width; col += gridDim.y) {
    const auto& dy = gradient_wrt_output[col];
    for (int row = gidx; row < local_height; row += nthreadsx) {
      const auto& err = (prediction[row + col * prediction_ldim]
                         - ground_truth[row + col * ground_truth_ldim]);
      auto& dx = gradient_wrt_prediction[row + col * gradient_wrt_prediction_ldim];
      auto& dxhat = gradient_wrt_ground_truth[row + col * gradient_wrt_ground_truth_ldim];
      dx = TensorDataType(2) * err * dy / TensorDataType(global_height);
      dxhat = TensorDataType(-2) * err * dy / TensorDataType(global_height);
    }
  }

}

template <typename TensorDataType>
void local_bp_gpu(El::Int height,
                  const El::AbstractMatrix<TensorDataType>& local_prediction,
                  const El::AbstractMatrix<TensorDataType>& local_ground_truth,
                  const El::AbstractMatrix<TensorDataType>& local_gradient_wrt_output,
                  El::AbstractMatrix<TensorDataType>& local_gradient_wrt_prediction,
                  El::AbstractMatrix<TensorDataType>& local_gradient_wrt_ground_truth) {
  const auto& local_height = local_prediction.Height();
  const auto& local_width = local_prediction.Width();
  if (local_height > 0 && local_width > 0) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_gradient_wrt_prediction),
                        gpu::get_sync_info(local_gradient_wrt_ground_truth),
                        gpu::get_sync_info(local_gradient_wrt_output),
                        gpu::get_sync_info(local_ground_truth),
                        gpu::get_sync_info(local_prediction));
    const int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    hydrogen::gpu::LaunchKernel(
      bp_kernel<block_size, TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      height, local_height, local_width,
      local_prediction.LockedBuffer(), local_prediction.LDim(),
      local_ground_truth.LockedBuffer(), local_ground_truth.LDim(),
      local_gradient_wrt_output.LockedBuffer(),
      local_gradient_wrt_prediction.Buffer(),
      local_gradient_wrt_prediction.LDim(),
      local_gradient_wrt_ground_truth.Buffer(),
      local_gradient_wrt_ground_truth.LDim());
  }
}

} // namespace

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void mean_squared_error_layer<TensorDataType, T_layout, Dev>::local_fp_compute() {
  local_fp_gpu(this->get_input_size(),
               this->get_local_prev_activations(0),
               this->get_local_prev_activations(1),
               this->m_workspace->Matrix());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void mean_squared_error_layer<TensorDataType, T_layout, Dev>::local_bp_compute() {
  local_bp_gpu(this->get_input_size(),
               this->get_local_prev_activations(0),
               this->get_local_prev_activations(1),
               this->m_workspace->LockedMatrix(),
               this->get_local_error_signals(0),
               this->get_local_error_signals(1));
}

#define PROTO(T)                                      \
  template class mean_squared_error_layer<            \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;  \
  template class mean_squared_error_layer<            \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
