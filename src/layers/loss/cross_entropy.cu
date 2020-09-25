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

#define LBANN_CROSS_ENTROPY_LAYER_INSTANTIATE
#include "lbann/layers/loss/cross_entropy.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

template <int block_size, typename TensorDataType>
__global__ void fp_kernel(int height, int width,
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
  for (int col = bidy; col < width; col += gridDim.y) {

    // Compute contributions for each thread
    auto private_contribution = TensorDataType(0.);
    for (int row = gidx; row < height; row += nthreadsx) {
      const auto& xhat = ground_truth[row + col * ground_truth_ldim];
      if (xhat > TensorDataType(0.)){
        const auto& x = prediction[row + col * prediction_ldim];
        private_contribution += - xhat * gpu_lib::log(x);
      }
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
      gpu_lib::atomic_add(&contribution[col], shared_contribution[0]);
    }

  }

}

template <typename TensorDataType>
void local_fp_gpu(const El::AbstractMatrix<TensorDataType>& local_prediction,
                  const El::AbstractMatrix<TensorDataType>& local_ground_truth,
                  El::AbstractMatrix<TensorDataType>& local_contribution) {
  El::Zero(local_contribution);
  const auto& height = local_prediction.Height();
  const auto& width = local_prediction.Width();
  if (height > 0 && width > 0) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_contribution),
                                       gpu::get_sync_info(local_prediction),
                                       gpu::get_sync_info(local_ground_truth));
    const int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (height + block_size - 1) / block_size;
    grid_dims.y = width;
    hydrogen::gpu::LaunchKernel(
      fp_kernel<block_size, TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      height, width,
      local_prediction.LockedBuffer(), local_prediction.LDim(),
      local_ground_truth.LockedBuffer(), local_ground_truth.LDim(),
      local_contribution.Buffer());
  }
}

template <int block_size, typename TensorDataType>
__global__ void bp_kernel(int height, int width,
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
  for (int col = bidy; col < width; col += gridDim.y) {
    const auto& dy = gradient_wrt_output[col];
    for (int row = gidx; row < height; row += nthreadsx) {
      const auto& x = prediction[row + col * prediction_ldim];
      const auto& xhat = ground_truth[row + col * ground_truth_ldim];
      auto& dx = gradient_wrt_prediction[row + col * gradient_wrt_prediction_ldim];
      auto& dxhat = gradient_wrt_ground_truth[row + col * gradient_wrt_ground_truth_ldim];
      dx = (xhat > TensorDataType(0.)) ? - dy * xhat / x : TensorDataType(0.);
      dxhat = - dy * gpu_lib::log(x);
    }
  }

}

template <typename TensorDataType>
void local_bp_gpu(const El::AbstractMatrix<TensorDataType>& local_prediction,
                  const El::AbstractMatrix<TensorDataType>& local_ground_truth,
                  const El::AbstractMatrix<TensorDataType>& local_gradient_wrt_output,
                  El::AbstractMatrix<TensorDataType>& local_gradient_wrt_prediction,
                  El::AbstractMatrix<TensorDataType>& local_gradient_wrt_ground_truth) {
  const auto& height = local_prediction.Height();
  const auto& width = local_prediction.Width();
  if (height > 0 && width > 0) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_gradient_wrt_prediction),
                        gpu::get_sync_info(local_gradient_wrt_ground_truth),
                        gpu::get_sync_info(local_gradient_wrt_output),
                        gpu::get_sync_info(local_prediction),
                        gpu::get_sync_info(local_ground_truth));

    const int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (height + block_size - 1) / block_size;
    grid_dims.y = width;
    hydrogen::gpu::LaunchKernel(
      bp_kernel<block_size, TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      height, width,
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
void cross_entropy_layer<TensorDataType, T_layout, Dev>::local_fp_compute() {
  local_fp_gpu(this->get_local_prev_activations(0),
               this->get_local_prev_activations(1),
               this->m_workspace->Matrix());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void cross_entropy_layer<TensorDataType, T_layout, Dev>::local_bp_compute() {
  local_bp_gpu(this->get_local_prev_activations(0),
               this->get_local_prev_activations(1),
               this->m_workspace->LockedMatrix(),
               this->get_local_error_signals(0),
               this->get_local_error_signals(1));
}

#define PROTO(T)                                      \
  template class cross_entropy_layer<                 \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;  \
  template class cross_entropy_layer<                 \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
