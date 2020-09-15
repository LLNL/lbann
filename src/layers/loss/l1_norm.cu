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

#define LBANN_L1_NORM_LAYER_INSTANTIATE
#include "lbann/layers/loss/l1_norm.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

template <El::Int block_size, typename TensorDataType>
__global__ void fp_kernel(El::Int local_height,
                          El::Int local_width,
                          const TensorDataType* __restrict__ input,
                          El::Int input_ldim,
                          TensorDataType* __restrict__ contribution) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;

  // Compute local contribution for each matrix column
  for (El::Int col = bidy; col < local_width; col += gridDim.y) {

    // Compute contributions for each thread
    TensorDataType private_contribution = 0;
    for (El::Int row = gidx; row < local_height; row += nthreadsx) {
      const auto& x = input[row + col * input_ldim];
      private_contribution += cuda::abs(x);
    }

    // Shared memory reduction to get contribution for each block
    /// @todo unroll loops
    __shared__ TensorDataType shared_contribution[block_size];
    shared_contribution[tid] = private_contribution;
    for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        shared_contribution[tid] += shared_contribution[tid + stride];
      }
    }
    if (tid == 0) {
      cuda::atomic_add(&contribution[col], shared_contribution[0]);
    }

  }

}

template <typename TensorDataType>
void local_fp_gpu(const El::AbstractMatrix<TensorDataType>& local_input,
                  El::AbstractMatrix<TensorDataType>& local_contribution) {
  El::Zero(local_contribution);
  if (!local_input.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_contribution),
                                       gpu::get_sync_info(local_input));
    const auto& local_height = local_input.Height();
    const auto& local_width = local_input.Width();
    const El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    hydrogen::gpu::LaunchKernel(
      fp_kernel<block_size, TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_height, local_width,
      local_input.LockedBuffer(), local_input.LDim(),
      local_contribution.Buffer());
  }
}

template <El::Int block_size, typename TensorDataType>
__global__ void bp_kernel(El::Int local_height, El::Int local_width,
                          const TensorDataType* __restrict__ input,
                          El::Int input_ldim,
                          const TensorDataType* __restrict__ gradient_wrt_output,
                          TensorDataType* __restrict__ gradient_wrt_input,
                          El::Int gradient_wrt_input_ldim) {
  const TensorDataType zero = 0.;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  for (El::Int col = bidy; col < local_width; col += gridDim.y) {
    const auto& dy = gradient_wrt_output[col];
    for (El::Int row = gidx; row < local_height; row += nthreadsx) {
      const auto& x = input[row + col * input_ldim];
      auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
      if (x > zero) {
        dx = dy;
      } else if (x < zero) {
        dx = -dy;
      } else {
        dx = zero;
      }
    }
  }
}

template <typename TensorDataType>
void local_bp_gpu(const El::AbstractMatrix<TensorDataType>& local_input,
                  const El::AbstractMatrix<TensorDataType>& local_gradient_wrt_output,
                  El::AbstractMatrix<TensorDataType>& local_gradient_wrt_input) {
  if (!local_input.IsEmpty()) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_gradient_wrt_input),
                        gpu::get_sync_info(local_gradient_wrt_output),
                        gpu::get_sync_info(local_input));
    const auto& local_height = local_input.Height();
    const auto& local_width = local_input.Width();
    const El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    grid_dims.y = local_width;
    hydrogen::gpu::LaunchKernel(
      bp_kernel<block_size, TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_height, local_width,
      local_input.LockedBuffer(), local_input.LDim(),
      local_gradient_wrt_output.LockedBuffer(),
      local_gradient_wrt_input.Buffer(),
      local_gradient_wrt_input.LDim());
  }
}

} // namespace

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void l1_norm_layer<TensorDataType, T_layout, Dev>::local_fp_compute() {
  local_fp_gpu(this->get_local_prev_activations(),
               this->m_workspace->Matrix());
}
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void l1_norm_layer<TensorDataType, T_layout, Dev>::local_bp_compute() {
  local_bp_gpu(this->get_local_prev_activations(),
               this->m_workspace->LockedMatrix(),
               this->get_local_error_signals());
}

#define PROTO(T)                                      \
  template class l1_norm_layer<                       \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;  \
  template class l1_norm_layer<                       \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
