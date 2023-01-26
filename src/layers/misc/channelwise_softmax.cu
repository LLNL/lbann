////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#define LBANN_CHANNELWISE_SOFTMAX_LAYER_INSTANTIATE
#include "lbann/layers/misc/channelwise_softmax.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include "channelwise_softmax_kernels.cuh"


namespace lbann {


template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_softmax_layer<TensorDataType,Layout,Device>::fp_compute() {
  
  #ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()){
    this->get_distconv_adapter().fp_compute();
    return ;
  }
  #endif // LBANN_HAS_DISTCONV

  // Local matrices
  const size_t num_channels = this->get_output_dims().front();
  const size_t channel_size = this->get_output_size() / num_channels;
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input = dynamic_cast<const LocalMat&>(this->get_prev_activations().LockedMatrix());
  auto& local_output = dynamic_cast<LocalMat&>(this->get_activations().Matrix());

  channelwise_softmax_fp_impl(num_channels,
                              channel_size,
                              local_input,
                              local_output);
}

// =========================================================
// Backprop
// =========================================================

namespace {

/** Compute dot product between output and gradient w.r.t. output.
 *
 *  Block dimensions: bdimx x 1 x 1
 *
 *  Grid dimensions: (output_dims[2] / bdimx) x output_dims[1] x output_dims[0]
 *
 *  y_dot_dy is a fully-packed 2D tensor with dimensions of
 *  output_dims[0] x output_dims[1].
 */
template <typename TensorDataType, size_t bdimx>
__global__ void bp_y_dot_dy_kernel(
  Size3 output_dims,
  const TensorDataType* __restrict__ output_buffer,
  Size3 output_strides,
  const TensorDataType* __restrict__ output_grad_buffer,
  Size3 output_grad_strides,
  TensorDataType* __restrict__ y_dot_dy) {

  // Indices and dimensions
  constexpr size_t bdimy = 1;
  constexpr size_t bdimz = 1;
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;

  for (size_t k = gidz; k < output_dims[0]; k += nthreadsz) {
    for (size_t j = gidy; j < output_dims[1]; j += nthreadsy) {

      // Compute contribution from each thread
      TensorDataType _y_dot_dy{0.};
      for (size_t i = gidx; i < output_dims[2]; i += nthreadsx) {
        const auto& y = output_buffer[k * output_strides[0]
                                      + j * output_strides[1]
                                      + i * output_strides[2]];
        const auto& dy = output_grad_buffer[k * output_grad_strides[0]
                                            + j * output_grad_strides[1]
                                            + i * output_grad_strides[2]];
        _y_dot_dy += y * dy;
      }

      // Compute contribution from each block
      _y_dot_dy = gpu_lib::block_reduce<bdimx,bdimy,bdimz>(_y_dot_dy);
      if (tid == 0) {
        gpu_lib::atomic_add(&y_dot_dy[j+k*output_dims[1]], _y_dot_dy);
      }

    }
  }

}

/** Compute gradient w.r.t. input.
 *
 *  dL/dx_i = y_i * ( dL/dy_i - dot(y,dL/dy) )
 *
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (output_dims[2] / bdimx) x (output_dims[1] / bdimy) x (output_dims[0] / bdimz)
 *
 *  y_dot_dy is a fully-packed 2D tensor with dimensions of
 *  output_dims[0] x output_dims[1].
 */
template <typename TensorDataType>
__global__ void bp_input_grad_kernel(
  Size3 output_dims,
  const TensorDataType* __restrict__ output_buffer,
  Size3 output_strides,
  const TensorDataType* __restrict__ output_grad_buffer,
  Size3 output_grad_strides,
  TensorDataType* __restrict__ input_grad_buffer,
  Size3 input_grad_strides,
  const TensorDataType* __restrict__ y_dot_dy) {

  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;
  for (size_t k = gidz; k < output_dims[0]; k += nthreadsz) {
    for (size_t j = gidy; j < output_dims[1]; j += nthreadsy) {
      const auto& _y_dot_dy = y_dot_dy[j + k*output_dims[1]];
      for (size_t i = gidx; i < output_dims[2]; i += nthreadsx) {
        const auto& y = output_buffer[k * output_strides[0]
                                      + j * output_strides[1]
                                      + i * output_strides[2]];
        const auto& dy = output_grad_buffer[k * output_grad_strides[0]
                                            + j * output_grad_strides[1]
                                            + i * output_grad_strides[2]];
        auto& dx = input_grad_buffer[k * input_grad_strides[0]
                                     + j * input_grad_strides[1]
                                     + i * input_grad_strides[2]];
        dx = y * (dy - _y_dot_dy);
      }
    }
  }

}

/** @brief Backprop */
template <typename TensorDataType>
void bp_impl(size_t num_channels,
             size_t channel_size,
             const El::AbstractDistMatrix<TensorDataType>& output,
             const El::AbstractDistMatrix<TensorDataType>& output_grad,
             El::AbstractDistMatrix<TensorDataType>& input_grad) {

  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_output = dynamic_cast<const LocalMat&>(output.LockedMatrix());
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(output_grad.LockedMatrix());
  auto& local_input_grad = dynamic_cast<LocalMat&>(input_grad.Matrix());

  // Dimensions
  const size_t local_mini_batch_size = local_output.Width();

  // dot(y,dL/dy)
  LocalMat local_y_dot_dy(num_channels, local_mini_batch_size);
  El::Zero(local_y_dot_dy);

  auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_y_dot_dy),
                                     gpu::get_sync_info(local_output_grad),
                                     gpu::get_sync_info(local_output),
                                     gpu::get_sync_info(local_input_grad));

  if (!local_output.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
    gpu_lib::clip_grid_dims(grid_dims);
    hydrogen::gpu::LaunchKernel(
      bp_y_dot_dy_kernel<TensorDataType, block_size>,
      grid_dims, block_dims, 0, multisync,
      Size3{local_mini_batch_size, num_channels, channel_size},
      local_output.LockedBuffer(),
      Size3{static_cast<size_t>(local_output.LDim()), channel_size, 1},
      local_output_grad.LockedBuffer(),
      Size3{static_cast<size_t>(local_output_grad.LDim()), channel_size, 1},
      local_y_dot_dy.Buffer());
  }

  // Compute gradient w.r.t. input
  if (!local_output.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
    gpu_lib::clip_grid_dims(grid_dims);
    hydrogen::gpu::LaunchKernel(
      bp_input_grad_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      Size3{local_mini_batch_size, num_channels, channel_size},
      local_output.LockedBuffer(),
      Size3{static_cast<size_t>(local_output.LDim()), channel_size, 1},
      local_output_grad.LockedBuffer(),
      Size3{static_cast<size_t>(local_output_grad.LDim()), channel_size, 1},
      local_input_grad.Buffer(),
      Size3{static_cast<size_t>(local_input_grad.LDim()), channel_size, 1},
      local_y_dot_dy.LockedBuffer());
  }

}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_softmax_layer<TensorDataType,Layout,Device>::bp_compute() {

  #ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()){
    this->get_distconv_adapter().bp_compute();
    return ;
  }
  #endif // LBANN_HAS_DISTCONV

  const size_t num_channels = this->get_output_dims().front();
  const size_t channel_size = this->get_output_size() / num_channels;
  bp_impl(num_channels,
          channel_size,
          this->get_activations(),
          this->get_prev_error_signals(),
          this->get_error_signals());
}

// =========================================================
// Explicit template instantiation
// =========================================================

#define PROTO(T)                                        \
  template class channelwise_softmax_layer<             \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
