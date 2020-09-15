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

#define LBANN_CHANNELWISE_MEAN_LAYER_INSTANTIATE
#include "lbann/layers/misc/channelwise_mean.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

template <El::Int block_size, typename TensorDataType>
__global__ void mean_kernel(El::Int num_channels,
                            El::Int channel_size,
                            El::Int width,
                            const TensorDataType* __restrict__ input,
                            El::Int input_ldim,
                            TensorDataType* __restrict__ output,
                            El::Int output_ldim) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int bidz = blockIdx.z;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksy = gridDim.y;
  const El::Int nblocksz = gridDim.z;

  // Compute local contribution for each channel
  for (El::Int col = bidz; col < width; col += nblocksz) {
    for (El::Int channel = bidy; channel < num_channels; channel += nblocksy) {

      // Sum for each thread
      TensorDataType private_sum = 0;
      for (El::Int i = gidx; i < channel_size; i += nthreadsx) {
        private_sum += input[i + channel*channel_size + col*input_ldim];
      }

      // Shared memory reduction to get sum for each block
      /// @todo unroll loops
      __shared__ TensorDataType shared_sums[block_size];
      shared_sums[tid] = private_sum;
      for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
          shared_sums[tid] += shared_sums[tid + stride];
        }
      }
      if (tid == 0) {
        cuda::atomic_add(&output[channel + col * output_ldim],
                         shared_sums[0] / TensorDataType(channel_size));
      }

    }
  }

}

template <typename TensorDataType>
__global__ void backprop_kernel(El::Int num_channels,
                                El::Int channel_size,
                                El::Int width,
                                const TensorDataType* __restrict__ gradient_wrt_output,
                                El::Int gradient_wrt_output_ldim,
                                TensorDataType* __restrict__ gradient_wrt_input,
                                El::Int gradient_wrt_input_ldim) {

  // Indices
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;
  const El::Int bidz = blockIdx.z;
  const El::Int nthreadsx = blockDim.x * gridDim.x;
  const El::Int nblocksy = gridDim.y;
  const El::Int nblocksz = gridDim.z;

  // Compute local contribution for each channel
  for (El::Int col = bidz; col < width; col += nblocksz) {
    for (El::Int channel = bidy; channel < num_channels; channel += nblocksy) {
      const auto& dy = gradient_wrt_output[channel + col * gradient_wrt_output_ldim];
      const auto& dx = dy / TensorDataType(channel_size);
      for (El::Int i = gidx; i < channel_size; i += nthreadsx) {
        gradient_wrt_input[i + channel*channel_size + col*gradient_wrt_input_ldim] = dx;
      }
    }
  }

}


} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_mean_layer<TensorDataType, Layout, Device>::fp_compute() {

  // Local matrices
  const auto& local_input = this->get_local_prev_activations();
  auto& local_output = this->get_local_activations();

  // Dimensions
  // Note: channel_size is the number of input entries per channel and
  // local_width is the number of local mini-batch samples.
  const auto& input_dims = this->get_input_dims();
  const El::Int num_channels = input_dims[0];
  const El::Int channel_size = std::accumulate(input_dims.begin() + 1,
                                               input_dims.end(),
                                               1, std::multiplies<int>());
  const auto& local_width = local_input.Width();

  // Compute channel-wise mean
  El::Zero(local_output);
  if (!local_input.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_output),
                                       gpu::get_sync_info(local_input));
    constexpr El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_width;
    hydrogen::gpu::LaunchKernel(
      mean_kernel<block_size, TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      num_channels, channel_size, local_width,
      local_input.LockedBuffer(), local_input.LDim(),
      local_output.Buffer(), local_output.LDim());
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_mean_layer<TensorDataType, Layout, Device>::bp_compute() {
  // Local matrices
  const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
  auto& local_gradient_wrt_input = this->get_local_error_signals();

  // Dimensions
  // Note: channel_size is the number of input entries per channel and
  // local_width is the number of local mini-batch samples.
  const auto& input_dims = this->get_input_dims();
  const El::Int num_channels = input_dims[0];
  const El::Int channel_size = std::accumulate(input_dims.begin() + 1,
                                               input_dims.end(),
                                               1, std::multiplies<int>());
  const auto& local_width = local_gradient_wrt_input.Width();

  // Compute gradients
  if (!local_gradient_wrt_input.IsEmpty()) {
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(local_gradient_wrt_input),
                        gpu::get_sync_info(local_gradient_wrt_output));
    constexpr El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_width;
    hydrogen::gpu::LaunchKernel(
      backprop_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      num_channels, channel_size, local_width,
      local_gradient_wrt_output.LockedBuffer(),
      local_gradient_wrt_output.LDim(),
      local_gradient_wrt_input.Buffer(), local_gradient_wrt_input.LDim());
  }

}

#define PROTO(T)                     \
  template class channelwise_mean_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
