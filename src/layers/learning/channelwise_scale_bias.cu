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

#define LBANN_CHANNELWISE_SCALE_BIAS_LAYER_INSTANTIATE
#include "lbann/layers/learning/channelwise_scale_bias.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#ifdef HYDROGEN_HAVE_CUB
#include "cub/block/block_reduce.cuh"
#endif // HYDROGEN_HAVE_CUB

namespace lbann {

namespace {

/**
 *  Block dimensions: bsizex x bsizey x 1
 *
 *  Grid dimensions: (channel_size / bsizex) x (width / bsizey) x num_channels
 */
template <typename TensorDataType>
__global__ void fp_kernel(size_t num_channels,
                          size_t channel_size,
                          size_t width,
                          const TensorDataType* __restrict__ input,
                          size_t input_ldim,
                          TensorDataType* __restrict__ output,
                          size_t output_ldim,
                          const TensorDataType* __restrict__ scale,
                          const TensorDataType* __restrict__ bias) {

  // Indices
  const size_t bidz = blockIdx.z;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nblocksz = gridDim.z;

  // Apply channel-wise scale/bias
  for (size_t channel = bidz; channel < num_channels; channel += nblocksz) {
    const auto a = scale[channel];
    const auto b = bias[channel];
    const size_t row_start = channel * channel_size;
    const size_t row_end = (channel + 1) * channel_size;
    const size_t col_start = 0;
    const size_t col_end = width;
    for (size_t col = col_start+gidy; col < col_end; col += nthreadsy) {
      for (size_t row = row_start+gidx; row < row_end; row += nthreadsx) {
        const auto& x = input[row + col*input_ldim];
        auto& y = output[row + col*output_ldim];
        y = a * x + b;
      }
    }
  }

}

/**
 *  Block dimensions: bsizex x bsizey x 1
 *
 *  Grid dimensions: (channel_size / bsizex) x (width / bsizey) x num_channels
 */
template <size_t bsizex, size_t bsizey, typename TensorDataType>
__global__ void bp_kernel(size_t num_channels,
                          size_t channel_size,
                          size_t width,
                          const TensorDataType* __restrict__ input,
                          size_t input_ldim,
                          const TensorDataType* __restrict__ gradient_wrt_output,
                          size_t gradient_wrt_output_ldim,
                          TensorDataType* __restrict__ gradient_wrt_input,
                          size_t gradient_wrt_input_ldim,
                          const TensorDataType* __restrict__ scale,
                          TensorDataType* __restrict__ gradient_wrt_scale,
                          TensorDataType* __restrict__ gradient_wrt_bias) {

  // Indices
  const size_t tid = threadIdx.x + threadIdx.y * blockDim.x;
  const size_t bidz = blockIdx.z;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nblocksz = gridDim.z;

  for (size_t channel = bidz; channel < num_channels; channel += nblocksz) {

    // Accumulate gradient contributions for thread in private memory
    TensorDataType private_da{0}, private_db{0};
    const auto a = scale[channel];
    const size_t row_start = channel * channel_size;
    const size_t row_end = (channel + 1) * channel_size;
    const size_t col_start = 0;
    const size_t col_end = width;
    for (size_t col = col_start+gidy; col < col_end; col += nthreadsy) {
      for (size_t row = row_start+gidx; row < row_end; row += nthreadsx) {
        const auto& x = input[row + col*input_ldim];
        const auto& dy = gradient_wrt_output[row + col*gradient_wrt_output_ldim];
        auto& dx = gradient_wrt_input[row + col*gradient_wrt_input_ldim];
        private_da += x * dy;
        private_db += dy;
        dx = a * dy;
      }
    }

    // Accumulate gradient contributions for block and add to result
#ifdef HYDROGEN_HAVE_CUB
    constexpr auto reduce_algo = cub::BLOCK_REDUCE_WARP_REDUCTIONS;
    using BlockReduce = cub::BlockReduce<TensorDataType, bsizex, reduce_algo, bsizey>;
    __shared__ typename BlockReduce::TempStorage workspace;
    __syncthreads();
    const auto da = BlockReduce(workspace).Sum(private_da);
    if (tid == 0) {
      cuda::atomic_add(&gradient_wrt_scale[channel], da);
    }
    __syncthreads();
    const auto db = BlockReduce(workspace).Sum(private_db);
    if (tid == 0) {
      cuda::atomic_add(&gradient_wrt_bias[channel], db);
    }
#else
    __shared__ TensorDataType workspace[bsizex*bsizey];
    workspace[tid] = private_da;
    for (size_t stride = bsizex*bsizey/2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        workspace[tid] += workspace[tid + stride];
      }
    }
    if (tid == 0) {
      cuda::atomic_add(&gradient_wrt_scale[channel], workspace[0]);
    }
    workspace[tid] = private_db;
    for (size_t stride = bsizex*bsizey/2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        workspace[tid] += workspace[tid + stride];
      }
    }
    if (tid == 0) {
      cuda::atomic_add(&gradient_wrt_bias[channel], workspace[0]);
    }
#endif // HYDROGEN_HAVE_CUB

  }

}

} // namespace

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void channelwise_scale_bias_layer<TensorDataType, T_layout, Dev>::fp_compute() {
  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;

  // Local matrices
  const auto& local_input = dynamic_cast<const GPUMatType&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<GPUMatType&>(this->get_local_activations());
  const auto& local_weights = dynamic_cast<const GPUMatType&>(this->weights_values(0).LockedMatrix());
  const auto local_scale = El::LockedView(local_weights,
                                          El::ALL, El::IR(0));
  const auto local_bias = El::LockedView(local_weights,
                                         El::ALL, El::IR(1));

  // Dimensions
  // Note: channel_size is the number of input entries per channel and
  // local_width is the number of local mini-batch samples.
  const auto dims = this->get_output_dims();
  const El::Int num_channels = dims[0];
  const El::Int channel_size = std::accumulate(dims.begin() + 1,
                                               dims.end(),
                                               1, std::multiplies<int>());
  const El::Int local_width = local_input.Width();

  // Apply channel-wise scale and bias
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size_x = 256;
    constexpr size_t block_size_y = 1;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size_x;
    block_dims.y = block_size_y;
    grid_dims.x = (channel_size + block_size_x - 1) / block_size_x;
    grid_dims.y = (local_width + block_size_y - 1) / block_size_y;
    grid_dims.z = num_channels;
    fp_kernel
      <<<grid_dims, block_dims, 0, hydrogen::cuda::GetDefaultStream()>>>(
        num_channels, channel_size, local_width,
        local_input.LockedBuffer(), local_input.LDim(),
        local_output.Buffer(), local_output.LDim(),
        local_scale.LockedBuffer(),
        local_bias.LockedBuffer());
  }

}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void channelwise_scale_bias_layer<TensorDataType, T_layout, Dev>::bp_compute() {

  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;

  // Local matrices
  const auto& local_input = dynamic_cast<const GPUMatType&>(this->get_local_prev_activations());
  const auto& local_gradient_wrt_output = dynamic_cast<const GPUMatType&>(this->get_local_prev_error_signals());
  auto& local_gradient_wrt_input = dynamic_cast<GPUMatType&>(this->get_local_error_signals());
  const auto& local_weights = dynamic_cast<const GPUMatType&>(this->weights_values(0).LockedMatrix());
  auto& local_gradient_wrt_weights = dynamic_cast<GPUMatType&>(this->m_weights_gradient->Matrix());
  const auto local_scale = El::LockedView(local_weights,
                                          El::ALL, El::IR(0));
  auto local_gradient_wrt_scale = El::View(local_gradient_wrt_weights,
                                           El::ALL, El::IR(0));
  auto local_gradient_wrt_bias = El::View(local_gradient_wrt_weights,
                                          El::ALL, El::IR(1));

  // Dimensions
  // Note: channel_size is the number of input entries per channel and
  // local_width is the number of local mini-batch samples.
  const auto dims = this->get_output_dims();
  const El::Int num_channels = dims[0];
  const El::Int channel_size = std::accumulate(dims.begin() + 1,
                                               dims.end(),
                                               1, std::multiplies<int>());
  const El::Int local_width = local_input.Width();

  // Compute gradients
  El::Zero(local_gradient_wrt_weights);
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size_x = 256;
    constexpr size_t block_size_y = 1;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size_x;
    block_dims.y = block_size_y;
    grid_dims.x = (channel_size + block_size_x - 1) / block_size_x;
    grid_dims.y = (local_width + block_size_y - 1) / block_size_y;
    grid_dims.z = num_channels;
    bp_kernel<block_size_x, block_size_y>
      <<<grid_dims, block_dims, 0, hydrogen::cuda::GetDefaultStream()>>>(
        num_channels, channel_size, local_width,
        local_input.LockedBuffer(), local_input.LDim(),
        local_gradient_wrt_output.LockedBuffer(), local_gradient_wrt_output.LDim(),
        local_gradient_wrt_input.Buffer(), local_gradient_wrt_input.LDim(),
        local_scale.LockedBuffer(),
        local_gradient_wrt_scale.Buffer(),
        local_gradient_wrt_bias.Buffer());
  }

  // Update optimizer with gradient
  auto* opt = this->get_weights(0).get_optimizer();
  if (opt != nullptr) {
    opt->add_to_gradient(*this->m_weights_gradient, El::TypeTraits<TensorDataType>::One(), true);
  }

}

#define PROTO(T)                                      \
  template class channelwise_scale_bias_layer<        \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
