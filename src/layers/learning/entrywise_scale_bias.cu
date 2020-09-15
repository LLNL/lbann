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

#define LBANN_ENTRYWISE_SCALE_BIAS_LAYER_INSTANTIATE
#include "lbann/layers/learning/entrywise_scale_bias.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

/**
 *  Block dimensions: bsizex x bsizey x 1
 *
 *  Grid dimensions: (height / bsizex) x (width / bsizey) x num_channels
 */
template <typename TensorDataType>
__global__ void fp_kernel(size_t height,
                          size_t width,
                          const TensorDataType* __restrict__ input,
                          size_t input_ldim,
                          TensorDataType* __restrict__ output,
                          size_t output_ldim,
                          const TensorDataType* __restrict__ scale,
                          const TensorDataType* __restrict__ bias) {
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  for (size_t row = gidx; row < height; row += nthreadsx) {
    const auto a = scale[row];
    const auto b = bias[row];
    for (size_t col = gidy; col < width; col += nthreadsy) {
      const auto& x = input[row + col*input_ldim];
      auto& y = output[row + col*output_ldim];
      y = a * x + b;
    }
  }
}

/**
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (height / bsize) x 1 x 1
 */
template <typename TensorDataType>
__global__ void bp_kernel(size_t height,
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
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  for (size_t row = gid; row < height; row += nthreads) {
    const auto a = scale[row];
    TensorDataType da{0}, db{0};
    for (size_t col = 0; col < width; ++col) {
      const auto& x = input[row + col * input_ldim];
      const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
      dx = a * dy;
      da += x * dy;
      db += dy;
    }
    gradient_wrt_scale[row] = da;
    gradient_wrt_bias[row] = db;
  }
}

template <typename TensorDataType>
void fp_impl(
  const El::Matrix<TensorDataType, El::Device::GPU>& local_input,
  El::Matrix<TensorDataType, El::Device::GPU>& local_output,
  El::Matrix<TensorDataType, El::Device::GPU> const& local_scale_bias) {

  // Local matrices
  const auto local_scale = El::LockedView(local_scale_bias,
                                          El::ALL, El::IR(0));
  const auto local_bias = El::LockedView(local_scale_bias,
                                         El::ALL, El::IR(1));

  // Apply entry-wise scale and bias
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size_x = 256;
    constexpr size_t block_size_y = 1;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size_x;
    block_dims.y = block_size_y;
    grid_dims.x = (local_height + block_size_x - 1) / block_size_x;
    grid_dims.y = (local_width + block_size_y - 1) / block_size_y;
    fp_kernel<<<grid_dims, block_dims, 0, hydrogen::cuda::GetDefaultStream()>>>(
        local_height, local_width,
        local_input.LockedBuffer(), local_input.LDim(),
        local_output.Buffer(), local_output.LDim(),
        local_scale.LockedBuffer(),
        local_bias.LockedBuffer());
  }

}

template <typename TensorDataType>
void bp_impl(
  const El::Matrix<TensorDataType, El::Device::GPU>& local_input,
  const El::Matrix<TensorDataType, El::Device::GPU>& local_gradient_wrt_output,
  El::Matrix<TensorDataType, El::Device::GPU>& local_gradient_wrt_input,
  El::Matrix<TensorDataType, El::Device::GPU> const& local_scale_bias,
  El::Matrix<TensorDataType, El::Device::GPU>& local_gradient_wrt_scale_bias) {

  // Local matrices
  const auto local_scale = El::LockedView(local_scale_bias,
                                          El::ALL, El::IR(0));
  auto local_gradient_wrt_scale = El::View(local_gradient_wrt_scale_bias,
                                           El::ALL, El::IR(0));
  auto local_gradient_wrt_bias = El::View(local_gradient_wrt_scale_bias,
                                          El::ALL, El::IR(1));

  // Compute gradients
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();
  El::Zero(local_gradient_wrt_scale_bias);
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_height + block_size - 1) / block_size;
    bp_kernel <<<grid_dims, block_dims, 0, hydrogen::cuda::GetDefaultStream()>>>(
      local_height, local_width,
      local_input.LockedBuffer(), local_input.LDim(),
      local_gradient_wrt_output.LockedBuffer(), local_gradient_wrt_output.LDim(),
      local_gradient_wrt_input.Buffer(), local_gradient_wrt_input.LDim(),
      local_scale.LockedBuffer(),
      local_gradient_wrt_scale.Buffer(),
      local_gradient_wrt_bias.Buffer());
  }

}

} // namespace

// Template instantiation

template <typename TensorDataType, data_layout Layout, El::Device Device>
void entrywise_scale_bias_layer<TensorDataType, Layout, Device>::fp_compute() {
  using LocalMatType = El::Matrix<TensorDataType, Device>;
  fp_impl(dynamic_cast<const LocalMatType&>(this->get_local_prev_activations()),
          dynamic_cast<LocalMatType&>(this->get_local_activations()),
          dynamic_cast<LocalMatType const&>(
            this->weights_values(0).LockedMatrix()));
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void entrywise_scale_bias_layer<TensorDataType, Layout, Device>::bp_compute() {
  using LocalMatType = El::Matrix<TensorDataType, Device>;

  auto& scale_bias = this->get_weights(0);
  auto& gradient_wrt_scale_bias = *this->m_weights_gradient;

  bp_impl(dynamic_cast<const LocalMatType&>(this->get_local_prev_activations()),
          dynamic_cast<const LocalMatType&>(this->get_local_prev_error_signals()),
          dynamic_cast<LocalMatType&>(this->get_local_error_signals()),
          dynamic_cast<LocalMatType const&>(
            this->weights_values(0).LockedMatrix()),
          dynamic_cast<LocalMatType&>(gradient_wrt_scale_bias.Matrix()));

  // Update optimizer with gradient
  auto* opt = scale_bias.get_optimizer();
  if (opt != nullptr) {
    opt->add_to_gradient(gradient_wrt_scale_bias, TensorDataType{1}, true);
  }
}

LBANN_LAYER_DEFAULT_BUILDER(entrywise_scale_bias)

#define PROTO(T)                                                     \
  template class entrywise_scale_bias_layer<                         \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;                 \
  template class entrywise_scale_bias_layer<                         \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>;                \
  LBANN_LAYER_BUILDER_ETI(entrywise_scale_bias, T, El::Device::GPU)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
