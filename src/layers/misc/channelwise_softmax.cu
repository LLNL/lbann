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

#define LBANN_CHANNELWISE_SOFTMAX_LAYER_INSTANTIATE
#include "lbann/layers/misc/channelwise_softmax.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

using Size3 = cuda::array<size_t,3>;

/** @brief Max functor */
template <class T>
struct max_op {
  __device__ __forceinline__
  DataType operator()(const T& x1, const T& x2) const {
    return cuda::max(x1, x2);
  }
};

} // namespace <anon>

// =========================================================
// Forward prop
// =========================================================

namespace {

/** @brief Max reduction over last dimension of 3D tensor.
 *
 *  Each CUDA block computes the max over a subset of tensor entries
 *  in @c vals and outputs the result to @c maxvals. This should be
 *  repeated multiple times to fully reduce the last tensor dimension.
 *
 *  Block dimensions: bdimx x 1 x 1
 *
 *  Grid dimensions: (vals_dims[2] / bdimx) x vals_dims[1] x vals_dims[0]
 *
 *  maxvals: vals_dims[0] x vals_dims[1] x (vals_dims[2] / bdimx)
 */
template <typename TensorDataType, size_t bdimx>
__global__ void fp_max_kernel(
  Size3 vals_dims,
  const TensorDataType* __restrict__ vals_buffer,
  Size3 vals_strides,
  TensorDataType* __restrict__ maxvals_buffer,
  Size3 maxvals_strides) {

  // Indices and dimensions
  constexpr size_t bdimy = 1;
  constexpr size_t bdimz = 1;
  const size_t tid = threadIdx.x;
  const size_t bidx = blockIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;

  for (size_t k = gidz; k < vals_dims[0]; k += nthreadsz) {
    for (size_t j = gidy; j < vals_dims[1]; j += nthreadsy) {

      // Find largest value for each thread
      TensorDataType maxval{-cuda::infinity<TensorDataType>()};
      for (size_t i = gidx; i < vals_dims[2]; i += nthreadsx) {
        const auto& val = vals_buffer[k * vals_strides[0]
                                      + j * vals_strides[1]
                                      + i * vals_strides[2]];
        maxval = cuda::max(maxval, val);
      }

      // Find largest value for each block
      maxval = cuda::block_reduce<bdimx,bdimy,bdimz,TensorDataType,max_op<TensorDataType>>(maxval);
      if (tid == 0) {
        const auto& pos = (k * maxvals_strides[0]
                           + j * maxvals_strides[1]
                           + bidx * maxvals_strides[2]);
        maxvals_buffer[pos] = maxval;
      }

    }
  }

}

/** Compute softmax denominator.
 *
 *  denom = sum( exp(x_i-shift) )
 *
 *  Block dimensions: bdimx x 1 x 1
 *
 *  Grid dimensions: (input_dims[2] / bdimx) x input_dims[1] x input_dims[0]
 *
 *  shifts and denoms are fully-packed 2D tensors with dimensions of
 *  input_dims[0] x input_dims[1].
 */
template <typename TensorDataType, size_t bdimx>
__global__ void fp_denom_kernel(
  Size3 input_dims,
  const TensorDataType* __restrict__ input_buffer,
  Size3 input_strides,
  const TensorDataType* __restrict__ shifts,
  TensorDataType* __restrict__ denoms) {

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

  for (size_t k = gidz; k < input_dims[0]; k += nthreadsz) {
    for (size_t j = gidy; j < input_dims[1]; j += nthreadsy) {

      // Compute contribution from each thread
      const auto& shift = shifts[j + k*input_dims[1]];
      TensorDataType denom{0.};
      for (size_t i = gidx; i < input_dims[2]; i += nthreadsx) {
        const auto& x = input_buffer[k * input_strides[0]
                                     + j * input_strides[1]
                                     + i * input_strides[2]];
        denom += cuda::exp(x-shift);
      }

      // Compute contribution from each block
      denom = cuda::block_reduce<bdimx,bdimy,bdimz>(denom);
      if (tid == 0) {
        cuda::atomic_add(&denoms[j+k*input_dims[1]], denom);
      }

    }
  }

}

/** Compute softmax.
 *
 *  y_i = exp(x_i-shift) / denom
 *
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (input_dims[2] / bdimx) x (input_dims[1] / bdimy) x (input_dims[0] / bdimz)
 *
 *  shifts and denoms are fully-packed 2D tensors with dimensions of
 *  input_dims[0] x input_dims[1].
 */
template <typename TensorDataType>
__global__ void fp_output_kernel(
  Size3 input_dims,
  const TensorDataType* __restrict__ input_buffer,
  Size3 input_strides,
  TensorDataType* __restrict__ output_buffer,
  Size3 output_strides,
  const TensorDataType* __restrict__ shifts,
  const TensorDataType* __restrict__ denoms) {

  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;
  for (size_t k = gidz; k < input_dims[0]; k += nthreadsz) {
    for (size_t j = gidy; j < input_dims[1]; j += nthreadsy) {
      const auto& shift = shifts[j + k*input_dims[1]];
      const auto& denom = denoms[j + k*input_dims[1]];
      for (size_t i = gidx; i < input_dims[2]; i += nthreadsx) {
        const auto& x = input_buffer[k * input_strides[0]
                                     + j * input_strides[1]
                                     + i * input_strides[2]];
        auto& y = output_buffer[k * output_strides[0]
                                + j * output_strides[1]
                                + i * output_strides[2]];
        y = cuda::exp(x-shift) / denom;
      }
    }
  }

}

/** @brief Forward prop */
template <typename TensorDataType>
void fp_impl(size_t num_channels,
             size_t channel_size,
             const El::AbstractDistMatrix<TensorDataType>& input,
             El::AbstractDistMatrix<TensorDataType>& output) {

  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<LocalMat&>(output.Matrix());

  auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_output),
                                     gpu::get_sync_info(local_input));

  // Dimensions
  const size_t local_mini_batch_size = local_input.Width();
  // const Size3 input_dims{local_mini_batch_size, num_channels, channel_size};

  // Compute softmax shifts
  LocalMat local_shifts;
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
    LocalMat maxvals(grid_dims.x * num_channels, local_mini_batch_size);
    hydrogen::gpu::LaunchKernel(
      fp_max_kernel<TensorDataType, block_size>,
      grid_dims, block_dims, 0, multisync,
      Size3{local_mini_batch_size, num_channels, channel_size},
      local_input.LockedBuffer(),
      Size3{static_cast<size_t>(local_input.LDim()), channel_size, 1},
      maxvals.Buffer(),
      Size3{static_cast<size_t>(maxvals.LDim()), grid_dims.x, 1});
    while (grid_dims.x > 1) {
      const size_t prev_dim = grid_dims.x;
      grid_dims.x = (prev_dim + block_size - 1) / block_size;
      const LocalMat prev_maxvals(std::move(maxvals));
      maxvals.Resize(grid_dims.x * num_channels, local_mini_batch_size);
      hydrogen::gpu::LaunchKernel(
        fp_max_kernel<TensorDataType, block_size>,
        grid_dims, block_dims, 0, multisync,
        Size3{local_mini_batch_size, num_channels, prev_dim},
        prev_maxvals.LockedBuffer(),
        Size3{static_cast<size_t>(prev_maxvals.LDim()), prev_dim, 1},
        maxvals.Buffer(),
        Size3{static_cast<size_t>(maxvals.LDim()), grid_dims.x, 1});
    }
    local_shifts = std::move(maxvals);
  }

  // Compute softmax denominators
  LocalMat local_denoms(num_channels, local_mini_batch_size);
  El::Zero(local_denoms);
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
    hydrogen::gpu::LaunchKernel(
      fp_denom_kernel<TensorDataType, block_size>,
      grid_dims, block_dims, 0, multisync,
      Size3{local_mini_batch_size, num_channels, channel_size},
      local_input.LockedBuffer(),
      Size3{static_cast<size_t>(local_input.LDim()), channel_size, 1},
      local_shifts.LockedBuffer(),
      local_denoms.Buffer());
  }

  // Compute softmax
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
    hydrogen::gpu::LaunchKernel(
      fp_output_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      Size3{local_mini_batch_size, num_channels, channel_size},
      local_input.LockedBuffer(),
      Size3{static_cast<size_t>(local_input.LDim()), channel_size, 1},
      local_output.Buffer(),
      Size3{static_cast<size_t>(local_output.LDim()), channel_size, 1},
      local_shifts.LockedBuffer(),
      local_denoms.LockedBuffer());
  }

}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_softmax_layer<TensorDataType,Layout,Device>::fp_compute() {
  const size_t num_channels = this->get_output_dims().front();
  const size_t channel_size = this->get_output_size() / num_channels;
  fp_impl(num_channels,
          channel_size,
          this->get_prev_activations(),
          this->get_activations());
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
      _y_dot_dy = cuda::block_reduce<bdimx,bdimy,bdimz>(_y_dot_dy);
      if (tid == 0) {
        cuda::atomic_add(&y_dot_dy[j+k*output_dims[1]], _y_dot_dy);
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

  auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_input_grad),
                                     gpu::get_sync_info(local_output_grad),
                                     gpu::get_sync_info(local_output),
                                     gpu::get_sync_info(local_y_dot_dy));

  if (!local_output.IsEmpty()) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    grid_dims.z = local_mini_batch_size;
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
