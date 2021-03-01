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

#define LBANN_GATHER_LAYER_INSTANTIATE
#include "lbann/layers/transform/gather.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

using Dim2 = gpu_lib::array<size_t, 2>;

/** @brief Kernel for scattering a 2D tensor along dim 1
 *
 *  output(j,indices(j,i)) = values(j,i)
 *
 *  Block dimensions: bdimx x bdimy x 1
 *
 *  Grid dimensions: (values_dim[1] / bdimx) x (values_dim[0] / bdimy) x 1
 */
template <typename T>
__global__ void scatter2d_kernel(
  const T* __restrict__ indices,
  Dim2 indices_strides,
  const T* __restrict__ values,
  Dim2 values_dims,
  Dim2 values_strides,
  T* __restrict__ output,
  Dim2 output_dims,
  Dim2 output_strides) {

  // Indices
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;

  for (size_t j=gidy; j<values_dims[0]; j+=nthreadsy) {
    for (size_t i=gidx; i<values_dims[1]; i+=nthreadsx) {
      const auto ind = static_cast<El::Int>(
        gpu_lib::floor(
          indices[j*indices_strides[0] + i*indices_strides[1]]));
      if (0<=ind && ind<static_cast<El::Int>(output_dims[1])) {
        const auto& x = values[j*values_strides[0] + i*values_strides[1]];
        auto& y = output[j*output_strides[0] + ind*output_strides[1]];
        gpu_lib::atomic_add(&y, x);
      }
    }
  }

}

/** @brief Kernel for gathering a 2D tensor along dim 1
 *
 *  output(j,i) = values(j,indices(j,i))
 *
 *  Block dimensions: bdimx x bdimy x 1
 *
 *  Grid dimensions: (output_dim[1] / bdimx) x (output_dim[0] / bdimy) x 1
 */
template <typename T>
__global__ void gather2d_kernel(
  const T* __restrict__ indices,
  Dim2 indices_strides,
  const T* __restrict__ values,
  Dim2 values_dims,
  Dim2 values_strides,
  T* __restrict__ output,
  Dim2 output_dims,
  Dim2 output_strides) {

  // Indices
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;

  for (size_t j=gidy; j<output_dims[0]; j+=nthreadsy) {
    for (size_t i=gidx; i<output_dims[1]; i+=nthreadsx) {
      const auto ind = static_cast<El::Int>(
        gpu_lib::floor(
          indices[j*indices_strides[0] + i*indices_strides[1]]));
      auto& y = output[j*output_strides[0] + i*output_strides[1]];
      if (0<=ind && ind<static_cast<El::Int>(values_dims[1])) {
        y = values[j*values_strides[0] + ind*values_strides[1]];
      }
      else {
        y = T{0.f};
      }
    }
  }

}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gather_layer<TensorDataType, Layout, Device>::fp_compute() {

  // Local matrices
  const auto& local_values = this->get_local_prev_activations(0);
  const auto& local_indices = this->get_local_prev_activations(1);
  auto& local_output = this->get_local_activations();
  const size_t values_size = this->get_input_size(0);
  const size_t output_size = this->get_output_size();
  const size_t local_mini_batch_size = local_indices.Width();

  // Gather into output tensor
  if (!local_output.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_output),
                                       gpu::get_sync_info(local_values),
                                       gpu::get_sync_info(local_indices));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    block_dims.y = 1;
    grid_dims.x = (output_size + block_dims.x - 1) / block_dims.x;
    grid_dims.y = (local_mini_batch_size + block_dims.y - 1) / block_dims.y;
    hydrogen::gpu::LaunchKernel(
      gather2d_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_indices.LockedBuffer(),
      Dim2{static_cast<size_t>(local_indices.LDim()), 1},
      local_values.LockedBuffer(),
      Dim2{local_mini_batch_size, values_size},
      Dim2{static_cast<size_t>(local_values.LDim()), 1},
      local_output.Buffer(),
      Dim2{local_mini_batch_size, output_size},
      Dim2{static_cast<size_t>(local_output.LDim()), 1});
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gather_layer<TensorDataType, Layout, Device>::bp_compute() {

  // Local matrices
  const auto& local_indices = this->get_local_prev_activations(1);
  const auto& local_output_grad = this->get_local_prev_error_signals();
  auto& local_values_grad = this->get_local_error_signals(0);
  auto& local_indices_grad = this->get_local_error_signals(1);
  const size_t values_size = this->get_input_size(0);
  const size_t output_size = this->get_output_size();
  const size_t local_mini_batch_size = local_indices.Width();

  // Zero out gradient w.r.t. indices
  El::Zero(local_indices_grad);

  // Scatter into output matrix
  El::Zero(local_values_grad);
  if (!local_output_grad.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_values_grad),
                                       gpu::get_sync_info(local_output_grad),
                                       gpu::get_sync_info(local_indices));
    constexpr size_t block_size = 64;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    block_dims.y = 1;
    grid_dims.x = (output_size + block_dims.x - 1) / block_dims.x;
    grid_dims.y = (local_mini_batch_size + block_dims.y - 1) / block_dims.y;
    hydrogen::gpu::LaunchKernel(
      scatter2d_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_indices.LockedBuffer(),
      Dim2{static_cast<size_t>(local_indices.LDim()), 1},
      local_output_grad.LockedBuffer(),
      Dim2{local_mini_batch_size, output_size},
      Dim2{static_cast<size_t>(local_output_grad.LDim()), 1},
      local_values_grad.Buffer(),
      Dim2{local_mini_batch_size, values_size},
      Dim2{static_cast<size_t>(local_values_grad.LDim()), 1});
  }

}

#define PROTO(T)                                        \
  template class gather_layer<                         \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
