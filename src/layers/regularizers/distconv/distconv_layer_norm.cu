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

#define LBANN_LAYERS_REGULARIZERS_DISTCONV_LAYER_NORM_INSTANTIATE

#include "../layer_norm_kernels.cuh"
#include "lbann/layers/regularizers/distconv/distconv_layer_norm.hpp"

#ifdef LBANN_HAS_DISTCONV

namespace distconv {

template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization<Backend, DataType>::calculate_forward_stats(
  const DCTensor<Allocator>& input,
  DCTensor<Allocator>& statistics)
{
  if (input.get_local_size() == 0) {
    util::MPIRootPrintStreamInfo() << "WARNING: EMPTY INPUT FOUND \n";
    return; // no op for empty inputs
  }

  const auto& input_dims = input.get_local_shape();
  const auto& statistics_dims = statistics.get_local_shape();
  const auto local_num_samples = input_dims[3];
  const auto global_num_samples = statistics_dims[3];

  const auto local_sample_size = std::accumulate(input_dims.begin(),
                                                 input_dims.end() - 1,
                                                 1,
                                                 std::multiplies<int>());

  using LocalMat = El::Matrix<DataType, El::Device::GPU>;
  LocalMat local_input(local_sample_size,
                       local_num_samples,
                       input.get_buffer(),
                       local_sample_size);

  LocalMat local_statistics(2, global_num_samples, statistics.get_buffer(), 2);

  El::Zero(local_statistics);
  auto local_means = El::View(local_statistics, El::IR(0), El::ALL);
  auto local_vars = El::View(local_statistics, El::IR(1), El::ALL);

  {
    using namespace hydrogen;
    auto multisync = El::MakeMultiSync(El::SyncInfoFromMatrix(local_statistics),
                                       El::SyncInfoFromMatrix(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_sample_size + block_size - 1) / block_size;
    grid_dims.y = local_num_samples;
    hydrogen::gpu::LaunchKernel(
      ::lbann::layer_norm_fp_sums_kernel<block_size, DataType>,
      grid_dims,
      block_dims,
      0,
      multisync,
      local_num_samples,
      local_sample_size,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_means.Buffer(),
      local_means.LDim(),
      local_vars.Buffer(),
      local_vars.LDim());
  }
}

template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization<Backend, DataType>::apply_normalization(
  const DCTensor<Allocator>& input,
  DCTensor<Allocator>& statistics,
  DCTensor<Allocator>& output)
{
  const auto& input_dims = input.get_local_shape();
  const auto& statistics_dims = statistics.get_local_shape();
  const auto local_num_samples = input_dims[3];
  const auto global_num_samples = statistics_dims[3];
  const auto local_sample_size = std::accumulate(input_dims.begin(),
                                                 input_dims.end() - 1,
                                                 1,
                                                 std::multiplies<int>());

  using LocalMat = El::Matrix<DataType, El::Device::GPU>;
  const LocalMat local_input(local_sample_size,
                             local_num_samples,
                             input.get_buffer(),
                             local_sample_size);

  LocalMat local_statistics(2, global_num_samples, statistics.get_buffer(), 2);

  LocalMat local_output(local_sample_size,
                        local_num_samples,
                        output.get_buffer(),
                        local_sample_size);

  auto local_means = El::View(local_statistics, El::IR(0), El::ALL);
  auto local_vars = El::View(local_statistics, El::IR(1), El::ALL);

  {
    using namespace hydrogen;
    auto sync_info = El::SyncInfoFromMatrix(local_statistics);
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_num_samples + block_size - 1) / block_size;
    hydrogen::gpu::LaunchKernel(
      ::lbann::layer_norm_fp_statistics_kernel<DataType>,
      grid_dims,
      block_dims,
      0,
      sync_info,
      local_sample_size,
      local_num_samples,
      local_means.Buffer(),
      local_means.LDim(),
      local_vars.Buffer(),
      local_vars.LDim());

    auto multisync = El::MakeMultiSync(El::SyncInfoFromMatrix(local_output),
                                       El::SyncInfoFromMatrix(local_statistics),
                                       El::SyncInfoFromMatrix(local_input));

    constexpr size_t block_size_output_kernel = 256;
    dim3 block_dims_output_kernel, grid_dims_output_kernel;
    block_dims_output_kernel.x = block_size_output_kernel;
    grid_dims_output_kernel.x =
      (local_sample_size + block_size - 1) / block_size_output_kernel;
    grid_dims_output_kernel.y = local_num_samples;
    hydrogen::gpu::LaunchKernel(::lbann::layer_norm_fp_output_kernel<DataType>,
                                grid_dims_output_kernel,
                                block_dims_output_kernel,
                                0,
                                multisync,
                                local_num_samples,
                                local_sample_size,
                                m_epsilon,
                                local_input.LockedBuffer(),
                                local_input.LDim(),
                                local_output.Buffer(),
                                local_output.LDim(),
                                local_means.Buffer(),
                                local_means.LDim(),
                                local_vars.Buffer(),
                                local_vars.LDim());
  }
}

template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization<Backend, DataType>::calculate_backward_stats(
  const DCTensor<Allocator>& input,
  const DCTensor<Allocator>& output_grad,
  const DCTensor<Allocator>& statistics,
  DCTensor<Allocator>& statistics_grad)
{

  const auto& input_dims = input.get_local_shape();
  const auto& statistics_dims = statistics.get_local_shape();
  const auto local_num_samples = input_dims[3];
  const auto global_num_samples = statistics_dims[3];
  const auto local_sample_size = std::accumulate(input_dims.begin(),
                                                 input_dims.end() - 1,
                                                 1,
                                                 std::multiplies<int>());
  using LocalMat = El::Matrix<DataType, El::Device::GPU>;
  const LocalMat local_input(local_sample_size,
                             local_num_samples,
                             input.get_buffer(),
                             local_sample_size);
  const LocalMat local_output_grad(local_sample_size,
                                   local_num_samples,
                                   output_grad.get_buffer(),
                                   local_sample_size);

  const LocalMat local_statistics(2,
                                  global_num_samples,
                                  statistics.get_buffer(),
                                  2);

  LocalMat local_statistics_grad(2,
                                 global_num_samples,
                                 statistics_grad.get_buffer(),
                                 2);
  const auto local_means = El::LockedView(local_statistics, El::IR(0), El::ALL);
  const auto local_vars = El::LockedView(local_statistics, El::IR(1), El::ALL);

  auto local_means_grad = El::View(local_statistics_grad, El::IR(0), El::ALL);
  auto local_vars_grad = El::View(local_statistics_grad, El::IR(1), El::ALL);

  {
    using namespace hydrogen;
    auto multisync =
      El::MakeMultiSync(El::SyncInfoFromMatrix(local_statistics_grad),
                        El::SyncInfoFromMatrix(local_output_grad),
                        El::SyncInfoFromMatrix(local_statistics),
                        El::SyncInfoFromMatrix(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_sample_size + block_size - 1) / block_size;
    grid_dims.y = local_num_samples;
    hydrogen::gpu::LaunchKernel(
      ::lbann::layer_norm_bp_statistics_grad_kernel<block_size, DataType>,
      grid_dims,
      block_dims,
      0,
      multisync,
      local_num_samples,
      local_sample_size,
      m_epsilon,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_output_grad.LockedBuffer(),
      local_output_grad.LDim(),
      local_means.LockedBuffer(),
      local_means.LDim(),
      local_vars.LockedBuffer(),
      local_vars.LDim(),
      local_means_grad.Buffer(),
      local_means_grad.LDim(),
      local_vars_grad.Buffer(),
      local_vars_grad.LDim());
  }
}

template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization<Backend, DataType>::apply_grad(
  const DCTensor<Allocator>& input,
  const DCTensor<Allocator>& output_grad,
  const DCTensor<Allocator>& statistics,
  const DCTensor<Allocator>& statistics_grad,
  DCTensor<Allocator>& input_grad)
{
  const auto& input_dims = input.get_local_shape();
  const auto& statistics_dims = statistics.get_local_shape();
  const auto local_num_samples = input_dims[3];
  const auto global_num_samples = statistics_dims[3];
  const auto local_sample_size = std::accumulate(input_dims.begin(),
                                                 input_dims.end() - 1,
                                                 1,
                                                 std::multiplies<int>());

  const auto global_sample_size = local_sample_size;

  using LocalMat = El::Matrix<DataType, El::Device::GPU>;
  const LocalMat local_input(local_sample_size,
                             local_num_samples,
                             input.get_buffer(),
                             local_sample_size);
  const LocalMat local_output_grad(local_sample_size,
                                   local_num_samples,
                                   output_grad.get_buffer(),
                                   local_sample_size);

  const LocalMat local_statistics(2,
                                  global_num_samples,
                                  statistics.get_buffer(),
                                  2);

  const LocalMat local_statistics_grad(2,
                                       global_num_samples,
                                       statistics_grad.get_buffer(),
                                       2);

  LocalMat local_input_grad(local_sample_size,
                            local_num_samples,
                            input_grad.get_buffer(),
                            local_sample_size);

  const auto local_means = El::LockedView(local_statistics, El::IR(0), El::ALL);
  const auto local_vars = El::LockedView(local_statistics, El::IR(1), El::ALL);
  const auto local_means_grad =
    El::LockedView(local_statistics_grad, El::IR(0), El::ALL);
  const auto local_vars_grad =
    El::LockedView(local_statistics_grad, El::IR(1), El::ALL);

  {
    using namespace hydrogen;
    auto multisync =
      El::MakeMultiSync(El::SyncInfoFromMatrix(local_statistics_grad),
                        El::SyncInfoFromMatrix(local_output_grad),
                        El::SyncInfoFromMatrix(local_statistics),
                        El::SyncInfoFromMatrix(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_sample_size + block_size - 1) / block_size;
    grid_dims.y = local_num_samples;
    hydrogen::gpu::LaunchKernel(
      ::lbann::layer_norm_bp_input_grad_kernel<DataType>,
      grid_dims,
      block_dims,
      0,
      multisync,
      global_sample_size,
      local_num_samples,
      local_sample_size,
      m_epsilon,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_output_grad.LockedBuffer(),
      local_output_grad.LDim(),
      local_input_grad.Buffer(),
      local_input_grad.LDim(),
      local_means.LockedBuffer(),
      local_means.LDim(),
      local_vars.LockedBuffer(),
      local_vars.LDim(),
      local_means_grad.LockedBuffer(),
      local_means_grad.LDim(),
      local_vars_grad.LockedBuffer(),
      local_vars_grad.LDim());
  }
}

#define ETI(T, Backend)                                                        \
  template class LayerNormalization<Backend, T>;                               \
  template void LayerNormalization<Backend, T>::calculate_forward_stats<       \
    tensor::CUDAAllocator>(                                                    \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& input,  \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& statistics);  \
  template void                                                                \
  LayerNormalization<Backend, T>::apply_normalization<tensor::CUDAAllocator>(  \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& input,  \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& statistics,   \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& output);      \
  template void LayerNormalization<Backend, T>::calculate_backward_stats<      \
    tensor::CUDAAllocator>(                                                    \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& input,  \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      output_grad,                                                             \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      statistics,                                                              \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&               \
      statistics_grad);                                                        \
  template void                                                                \
  LayerNormalization<Backend, T>::apply_grad<tensor::CUDAAllocator>(           \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& input,  \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      output_grad,                                                             \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      statistics,                                                              \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      statistics_grad,                                                         \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& input_grad);

ETI(float, BackendDNNLib)
ETI(double, BackendDNNLib)
#undef ETI
} // namespace distconv
#endif // LBANN_HAS_DISTCONV