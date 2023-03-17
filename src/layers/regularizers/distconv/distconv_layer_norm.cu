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

#include "../layer_norm_kernel.cuh"
#include "lbann/layers/regularizers/distconv/distonv_layer_norm.hpp"

#ifdef LBANN_HAS_DISTCONV

template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization ::calculate_forward_stats(
  const DCTensor<Allocator>& input,
  DCTensor<Allocator>& statistics)
{
  if (input_0.get_local_size() == 0) {
    util::MPIRootPrintStreamInfo() << "WARNING: EMPTY INPUT FOUND \n";
    return; // no op for empty inputs
  }

  const auto& input_dims = input.get_local_shape();
  const auto& statistics_dims = statistics.get_local_shape();

  const auto local_num_samples = input_0_dims[3];

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

  LocalMat local_statistics(2,
                            local_num_samples,
                            statistics.get_local_shape(),
                            2);

  El::Zero(local_statistics);
  auto local_means = El::View(local_statistics, El::IR(0), El::ALL);
  auto local_vars = El::View(local_statistics, El::IR(1), El::ALL);

  {
    using namespace hydrogen;
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_statistics),
                                       gpu::get_sync_info(local_input));
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (local_sample_size + block_size - 1) / block_size;
    grid_dims.y = local_num_samples;
    hydrogen::gpu::LaunchKernel(
      ::lbann::layer_norm_fp_sums_kernel<block_size, TensorDataType>,
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
void LayerNormalization::apply_normalization(
  const DCTensor<Allocator>& input,
  const DCTensor<Allocator>& statistics,
  DCTensor<Allocator>& output)
{}

template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization::calculate_backward_stats(
  const DCTensor<Allocator>& input,
  const DCTensor<Allocator>& output_grad,
  const DCTensor<Allocator>& statistics,
  DCTensor<Allocator>& statistics_grad)
{}
template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization::apply_grad(const DCTensor<Allocator>& input,
                                    const DCTensor<Allocator>& output_grad,
                                    const DCTensor<Allocator>& statistics,
                                    const DCTensor<Allocator>& statistics_grad,
                                    DCTensor<Allocator>& input_grad)
{}

#define ETI(T, Backend)                                                        \
  template class LayerNormalization<Backend, T>;                               \
  template void LayerNormalization<Backend, T>::calculate_forward_stats<       \
    tensor::CUDAAllocator>(                                                    \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& input,  \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& statistics);  \
  template void                                                                \
  LayerNormalization<Backend, T>::apply_normalization<tensor::CUDAAllocator>(  \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& input,  \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      statistics,                                                              \
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
#endef ETI
#endif // LBANN_HAS_DISTCONV