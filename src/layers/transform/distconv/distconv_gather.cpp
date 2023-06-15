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
#define LBANN_LAYERS_TRANSFORM_DISTCONV_GATHER_INSTANTIATE
#include "lbann/layers/transform/distconv/distconv_gather.hpp"
#include "lbann/base.hpp"
#include "lbann/utils/distconv.hpp"

namespace distconv {
template <typename Backend, typename DataType>
template <typename Allocator>
int Gather<Backend, DataType>::forward(
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& values,
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& indices,
  tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& output)
{

  if (output.get_buffer() == nullptr) {
    util::MPIPrintStreamDebug()
      << "output buffer is null in distconv Gather layer";
    return 0;
  }

  if (values.get_buffer() == nullptr) {
    util::MPIPrintStreamDebug()
      << "values buffer is null in distconv Gather layer";
    return 0;
  }

  if (indices.get_buffer() == nullptr) {
    util::MPIPrintStreamDebug()
      << "indices buffer is null in distconv Gather layer";
    return 0;
  }

  const auto& values_shape = values.get_local_shape(); // Should be {1, F, N, B}
  const auto& indices_shape =
    indices.get_local_shape();                         // Should be {1, 1, E, B}
  const auto& output_shape = output.get_local_shape(); // Should be {1, F, E, B}

  const auto& num_columns = values_shape[1];
  const auto& num_values_rows = values_shape[2];
  const auto& local_mini_batch_size = values_shape[3];
  const auto& num_output_rows = output_shape[2];

  m_dist_gather->gather(values.get_buffer(),
                        indices.get_buffer(),
                        output.get_buffer(),
                        local_mini_batch_size,
                        num_values_rows,
                        num_columns,
                        num_output_rows);

  return 1;
}

template <typename Backend, typename DataType>
template <typename Allocator>
int Gather<Backend, DataType>::backward(
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& output_grad,
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& indices,
  tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& values_grad,
  tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& indices_grad)
{

  const auto& output_grad_shape =
    output_grad.get_local_shape(); // Should be {1, F, B, B}
  const auto& indices_shape =
    indices.get_local_shape(); // Should be {1, 1, E, B}
  const auto& values_grad_shape =
    values_grad.get_local_shape(); // Should be {1, F, E, B}

  const auto num_columns = output_grad_shape[1];           // F
  const auto num_output_grad_rows = output_grad_shape[2];  // E
  const auto local_mini_batch_size = output_grad_shape[3]; // B
  const auto num_values_grad_rows = values_grad_shape[2];  // N

  if (output_grad.get_buffer() == nullptr) {
    util::MPIPrintStreamDebug()
      << "output grad buffer is null in distconv Gather layer";
    return 0;
  }

  if (indices.get_buffer() == nullptr) {
    util::MPIPrintStreamDebug()
      << "indices buffer is null in distconv Gather layer";
    return 0;
  }

  if (values_grad.get_buffer() == nullptr) {
    util::MPIPrintStreamDebug()
      << "values grad buffer is null in distconv Gather layer";
    return 0;
  }

  m_dist_scatter->scatter(output_grad.get_buffer(),
                          indices.get_buffer(),
                          values_grad.get_buffer(),
                          local_mini_batch_size,
                          num_output_grad_rows,
                          num_columns,
                          num_values_grad_rows);

  const auto& zero = El::TypeTraits<DataType>::Zero();

  // Explicitly zero the indices grad matrix
  El::Matrix<DataType, El::Device::GPU> ind_grad_mat(num_values_grad_rows,
                                                     local_mini_batch_size,
                                                     indices_grad.get_buffer(),
                                                     num_values_grad_rows);
  El::Fill(ind_grad_mat, zero);
  return 1;
}

template <typename Backend, typename DataType>
template <typename Allocator>
void Gather<Backend, DataType>::setup(
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& values,
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& indices,
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& output)
{

  const auto channel_splits = values.get_distribution().get_split_shape()[2];
  const auto sample_splits = values.get_distribution().get_split_shape()[3];
  const auto sample_size = values.get_shape()[3];
  const auto input_channel_size = values.get_shape()[2];
  const auto output_channel_size = output.get_shape()[2];
  const auto feature_dim_size = values.get_shape()[0] * values.get_shape()[1];

  const auto max_samples_per_rank = static_cast<int>(
    std::ceil(static_cast<float>(sample_size) / sample_splits));
  const auto max_input_channels_per_rank = static_cast<int>(
    std::ceil(static_cast<float>(input_channel_size) / channel_splits));
  const auto max_output_channels_per_rank = static_cast<int>(
    std::ceil(static_cast<float>(output_channel_size) / channel_splits));

  const auto input_ws_size =
    max_samples_per_rank * max_input_channels_per_rank * feature_dim_size;
  const auto output_ws_size =
    max_samples_per_rank * max_output_channels_per_rank * feature_dim_size;

  util::MPIPrintStreamDebug()
    << " Sample dim size: " << sample_size
    << "\n Input channel size: " << input_channel_size
    << "\n Output channel size: " << output_channel_size
    << "\n Max samples / rank: " << max_samples_per_rank
    << "\n Max input channels / rank: " << max_input_channels_per_rank
    << "\n Max output channels / rank: " << max_output_channels_per_rank
    << "\n Input buffer size: " << input_ws_size
    << "\n Output buffer size: " << output_ws_size;

  const auto num_pes = m_dist_scatter->get_num_ranks();
  const auto pid = m_dist_scatter->get_rank();

  m_dist_scatter->ensure_buffer(input_ws_size);
  m_dist_gather->ensure_buffer(output_ws_size);
  // Check if in hybrid data-parallel channel-parallel mode
  if ((int)channel_splits == num_pes) {
    // Default setup is sufficent. No further changes needed
    return;
  }
  // hybrid data-parallel channel-parallel mode. Must set scatter / gather
  // stides and groups

  const auto num_groups = num_pes / channel_splits;
  const auto group = pid / num_groups;
  m_dist_scatter->set_stride(channel_splits);
  m_dist_gather->set_stride(channel_splits);
  m_dist_scatter->set_group(group);
  m_dist_gather->set_group(group);
}

// Explicit template instantiation

#define ETI(T, Backend)                                                        \
  template class Gather<Backend, T>;                                           \
  template void Gather<Backend, T>::setup<tensor::CUDAAllocator>(              \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& values, \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      indices,                                                                 \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      output);                                                                 \
  template int Gather<Backend, T>::forward<tensor::CUDAAllocator>(             \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& values, \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      indices,                                                                 \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& output);      \
  template int Gather<Backend, T>::backward<tensor::CUDAAllocator>(            \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      output_grad,                                                             \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      indices,                                                                 \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& values_grad,  \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&               \
      indices_grad);

ETI(float, ::distconv::BackendDNNLib)
ETI(double, ::distconv::BackendDNNLib)
#undef ETI
} // namespace distconv
