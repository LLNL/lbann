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

#define LBANN_SORT_LAYER_INSTANTIATE
#include "lbann/layers/transform/sort.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
sort_layer<TensorDataType, T_layout, Dev>::sort_layer(const sort_layer& other)
  : data_type_layer<TensorDataType>(other), m_descending(other.m_descending)
{
  if (other.m_indices) {
    switch (other.m_indices->GetDevice()) {
    case El::Device::CPU:
      m_indices.reset(new El::Matrix<El::Int, El::Device::CPU>());
      break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      m_indices.reset(new El::Matrix<El::Int, El::Device::GPU>());
      break;
#endif // LBANN_HAS_GPU
    default:
      LBANN_ERROR("invalid device");
    }
    El::Copy(*other.m_indices, *m_indices);
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
sort_layer<TensorDataType, T_layout, Dev>& sort_layer<TensorDataType, T_layout, Dev>::operator=(const sort_layer& other) {
  data_type_layer<TensorDataType>::operator=(other);
  m_descending = other.m_descending;
  if (!other.m_indices) {
    m_indices.reset(nullptr);
  }
  else {
    switch (other.m_indices->GetDevice()) {
    case El::Device::CPU:
      m_indices.reset(new El::Matrix<El::Int, El::Device::CPU>());
      break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      m_indices.reset(new El::Matrix<El::Int, El::Device::GPU>());
      break;
#endif // LBANN_HAS_GPU
    default:
      LBANN_ERROR("invalid device");
    }
    El::Copy(*other.m_indices, *m_indices);
  }
  return *this;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void sort_layer<TensorDataType, T_layout, Dev>::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  this->set_output_dims(this->get_input_dims());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void sort_layer<TensorDataType, T_layout, Dev>::setup_data(size_t max_mini_batch_size) {
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);
  const auto& dist = this->get_activations().DistData();
  switch (dist.device) {
  case El::Device::CPU:
    m_indices.reset(new El::Matrix<El::Int, El::Device::CPU>());
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    m_indices.reset(new El::Matrix<El::Int, El::Device::GPU>());
    m_indices->SetMemoryMode(0); // Allocate GPU memory with the CUDA API
    break;
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("invalid device");
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void sort_layer<TensorDataType, T_layout, Dev>::fp_setup_outputs(El::Int mini_batch_size) {
  data_type_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);
  const auto& output = this->get_activations();
  m_indices->Resize(output.LocalHeight(), output.LocalWidth());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void sort_layer<TensorDataType, T_layout, Dev>::fp_compute() {

  // Local matrices
  const auto& local_input = this->get_local_prev_activations();
  auto& local_output = this->get_local_activations();
  auto& local_indices = *this->m_indices;
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // Sort each matrix column
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    std::multimap<DataType, El::Int> sorted_list;
    for (El::Int row = 0; row < local_height; ++row) {
      sorted_list.emplace(local_input(row, col), row);
    }
    if (this->m_descending) {
      auto&& it = sorted_list.rbegin();
      for (El::Int row = 0; row < local_height; ++row, ++it) {
        local_output(row, col) = it->first;
        local_indices(row, col) = it->second;
      }
    } else {
      auto&& it = sorted_list.begin();
      for (El::Int row = 0; row < local_height; ++row, ++it) {
        local_output(row, col) = it->first;
        local_indices(row, col) = it->second;
      }
    }
  }

}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void sort_layer<TensorDataType, T_layout, Dev>::bp_compute() {

  // Local matrices
  const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
  auto& local_gradient_wrt_input = this->get_local_error_signals();
  const auto& local_indices = *this->m_indices;
  const auto& local_height = local_gradient_wrt_input.Height();
  const auto& local_width = local_gradient_wrt_input.Width();

  // Scatter gradients based on sorted indices
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& dy = local_gradient_wrt_output(row, col);
      auto& dx = local_gradient_wrt_input(local_indices(row, col), col);
      dx = dy;
    }
  }

}

#define PROTO(T)                                      \
  template class sort_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
