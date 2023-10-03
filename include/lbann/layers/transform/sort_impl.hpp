////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYER_SORT_IMPL_HPP_INCLUDED
#define LBANN_LAYER_SORT_IMPL_HPP_INCLUDED

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
sort_layer<TensorDataType, T_layout, Dev>&
sort_layer<TensorDataType, T_layout, Dev>::operator=(const sort_layer& other)
{
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
void sort_layer<TensorDataType, T_layout, Dev>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();
  this->set_output_dims(this->get_input_dims());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void sort_layer<TensorDataType, T_layout, Dev>::setup_data(
  size_t max_mini_batch_size)
{
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
void sort_layer<TensorDataType, T_layout, Dev>::fp_setup_outputs(
  El::Int mini_batch_size)
{
  data_type_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);
  const auto& output = this->get_activations();
  m_indices->Resize(output.LocalHeight(), output.LocalWidth());
}

} // namespace lbann

#endif // LBANN_LAYER_SORT_IMPL_HPP_INCLUDED
