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

#ifndef LBANN_LAYERS_REGULARIZERS_CHANNELWISE_SOFTMAX_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_REGULARIZERS_CHANNELWISE_SOFTMAX_IMPL_HPP_INCLUDED

#include "lbann/layers/misc/channelwise_softmax.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_softmax_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  int64_t dims = static_cast<int64_t>(this->get_input_dims().size());
  if (this->m_dim < -dims || this->m_dim >= dims) {
    LBANN_ERROR("Dimension ",
                this->m_dim,
                " is out of bounds for Channelwise "
                "Softmax layer on tensor with ",
                dims,
                " dimensions.");
  }
  if (!this->m_single_dim_mode && this->m_dim != 0 && this->m_dim != -dims &&
      this->m_dim != (dims - 1) && this->m_dim != -1) {
    LBANN_ERROR("Channelwise softmax with all dimensions is only supported for "
                "the first or last tensor dimensions. Got dimension ",
                this->m_dim);
  }

  this->set_output_dims(this->get_input_dims());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_softmax_layer<TensorDataType, Layout, Device>::
  get_channel_size_and_stride(El::Int& channel_size,
                              El::Int& channel_stride,
                              El::Int& num_channels) const
{
  auto const& input_dims = this->get_input_dims();
  int dims = static_cast<int>(input_dims.size());
  int dim = this->m_dim;
  if (dim < 0) // Handle negative dimensions
    dim += dims;

  size_t total_size = 1;
  for (int i = 0; i < dims; ++i) {
    total_size *= input_dims[i];
  }

  // Definitions:
  // * Channel size: The channel size being normalized
  // * Number of channels: The number of normalized sub-tensors
  // * Channel stride: The number of elements to jump between two channels

  // Single dimension mode: size = dim size, stride = dim stride
  if (m_single_dim_mode) {
    channel_size = input_dims[dim];
    num_channels = total_size / channel_size;
    // Assuming contiguous tensors with C stride ordering
    channel_stride = 1;
    for (int i = dims - 1; i >= dim; --i) {
      channel_stride *= input_dims[i];
    }
  }
  else {
    // All other dimensions mode:
    // size = total size / dim size
    channel_size = total_size / input_dims[dim];
    num_channels = input_dims[dim];
    // -if dim = first: stride = total size / dim size (product of all other
    // dims) -if dim = last: stride = dim size
    if (dim == 0) { // First dimension
      channel_stride = channel_size;
    }
    else { // Last dimension
      channel_stride = 1;
    }
  }
}

} // namespace lbann

#endif // LBANN_LAYERS_REGULARIZERS_CHANNELWISE_SOFTMAX_IMPL_HPP_INCLUDED
