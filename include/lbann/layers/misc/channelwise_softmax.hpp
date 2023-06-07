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

#ifndef LBANN_LAYERS_REGULARIZERS_CHANNELWISE_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYERS_REGULARIZERS_CHANNELWISE_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"

#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief Apply softmax to tensor channels.
 *
 *  The input tensor is sliced along the first tensor dimension (the
 *  "channel" dimension for image data in CHW format) and the softmax
 *  function is applied to each slice:
 *  @f[ \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} @f]
 *
 *  This is not to be confused with @c softmax_mode::CHANNEL for
 *  @c softmax_layer, which applies the softmax function to entries
 *  corresponding to the same spatial position. "Channel mode" softmax
 *  might be described as "position-wise softmax".
 *
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class channelwise_softmax_layer : public data_type_layer<TensorDataType>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "channelwise_softmax_layer only supports "
                "data-parallel data layout");

public:
  channelwise_softmax_layer(lbann_comm* comm,
                            int64_t dim,
                            bool single_dim_mode);

  channelwise_softmax_layer(const channelwise_softmax_layer& other) = default;
  channelwise_softmax_layer&
  operator=(const channelwise_softmax_layer& other) = default;
  channelwise_softmax_layer* copy() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | ACTIVATIONS;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  channelwise_softmax_layer() : channelwise_softmax_layer(nullptr, 0, false) {}

  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void fp_compute() override;
  void bp_compute() override;

private:
  void get_channel_size_and_stride(El::Int& channel_size,
                                   El::Int& channel_stride,
                                   El::Int& num_channels) const;

  /** Specifies the dimension of the tensor to perform softmax on. */
  int64_t m_dim;

  /** @brief If true, only performs softmax on the chosen dimension. Otherwise
             all dimensions but ``m_dim`` will be used. */
  bool m_single_dim_mode;
};

// Builder function

// =========================================================
// Implementation
// =========================================================

template <typename T, data_layout L, El::Device D>
void channelwise_softmax_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_channelwise_softmax();
  msg->set_dim(m_dim);
  msg->set_single_dim_mode(m_single_dim_mode);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
channelwise_softmax_layer<TensorDataType, Layout, Device>::
  channelwise_softmax_layer(lbann_comm* comm, int64_t dim, bool single_dim_mode)
  : data_type_layer<TensorDataType>(comm),
    m_dim(dim),
    m_single_dim_mode(single_dim_mode)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
channelwise_softmax_layer<TensorDataType, Layout, Device>*
channelwise_softmax_layer<TensorDataType, Layout, Device>::copy() const
{
  return new channelwise_softmax_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string
channelwise_softmax_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "channel-wise softmax";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
channelwise_softmax_layer<TensorDataType, Layout, Device>::get_data_layout()
  const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device channelwise_softmax_layer<TensorDataType, Layout, Device>::
  get_device_allocation() const
{
  return Device;
}

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

// =========================================================
// Explicit template instantiation
// =========================================================

#ifndef LBANN_CHANNELWISE_SOFTMAX_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class channelwise_softmax_layer<T,                           \
                                                  data_layout::DATA_PARALLEL,  \
                                                  Device>;
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_CHANNELWISE_SOFTMAX_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_REGULARIZERS_CHANNELWISE_SOFTMAX_HPP_INCLUDED
