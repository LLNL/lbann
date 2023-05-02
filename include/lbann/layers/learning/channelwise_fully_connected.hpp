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

#ifndef LBANN_LAYERS_LEARNING_CHANNELWISE_FULLY_CONNECTED_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_CHANNELWISE_FULLY_CONNECTED_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#include "lbann/layers/learning/distconv/distconv_layers.hpp"
#endif
namespace lbann {

#ifdef LBANN_HAS_DISTCONV
namespace dc {
template <typename TensorDataType>
using ChannelwiseFullyConnected =
  ::distconv::ChannelwiseFullyConnected<Backend, TensorDataType>;
} // namespace dc

template <typename TensorDataType, data_layout Layout, El::Device Device>
class channelwise_fully_connected_distconv_adapter
  : public data_type_distconv_adapter<TensorDataType>
{

public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;

  channelwise_fully_connected_distconv_adapter(Layer& layer)
    : data_type_distconv_adapter<TensorDataType>(layer)
  {}
  virtual ~channelwise_fully_connected_distconv_adapter() = default;

  void setup_fp_tensors() override;
  void setup_bp_tensors() override;
  void setup_distributions(tensor_overlap_constraints& constraints) override;
  void setup_layer(size_t workspace_capacity) override;

  void fp_compute();
  void bp_compute();

  dc::Shape get_activations_local_shape(int index = 0) const override;

  std::unique_ptr<dc::ChannelwiseFullyConnected<TensorDataType>>
    m_linear_operator;
  std::unique_ptr<TensorDevType> m_linear;
  std::unique_ptr<TensorDevType> m_bias;
  std::unique_ptr<TensorDevType> m_linearity_gradient;
  std::unique_ptr<TensorDevType> m_bias_gradient;
}; // class definition channelwise_fully_connected_distconv_adapter

#endif // LBANN_HAS_DISTCONV

/** @brief Apply affine transformation to tensor channels.
 *
 *  The input tensor is sliced along the first tensor dimension (the
 *  "channel" dimension for image data in CHW format) and the same
 *  affine transformation is applied to each slice. Following a
 *  row-vector convention:
 *    @f[ y(i,*) = \text{vec}( x(i,*) ) W^T + b @f]
 *
 *  Two weights are required if bias is applied: the linearity and the
 *  bias. Only the linearity weights are required if bias is not
 *  applied. If weights aren't provided, the linearity weights are
 *  initialized with He normal initialization and the bias weights are
 *  initialized to zero.
 *
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class channelwise_fully_connected_layer : public data_type_layer<TensorDataType>
{

  static_assert(Layout == data_layout::DATA_PARALLEL,
                "channelwise_fully_connected layer "
                "only supports data parallel layout");

public:
  /** @brief Constructor.
   *  @param output_channel_dims    Output tensor dimensions,
   *                                excluding the first dimension.
   *  @param bias                   Whether to apply bias.
   *  @param transpose              Whether to apply transpose of
   *                                weights matrix.
   */
  channelwise_fully_connected_layer(std::vector<size_t> output_channel_dims,
                                    bool bias,
                                    bool transpose);

  channelwise_fully_connected_layer(
    const channelwise_fully_connected_layer& other) = default;
  channelwise_fully_connected_layer&
  operator=(const channelwise_fully_connected_layer& other) = default;
  ~channelwise_fully_connected_layer() = default;

  channelwise_fully_connected_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | WEIGHTS | PREV_ACTIVATIONS;
  }

  description get_description() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  channelwise_fully_connected_layer();

  void setup_dims(DataReaderMetaData& dr_metadata) override;
  void setup_data(size_t max_mini_batch_size) override;

  void fp_compute() override;
  void bp_compute() override;
  std::vector<int> get_linearity_dims() const;
  std::vector<int> get_bias_dims() const;

#ifdef LBANN_HAS_DISTCONV
  friend class channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                            Layout,
                                                            Device>;

protected:
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override;
  bool is_distconv_supported() const override;
  channelwise_fully_connected_distconv_adapter<TensorDataType, Layout, Device>&
  get_distconv_adapter() override;
  const channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                     Layout,
                                                     Device>&
  get_distconv_adapter() const override;
#endif

private:
  /** Whether to apply bias. */
  bool m_has_bias;
  /** Whether to transpose linearity. */
  bool m_transpose;
};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(channelwise_fully_connected);

// Explicit template instantiation
#ifndef LBANN_CHANNELWISE_FULLY_CONNECTED_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class channelwise_fully_connected_layer<                     \
    T,                                                                         \
    data_layout::DATA_PARALLEL,                                                \
    Device>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_CHANNELWISE_FULLY_CONNECTED_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_CHANNELWISE_FULLY_CONNECTED_HPP_INCLUDED
