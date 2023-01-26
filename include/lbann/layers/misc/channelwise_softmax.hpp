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

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#include "lbann/layers/misc/distconv/distconv_channelwise_softmax.hpp"
#endif


namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
class channelwise_softmax_distconv_adapter
  : public data_type_distconv_adapter<TensorDataType>{
  public:
    using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType; 

    channelwise_softmax_distconv_adapter(Layer& layer)
      : data_type_distconv_adapter<TensorDataType>(layer){}
    
    virtual ~channelwise_softmax_distconv_adapter() = default;
    void setup_distributions(tensor_overlap_constraints &constraints) override;
    void setup_layer(size_t workspace_capacity) override; 
    void fp_compute();
    void bp_compute();
    std::unique_ptr<dc::ChannelwiseSoftmax<TensorDataType>> m_channelwise_softmax_operator; 
  }; // class definition channelwise_softmax_distconv_adapter 

#endif  // LBANN_HAS_DISTCONV


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

  void setup_dims() override;

  void fp_compute() override;
  void bp_compute() override;

#ifdef LBANN_HAS_DISTCONV
  friend class channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>;
  protected:
    void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override;
    bool is_distconv_supported() const override;
    channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>& get_distconv_adapter() override;
    const channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>& get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV

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
void channelwise_softmax_layer<TensorDataType,Layout,Device>::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  this->set_output_dims(this->get_input_dims());
}

#ifdef LBANN_HAS_DISTCONV

// =========================================================
// DistConv-Adapter member functions
// =========================================================
template <typename TensorDataType, data_layout Layout, El::Device Device>
void
channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>
::setup_distributions(tensor_overlap_constraints &constraints){
  data_type_distconv_adapter<TensorDataType>::setup_distributions(constraints);

  for (auto &d: this->m_prev_activations_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto &d: this->m_activations_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto &d: this->m_prev_error_signals_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto &d: this->m_error_signals_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void
channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>
::setup_layer(size_t workspace_capacity){
  data_type_distconv_adapter<TensorDataType>::setup_layer(workspace_capacity);

  m_channelwise_softmax_operator = std::make_unique<dc::ChannelwiseSoftmax<TensorDataType>>(dc::get_backend());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void
channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>
::fp_compute(){
  auto &layer = dynamic_cast<
    channelwise_softmax_layer<TensorDataType, Layout, Device>&>(this->layer());
  m_channelwise_softmax_operator->forward(this->get_prev_activations(0),
                                          this->get_activations(0));
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void
channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>
::bp_compute(){
  auto &layer = dynamic_cast<
    channelwise_softmax_layer<TensorDataType, Layout, Device>&>(this->layer());
    m_channelwise_softmax_operator->backward(this->get_prev_activations(0),
                                             this->get_prev_error_signals(),
                                             this->get_error_signals(0));
}
// =============================================================
// DistConv-enabled Channelwise-Softmax member functions
// =============================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
bool
channelwise_softmax_layer<TensorDataType, Layout, Device>
::is_distconv_supported() const {
  return Device==El::Device::GPU && Layout == data_layout::DATA_PARALLEL;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void
channelwise_softmax_layer<TensorDataType, Layout, Device>
::setup_distconv_adapter(const DataReaderMetaData& dr_metadata){
  this->get_distconv_adapter_ptr() = std::make_unique<channelwise_softmax_distconv_adapter<
    TensorDataType, Layout, Device>>(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
const channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>&
channelwise_softmax_layer<TensorDataType, Layout, Device>
::get_distconv_adapter() const{
  return dynamic_cast<const channelwise_softmax_distconv_adapter< 
    TensorDataType, Layout, Device>&>(data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>&
channelwise_softmax_layer<TensorDataType, Layout, Device>
::get_distconv_adapter(){
  return const_cast<channelwise_softmax_distconv_adapter<TensorDataType, Layout, Device>&>(
    static_cast<const channelwise_softmax_layer<TensorDataType, Layout, Device>&>(*this).get_distconv_adapter());
}


#endif //  LBANN_HAS_DISTCONV

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
