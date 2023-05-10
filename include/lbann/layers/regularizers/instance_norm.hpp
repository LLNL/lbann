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

#ifndef LBANN_LAYERS_REGULARIZERS_INSTANCE_NORM_HPP_INCLUDED
#define LBANN_LAYERS_REGULARIZERS_INSTANCE_NORM_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief Normalize over data channels
 *
 *  Each channel within a data sample is normalized to have zero mean
 *  and unit standard deviation. See:
 *
 *  Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. "Instance
 *  normalization: The missing ingredient for fast stylization." arXiv
 *  preprint arXiv:1607.08022 (2016).
 *
 *  This is equivalent to applying layer normalization independently
 *  to each channel. Note that this layer does not apply a
 *  channel-wise scale and bias. Use the channel-wise scale/bias layer
 *  to reproduce that functionality.
 *
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class instance_norm_layer : public data_type_layer<TensorDataType>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "instance norm layer only supports data parallel layout");

public:
  /**
   *  @param epsilon    Small number to avoid division by zero
   */
  instance_norm_layer(TensorDataType epsilon = El::To<TensorDataType>(1e-5));

  instance_norm_layer(const instance_norm_layer& other) = default;
  instance_norm_layer& operator=(const instance_norm_layer& other) = default;
  instance_norm_layer* copy() const override;

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  description get_description() const override;
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void fp_compute() override;
  void bp_compute() override;

private:
  /** Small number to avoid division by zero. */
  TensorDataType m_epsilon;

  /** Contains per-channel sums and sums of squares. */
  El::Matrix<TensorDataType, Device> m_workspace;
};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(instance_norm);

// =========================================================
// Implementation
// =========================================================

template <typename T, data_layout L, El::Device D>
void instance_norm_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_instance_norm();
  msg->mutable_epsilon()->set_value(m_epsilon);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
instance_norm_layer<TensorDataType, Layout, Device>::instance_norm_layer(
  TensorDataType epsilon)
  : data_type_layer<TensorDataType>(nullptr), m_epsilon{epsilon}
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
instance_norm_layer<TensorDataType, Layout, Device>*
instance_norm_layer<TensorDataType, Layout, Device>::copy() const
{
  return new instance_norm_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string
instance_norm_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "instance norm";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
instance_norm_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
instance_norm_layer<TensorDataType, Layout, Device>::get_device_allocation()
  const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
instance_norm_layer<TensorDataType, Layout, Device>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Epsilon", m_epsilon);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void instance_norm_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  this->set_output_dims(this->get_input_dims());
}

// =========================================================
// Explicit template instantiation
// =========================================================

#ifndef LBANN_INSTANCE_NORM_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class instance_norm_layer<T,                                 \
                                            data_layout::DATA_PARALLEL,        \
                                            Device>;
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_INSTANCE_NORM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_REGULARIZERS_INSTANCE_NORM_HPP_INCLUDED
