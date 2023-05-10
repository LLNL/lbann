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

#ifndef LBANN_LAYERS_MISC_CHANNELWISE_MEAN_HPP_INCLUDED
#define LBANN_LAYERS_MISC_CHANNELWISE_MEAN_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @todo Replace with more general reduction layer. */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class channelwise_mean_layer : public data_type_layer<TensorDataType>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "channelwise_mean_layer only supports "
                "data-parallel data layout");

public:
  channelwise_mean_layer(lbann_comm* = nullptr)
    : data_type_layer<TensorDataType>(nullptr)
  {}

  channelwise_mean_layer* copy() const override
  {
    return new channelwise_mean_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "channel-wise mean"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void setup_dims(DataReaderMetaData& dr_metadata) override
  {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    const auto& input_dims = this->get_input_dims();
    this->set_output_dims({input_dims[0]});
  }

  void fp_compute() override;
  void bp_compute() override;
};

template <typename T, data_layout L, El::Device D>
void channelwise_mean_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_channelwise_mean();
}

#ifndef LBANN_CHANNELWISE_MEAN_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class channelwise_mean_layer<T,                              \
                                               data_layout::DATA_PARALLEL,     \
                                               Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_CHANNELWISE_MEAN_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MISC_CHANNELWISE_MEAN_HPP_INCLUDED
