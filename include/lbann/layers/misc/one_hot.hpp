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

#ifndef LBANN_LAYERS_MISC_ONE_HOT_HPP_INCLUDED
#define LBANN_LAYERS_MISC_ONE_HOT_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"

#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief Convert index to a one-hot vector
 *
 *  Expects a scalar input tensor and outputs a 1D tensor. The input
 *  is interpreted as an index, and output entries are one if they
 *  correspond to that index and zero otherwise. Out-of-range indices
 *  are ignored.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class one_hot_layer : public data_type_layer<TensorDataType>
{
public:
  one_hot_layer(size_t size) : data_type_layer<TensorDataType>(nullptr)
  {
    this->set_output_dims({static_cast<int>(size)});
  }
  one_hot_layer* copy() const override { return new one_hot_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "one-hot"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  one_hot_layer() : one_hot_layer(0) {}

  void setup_dims() override;

  void fp_compute() override;
};

template <typename T, data_layout L, El::Device D>
void one_hot_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_one_hot();
  msg->set_size(this->get_output_dims()[0]);
}

#ifndef LBANN_ONE_HOT_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class one_hot_layer<T, data_layout::DATA_PARALLEL, Device>;  \
  extern template class one_hot_layer<T, data_layout::MODEL_PARALLEL, Device>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_ONE_HOT_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MISC_ONE_HOT_HPP_INCLUDED
