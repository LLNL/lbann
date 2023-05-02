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

#ifndef LBANN_LAYERS_LOSS_TOP_K_CATEGORICAL_ACCURACY_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_TOP_K_CATEGORICAL_ACCURACY_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

/**
 *  Requires two inputs, which are respectively interpreted as
 *  prediction scores and as a one-hot label vector. The output is one
 *  if the corresponding label matches one of the top-k prediction
 *  scores and is otherwise zero. Ties in the top-k prediction scores
 *  are broken in favor of entries with smaller indices.
 *
 *  @todo Gracefully handle case where label is not a one-hot vector.
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class top_k_categorical_accuracy_layer : public data_type_layer<TensorDataType>
{
public:
  top_k_categorical_accuracy_layer(lbann_comm* comm, El::Int k)
    : data_type_layer<TensorDataType>(comm), m_k(k)
  {
    this->m_expected_num_parent_layers = 2;
  }

  top_k_categorical_accuracy_layer* copy() const override
  {
    return new top_k_categorical_accuracy_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "top-k accuracy"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }

  description get_description() const override
  {
    auto desc = data_type_layer<TensorDataType>::get_description();
    desc.add("k", m_k);
    return desc;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  top_k_categorical_accuracy_layer()
    : top_k_categorical_accuracy_layer(nullptr, 1)
  {}

  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void fp_compute() override;

private:
  /** Parameter for top-k search. */
  /*const*/ El::Int m_k;
};

template <typename T, data_layout L, El::Device D>
void top_k_categorical_accuracy_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_top_k_categorical_accuracy();
  msg->set_k(m_k);
}

#ifndef LBANN_TOP_K_CATEGORICAL_ACCURACY_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class top_k_categorical_accuracy_layer<                      \
    T,                                                                         \
    data_layout::DATA_PARALLEL,                                                \
    Device>;                                                                   \
  extern template class top_k_categorical_accuracy_layer<                      \
    T,                                                                         \
    data_layout::MODEL_PARALLEL,                                               \
    Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_TOP_K_CATEGORICAL_ACCURACY_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_TOP_K_CATEGORICAL_ACCURACY_HPP_INCLUDED
