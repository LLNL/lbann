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

#ifndef LBANN_LAYERS_LOSS_CATEGORICAL_ACCURACY_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_CATEGORICAL_ACCURACY_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief 0-1 loss function
 *
 *  Requires two inputs, which are respectively interpreted as
 *  prediction scores and as a one-hot label vector. The output is one
 *  if the top entries in both inputs are in the same position and is
 *  otherwise zero. Ties are broken in favor of entries with smaller
 *  indices.
 *
 *  This is primarily intended for use as a metric since it is not
 *  differentiable.
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class categorical_accuracy_layer : public data_type_layer<TensorDataType>
{
public:
  categorical_accuracy_layer(lbann_comm* comm)
    : data_type_layer<TensorDataType>(comm)
  {
    this->m_expected_num_parent_layers = 2;
  }

  categorical_accuracy_layer* copy() const override
  {
    return new categorical_accuracy_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "categorical accuracy"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

  void setup_dims() override;

  void fp_compute() override;

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  categorical_accuracy_layer() : categorical_accuracy_layer(nullptr) {}
};

template <typename T, data_layout L, El::Device D>
void categorical_accuracy_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_categorical_accuracy();
}

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void categorical_accuracy_layer<T, L, D>::fill_onnx_node(
  onnx::GraphProto& graph) const
{
  auto* shape = graph.add_initializer();
  shape->set_name(this->get_name() + "_shape_0");
  shape->set_data_type(onnx::TensorProto::INT64);
  shape->add_dims(2);
  shape->add_int64_data(0);
  shape->add_int64_data(-1);
  shape->set_doc_string(this->get_name() + " shape");

  auto* equal = graph.add_node();

  for (auto const* parent : this->get_parent_layers()) {
    size_t idx = parent->find_child_layer_index(*this);
    size_t prt_idx = this->find_parent_layer_index(*parent);

    // x = Reshape(data=x, shape=[0,-1])
    // y = Reshape(data=y, shape=[0,-1])
    auto* reshape = graph.add_node();
    reshape->add_input(parent->get_name() + "_" + std::to_string(idx));
    reshape->add_input(this->get_name() + "_shape_0");
    reshape->add_output(this->get_name() + "_reshape_" +
                        std::to_string(prt_idx));
    reshape->set_name(this->get_name() + "_reshape" + std::to_string(prt_idx));
    reshape->set_op_type("Reshape");
    reshape->set_domain("");
    reshape->set_doc_string("Reshape node for Categorical Accuracy Layer");

    // xmax = ArgMax(data=x, axis=-1)
    // ymax = ArgMax(data=y, axis=-1)
    auto* argmax = graph.add_node();
    argmax->add_input(reshape->output(0));
    auto* attribute = argmax->add_attribute();
    attribute->set_name("axis");
    attribute->set_type(onnx::AttributeProto::INT);
    attribute->set_i(-1);
    argmax->add_output(this->get_name() + "_argmax_" + std::to_string(prt_idx));
    argmax->set_name(this->get_name() + "_argmax_" + std::to_string(prt_idx));
    argmax->set_op_type("ArgMax");
    argmax->set_domain("");
    argmax->set_doc_string("Argmax node for Categorical Accuracy Layer");

    // z = Equal(A=xmax, B=ymax)
    equal->add_input(argmax->output(0));
  }
  equal->add_output(this->get_name() + "_equal_0");
  equal->set_name(this->get_name() + "_equal_0");
  equal->set_op_type("Equal");
  equal->set_domain("");
  equal->set_doc_string("Equal node for Categorical Accuracy Layer");

  // z = Cast(input=z, to=float)
  auto* cast = graph.add_node();
  cast->add_input(equal->output(0));
  auto* attribute = cast->add_attribute();
  attribute->set_name("to");
  attribute->set_type(onnx::AttributeProto::INT);
  attribute->set_i(onnx::TensorProto::FLOAT);
  for (auto const* child : this->get_child_layers()) {
    auto idx = this->find_child_layer_index(*child);
    cast->add_output(this->get_name() + "_" + std::to_string(idx));
  }
  cast->set_name(this->get_name() + "_cast_0");
  cast->set_op_type("Cast");
  cast->set_domain("");
  cast->set_doc_string("Cast node for Categorical Accuracy Layer");
}
#endif // LBANN_HAS_ONNX

#ifndef LBANN_CATEGORICAL_ACCURACY_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class categorical_accuracy_layer<T,                          \
                                                   data_layout::DATA_PARALLEL, \
                                                   Device>;                    \
  extern template class categorical_accuracy_layer<                            \
    T,                                                                         \
    data_layout::MODEL_PARALLEL,                                               \
    Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CATEGORICAL_ACCURACY_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_CATEGORICAL_ACCURACY_HPP_INCLUDED
