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

#define LBANN_INSTANTIATE_OPERATOR_LAYER
#include "lbann/layers/operator_layer_impl.hpp"
#include "lbann/proto/datatype_helpers.hpp"

namespace lbann {

#define PROTO_DEVICE(T, D)                                                     \
  template class OperatorLayer<T, T, data_layout::DATA_PARALLEL, D>;           \
  template class OperatorLayer<T, T, data_layout::MODEL_PARALLEL, D>;          \
  template std::unique_ptr<Layer>                                              \
  build_operator_layer_from_pbuf<T, T, data_layout::DATA_PARALLEL, D>(         \
    lbann_comm*,                                                               \
    lbann_data::Layer const&);                                                 \
  template std::unique_ptr<Layer>                                              \
  build_operator_layer_from_pbuf<T, T, data_layout::MODEL_PARALLEL, D>(        \
    lbann_comm*,                                                               \
    lbann_data::Layer const&)

#include "lbann/macros/instantiate_device.hpp"

template <typename T, typename O, data_layout L, El::Device D>
void OperatorLayer<T, O, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{

  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_operator_layer();
  auto* op = msg->add_ops();
  op->set_input_datatype(proto::ProtoDataType<T>);
  op->set_output_datatype(proto::ProtoDataType<O>);
  op->set_device_allocation(proto::ProtoDevice<D>);
}

#ifdef LBANN_HAS_ONNX
template <typename T, typename O, data_layout L, El::Device D>
void OperatorLayer<T, O, L, D>::fill_onnx_node(
  onnx::GraphProto& graph) const
{
    std::vector<onnx::NodeProto> nodes(2UL);
    nodes.front().add_attribute()->set_type(onnx::AttributeProto::FLOAT);
    nodes.front().add_attribute()->set_f(El::To<float>(5));
    nodes.front().set_op_type("PostConstant");
    nodes.back().set_op_type("Add");

  //OperatorPtr op;
  //auto nodes = op->get_onnx_nodes();
  const auto* parent = this->get_parent_layers()[0];

  auto* const_node = graph.add_node();
  *const_node = nodes.front();

  auto* node = graph.add_node();
  *node = nodes.back();
  node->set_name(this->get_name());
  node->set_domain("");
  node->set_doc_string(this->get_name());
  if(const_node->op_type() == "PostConstant")
  {
    node->add_input(parent->get_name() + "_0");
    node->add_input(const_node->output(0));
    const_node->set_op_type("Constant");
  }
  else if(const_node->op_type() == "PreConstant")
  {
    node->add_input(const_node->output(0));
    node->add_input(parent->get_name() + "_0");
    const_node->set_op_type("Constant");
  }
  else
    LBANN_ERROR("Unknown onnx op type for constant.");

  // Not equal operator
  if(nodes.size() == 3)
  {
    node->add_output("EqualOperator");
    auto* not_node = graph.add_node();
    not_node->add_input(node->output(0));
    not_node->add_output(this->get_child_layers()[0]->get_name() + "_0");
    not_node->set_name("Not operator");
    not_node->set_op_type("Not");
    not_node->set_domain("");
    not_node->set_doc_string("Not node for not equal operation.");
  }
  else if(nodes.size() == 2)
  {
    node->add_output(this->get_child_layers()[0]->get_name() + "_0");
  }
  else
    LBANN_ERROR("Expected two or three nodes for binary constant operation, received ", nodes.size());
}
#endif // LBANN_HAS_ONNX

} // namespace lbann
