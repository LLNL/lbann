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
void OperatorLayer<T, O, L, D>::fill_onnx_node(onnx::GraphProto& graph) const
{
  const auto& parents = this->get_parent_layers();
  auto nodes = m_ops.front()->get_onnx_nodes();

  auto* op_node = graph.add_node();
  *op_node = nodes.front();

  op_node->set_name(this->get_name());
  op_node->set_domain("");
  op_node->set_doc_string(this->get_name());

  // binary operators
  if (nodes.size() == 1) {
    for (auto* parent : parents) {
      size_t idx = parent->find_child_layer_index(*this);
      op_node->add_input(parent->get_name() + "_" + std::to_string(idx));
    }
  }
  // Binary w/ constant operators
  else if (nodes.size() == 2 || nodes.size() == 3) {
    auto* const_node = graph.add_node();
    *const_node = nodes.back();
    if (const_node->op_type() == "PostConstant") {
      op_node->add_input(parents[0]->get_name() + "_0");
      op_node->add_input(const_node->output(0));
    }
    else if (const_node->op_type() == "PreConstant") {
      op_node->add_input(const_node->output(0));
      op_node->add_input(parents[0]->get_name() + "_0");
    }
    else
      LBANN_ERROR("Unknown onnx op type for constant.");

    const_node->set_op_type("Constant");
  }
  else
    LBANN_ERROR("Expected 1-3 ONNX nodes for binary operation, received ",
                nodes.size());

  // Not equal operator
  if (nodes.size() == 3) {
    op_node->add_output("EqualOperator");
    auto* not_node = graph.add_node();
    not_node->add_input(op_node->output(0));
    not_node->set_name("Not operator");
    not_node->set_op_type("Not");
    not_node->set_domain("");
    not_node->set_doc_string("Not node for not equal operation.");
    op_node = not_node;
  }

  for (auto const* child : this->get_child_layers()) {
    auto idx = this->find_child_layer_index(*child);
    op_node->add_output(this->get_name() + "_" + std::to_string(idx));
  }
}
#endif // LBANN_HAS_ONNX

} // namespace lbann
