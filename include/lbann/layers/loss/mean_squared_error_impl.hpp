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

#ifndef LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_IMPL_HPP_INCLUDED

#include "lbann/layers/loss/mean_squared_error.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {
template <typename TensorDataType, data_layout Layout, El::Device Device>
void mean_squared_error_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();
  this->set_output_dims({1});

#ifdef LBANN_HAS_DISTCONV
  // In the current implementation of mean squared error in Distconv, we
  // do not use the reshape layer and just assumes both inputs have
  // the matching shape. Therefore, the following check on the input
  // dimensions would fail. We could address this by either 1)
  // implementing the reshape layer, or 2) giving a proper shape to
  // the ground-truth data.
  //
  if (this->distconv_enabled()) {
    return;
  }
#endif

  // Check that input dimensions match
  if (this->get_input_dims(0) != this->get_input_dims(1)) {
    const auto& parents = this->get_parent_layers();
    std::stringstream err;
    err << get_type() << " layer \"" << this->get_name() << "\" "
        << "has input tensors with different dimensions (";
    for (int i = 0; i < this->get_num_parents(); ++i) {
      const auto& dims = this->get_input_dims(i);
      err << (i > 0 ? ", " : "") << "layer \"" << parents[i]->get_name()
          << "\" outputs ";
      for (size_t j = 0; j < dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << dims[j];
      }
    }
    err << ")";
    LBANN_ERROR(err.str());
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void mean_squared_error_layer<TensorDataType, Layout, Device>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Initialize workspace
  const auto& input_dist = this->get_prev_activations(0).DistData();
  m_workspace.reset(AbsDistMatrixType::Instantiate(
    *input_dist.grid,
    input_dist.root,
    El::STAR,
    input_dist.rowDist,
    (input_dist.blockHeight == 1 && input_dist.blockWidth == 1 ? El::ELEMENT
                                                               : El::BLOCK),
    input_dist.device));
#ifdef HYDROGEN_HAVE_CUB
  if (m_workspace->GetLocalDevice() == El::Device::GPU) {
    m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
  }
#endif // HYDROGEN_HAVE_CUB
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void mean_squared_error_layer<TensorDataType, Layout, Device>::fp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    fp_compute_distconv();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  // Initialize workspace
  m_workspace->Empty();
  m_workspace->AlignWith(this->get_prev_activations());
  m_workspace->Resize(1, this->get_prev_activations().Width());

  // Compute local contributions and accumulate
  /// @todo Consider reduce rather than allreduce
  local_fp_compute();
  this->get_comm()->allreduce(*m_workspace, m_workspace->RedundantComm());
  El::Copy(*m_workspace, this->get_activations());

  // Clean up
  m_workspace->Empty();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void mean_squared_error_layer<TensorDataType, Layout, Device>::bp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    bp_compute_distconv();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  // Initialize workspace
  m_workspace->Empty();
  m_workspace->AlignWith(this->get_prev_activations());
  El::Copy(this->get_prev_error_signals(), *m_workspace);

  // Compute local gradients
  local_bp_compute();

  // Clean up
  m_workspace->Empty();
}

template <typename T, data_layout L, El::Device D>
void mean_squared_error_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_mean_squared_error();
}

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void mean_squared_error_layer<T, L, D>::fill_onnx_node(
  onnx::GraphProto& graph) const
{
  // def mean_squared_error(x, y):
  // z = x - y
  auto* sub = graph.add_node();
  for (auto const* parent : this->get_parent_layers()) {
    size_t idx = parent->find_child_layer_index(*this);
    sub->add_input(parent->get_name() + "_" + std::to_string(idx));
  }
  sub->add_output(this->get_name() + "_sub");
  sub->set_name(this->get_name() + "_sub");
  sub->set_op_type("Sub");
  sub->set_domain("");
  sub->set_doc_string("Sub node for Mean Squared Error Layer");

  // z = z * z
  // FIXME: Use Pow instead of Sub?
  auto* square = graph.add_node();
  square->add_input(sub->output(0));
  square->add_input(sub->output(0));
  square->add_output(this->get_name() + "_square");
  square->set_name(this->get_name() + "_square");
  square->set_op_type("Mul");
  square->set_domain("");
  square->set_doc_string("Square node for Mean Squared Error Layer");

  // z = Reshape(data=z, shape=[0,-1])
  auto* shape = graph.add_initializer();
  shape->set_name(this->get_name() + "_square_shape");
  shape->set_data_type(onnx::TensorProto::INT64);
  shape->add_dims(2);
  shape->add_int64_data(0);
  shape->add_int64_data(-1);
  shape->set_doc_string(this->get_name() + "shape to reshape square");

  auto* reshape = graph.add_node();
  reshape->add_input(square->output(0));
  reshape->add_input(shape->name());
  reshape->add_output(this->get_name() + "_square_reshape");
  reshape->set_name(this->get_name() + "_square_reshape");
  reshape->set_op_type("Reshape");
  reshape->set_domain("");
  reshape->set_doc_string("Reshape square for Mean Squared Error Layer");

  // z = ReduceMean(data=z, axes=-1)
  // return z
  auto* reduce_mean = graph.add_node();
  auto* attribute = reduce_mean->add_attribute();
  attribute->set_name("axes");
  attribute->set_type(onnx::AttributeProto::INTS);
  attribute->add_ints(-1);
  reduce_mean->add_input(square->output(0));
  for (auto const* child : this->get_child_layers()) {
    size_t idx = this->find_child_layer_index(*child);
    reduce_mean->add_output(this->get_name() + "_" + std::to_string(idx));
  }
  reduce_mean->set_name(this->get_name() + "_reducemean");
  reduce_mean->set_op_type("ReduceMean");
  reduce_mean->set_domain("");
  reduce_mean->set_doc_string("ReduceMean node for Mean Squared Error Layer");
}
#endif // LBANN_HAS_ONNX

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>&
mean_squared_error_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
  const
{
  return dynamic_cast<
    const mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>&
mean_squared_error_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<
    mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<const mean_squared_error_layer<TensorDataType, T_layout, Dev>&>(
      *this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::
  get_prev_activations_shape(int index) const
{
  // Assumes both of the two input tensors have the equal shape.
  return data_type_distconv_adapter<TensorDataType>::get_prev_activations_shape(
    0);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::
  get_activations_shape(int output_index) const
{
  // NOTE: LBANN matrix is a 2-D matrix, while Distconv keeps the
  // original spatial and channel dimensions, so
  // get_output_tensor_shape() doesn't work here.
  dc::Shape shape = this->get_prev_activations_shape(0);
  for (int i = 0; i < shape.num_dims() - 1; ++i) {
    shape[i] = 1;
  }
  return shape;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::
  get_activations_local_shape(int index) const
{
  assert_eq(index, 0);
  auto input_shape = this->get_prev_activations().get_local_shape();
  for (int i = 0; i < input_shape.length() - 1; ++i) {
    input_shape[i] = 1;
  }
  return input_shape;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_distributions(tensor_overlap_constraints& constraints)
{
  data_type_distconv_adapter<TensorDataType>::setup_distributions(constraints);
  // Output tensors share all dimensions except for the sample dimension
  auto activations_split = this->get_activations_dist().get_split_shape();
  auto prev_error_signals_split =
    this->get_prev_error_signals_dist().get_split_shape();
  for (int i = 0; i < activations_split.length() - 1; ++i) {
    activations_split[i] = 1;
    prev_error_signals_split[i] = 1;
  }
  this->get_activations_dist().set_split_shape(activations_split);
  this->get_prev_error_signals_dist().set_split_shape(prev_error_signals_split);

  for (auto& d : this->m_prev_activations_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto& d : this->m_activations_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto& d : this->m_prev_error_signals_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto& d : this->m_error_signals_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_layer(size_t workspace_capacity)
{
  m_mean_squared_error =
    std::make_unique<dc::MeanSquaredError>(dc::get_backend());
  m_mean_squared_error->setup(this->get_prev_activations(0),
                              this->get_prev_activations(1),
                              this->get_activations(0));
}
#endif // LBANN_HAS_DISTCONV

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_IMPL_HPP_INCLUDED
