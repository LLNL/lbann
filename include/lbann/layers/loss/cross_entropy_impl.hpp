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

#ifndef LBANN_LAYERS_LOSS_CROSS_ENTROPY_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_CROSS_ENTROPY_IMPL_HPP_INCLUDED

#include "lbann/layers/loss/cross_entropy.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void cross_entropy_layer<TensorDataType, T_layout, Dev>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();
  this->set_output_dims({1});

#ifdef LBANN_HAS_DISTCONV
  // In the current implementation of cross entropy in Distconv, we
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

  if (m_use_labels) {
    const auto& parents = this->get_parent_layers();

    if (T_layout == data_layout::MODEL_PARALLEL) {
      std::stringstream err;
      err
        << get_type() << " layer \"" << this->get_name() << "\" "
        << "only supports use_labels is not supported in model parallel layout"
        << " (for now)";
      LBANN_ERROR(err.str());
    }

    const auto& predictions_dims = this->get_input_dims(0);
    const auto& labels_dims = this->get_input_dims(1);
    // Check if the first dimension is 1 for the labels tensor
    if (labels_dims[0] != 1) {
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "expects the 0-th dimension of the tensor to be 1 when use labels "
          << "is enabled. Found tensor with shape (";

      // TODO: Put this loop in util as it's used frequently to
      // print layer dimensions
      for (size_t j = 0; j < labels_dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << labels_dims[j];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }

    // Check if the number of dimensions match for predictions and labels
    // tensors

    if (predictions_dims.size() != labels_dims.size() ||
        predictions_dims.size() < 2) {
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "expects both input tensors to have the same number of dimensions "
          << "and have >2 dimensions when use_labels is enabled. "
          << "Found tensors with shape (";

      // TODO: Put this loop in util as it's used frequently to
      // print layer dimensions
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
    // Check if all spatial dimensions match for predictions and labels
    // tensors
    if (!std::equal(predictions_dims.begin() + 1,
                    predictions_dims.end(),
                    labels_dims.begin() + 1)) {
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "expects both input tensors to have the same shape after the 0-th "
          << "dimesion when use_labels is enabled. Found tensors with shape (";

      // TODO: Put this loop in util as it's used frequently to
      // print layer dimensions
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
  else {
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
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void cross_entropy_layer<TensorDataType, T_layout, Dev>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Initialize workspace
  const auto& prediction = this->get_prev_activations(0);
  switch (this->get_data_layout()) {
  case data_layout::DATA_PARALLEL:
    m_workspace.reset(new StarVCMatDT<TensorDataType, Dev>(prediction.Grid(),
                                                           prediction.Root()));
    break;
  case data_layout::MODEL_PARALLEL:
    m_workspace.reset(new StarMRMatDT<TensorDataType, Dev>(prediction.Grid(),
                                                           prediction.Root()));
    break;
  default:
    LBANN_ERROR("invalid data layout");
  }
#ifdef HYDROGEN_HAVE_CUB
  if (m_workspace->GetLocalDevice() == El::Device::GPU) {
    m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
  }
#endif // HYDROGEN_HAVE_CUB
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void cross_entropy_layer<TensorDataType, T_layout, Dev>::fp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    fp_compute_distconv();
    return;
  }

#endif // LBANN_HAS_DISTCONV

  // Initialize workspace
  const auto& prediction = this->get_prev_activations(0);
  m_workspace->AlignWith(prediction.DistData());
  m_workspace->Resize(1, prediction.Width());

  // Compute local contributions and accumulate
  /// @todo Consider reduce rather than allreduce
  local_fp_compute();
  this->get_comm()->allreduce(*m_workspace, m_workspace->RedundantComm());
  El::Copy(*m_workspace, this->get_activations());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void cross_entropy_layer<TensorDataType, T_layout, Dev>::bp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    bp_compute_distconv();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  // Initialize workspace
  const auto& prediction = this->get_prev_activations(0);
  m_workspace->AlignWith(prediction.DistData());
  El::Copy(this->get_prev_error_signals(), *m_workspace);

  // Compute local gradients
  local_bp_compute();
}

template <typename T, data_layout L, El::Device D>
void cross_entropy_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_cross_entropy();
  msg->set_use_labels(m_use_labels);
}

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void cross_entropy_layer<T, L, D>::fill_onnx_node(onnx::GraphProto& graph) const
{
  auto const parents = this->get_parent_layers();
  // z = Log(input=x)
  auto* log = graph.add_node();
  size_t idx = parents[0]->find_child_layer_index(*this);
  log->add_input(parents[0]->get_name() + "_" + std::to_string(idx));
  log->add_output(this->get_name() + "_log");
  log->set_name(this->get_name() + "_log");
  log->set_op_type("Log");
  log->set_domain("");
  log->set_doc_string("Log node for Cross Entropy Layer");

  // z = Mul(A=y, B=z)
  auto* mul = graph.add_node();
  idx = parents[1]->find_child_layer_index(*this);
  mul->add_input(parents[1]->get_name() + "_" + std::to_string(idx));
  mul->add_input(log->output(0));
  mul->add_output(this->get_name() + "_mul");
  mul->set_name(this->get_name() + "_mul");
  mul->set_op_type("Mul");
  mul->set_domain("");
  mul->set_doc_string("Multiply node for Cross Entropy Layer");

  // z = Reshape(data=z, shape=[0,-1])
  auto* shape = graph.add_initializer();
  shape->set_name(this->get_name() + "_mul_shape");
  shape->set_data_type(onnx::TensorProto::INT64);
  shape->add_dims(2);
  shape->add_int64_data(0);
  shape->add_int64_data(-1);
  shape->set_doc_string(this->get_name() + " shape to reshape multiply");

  auto* reshape = graph.add_node();
  reshape->add_input(mul->output(0));
  reshape->add_input(shape->name());
  reshape->add_output(this->get_name() + "_mul_reshape");
  reshape->set_name(this->get_name() + "_mul_reshape");
  reshape->set_op_type("Reshape");
  reshape->set_domain("");
  reshape->set_doc_string("Reshape muultiply result for Cross Entropy Layer");

  // z = ReduceSum(data=z, axes=-1)

  auto* axes = graph.add_initializer();
  axes->set_name(this->get_name() + "_reducesum_axes");
  axes->set_data_type(onnx::TensorProto::INT64);
  axes->add_dims(1);
  axes->add_int64_data(-1);
  axes->set_doc_string(this->get_name() + "ReduceSum axes");

  auto* reduce_sum = graph.add_node();
  reduce_sum->add_input(reshape->output(0));
  reduce_sum->add_input(axes->name());
  for (auto const* child : this->get_child_layers()) {
    idx = this->find_child_layer_index(*child);
    reduce_sum->add_output(this->get_name() + "_" + std::to_string(idx));
  }
  reduce_sum->set_name(this->get_name() + "_reducesum");
  reduce_sum->set_op_type("ReduceSum");
  reduce_sum->set_domain("");
  reduce_sum->set_doc_string("ReduceSum node for Cross Entropy Layer");
}
#endif // LBANN_HAS_ONNX

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>&
cross_entropy_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const
{
  return dynamic_cast<
    const cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>&
cross_entropy_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<
    cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<const cross_entropy_layer<TensorDataType, T_layout, Dev>&>(
      *this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>::
  get_prev_activations_shape(int index) const
{
  // Assumes both of the two input tensors have the equal shape.
  return data_type_distconv_adapter<TensorDataType>::get_prev_activations_shape(
    0);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>::
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
dc::Shape cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>::
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
void cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>::
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
void cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
  size_t workspace_capacity)
{
  m_cross_entropy =
    std::make_unique<dc::CrossEntropy>(dc::get_backend(), m_use_labels);
  m_cross_entropy->setup(this->get_prev_activations(0),
                         this->get_prev_activations(1),
                         this->get_activations(0));
}
#endif // LBANN_HAS_DISTCONV

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_CROSS_ENTROPY_IMPL_HPP_INCLUDED
