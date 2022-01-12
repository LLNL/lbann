////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class cross_entropy_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  cross_entropy_distconv_adapter(Layer& layer, bool use_labels)
      : data_type_distconv_adapter<TensorDataType>(layer),
        m_use_labels(use_labels){}
  virtual ~cross_entropy_distconv_adapter() = default;
  void setup_distributions(tensor_overlap_constraints &constraints) override;
  dc::Shape get_prev_activations_shape(int index) const override;
  dc::Shape get_activations_shape(int index) const override;
  dc::Shape get_activations_local_shape(int index) const override;
  void setup_layer(size_t workspace_capacity) override;
  std::unique_ptr<dc::CrossEntropy> m_cross_entropy;
  bool m_use_labels;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Cross entropy between probability vectors
 *
 *  Given a predicted distribution @f$y@f$ and ground truth
 *  distribution @f$\hat{y}@f$,
 *  @f[ CE(y,\hat{y}) = - \sum\limits_{i} \hat{y}_i \log y_i @f]
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class cross_entropy_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  cross_entropy_layer(lbann_comm *comm, bool use_labels)
      : data_type_layer<TensorDataType>(comm),
        m_use_labels(use_labels) {
    this->m_expected_num_parent_layers = 2;
  }

  cross_entropy_layer(const cross_entropy_layer& other)
    : data_type_layer<TensorDataType>(other), m_use_labels(other.m_use_labels) {
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
  }

  cross_entropy_layer& operator=(const cross_entropy_layer& other) {
    data_type_layer<TensorDataType>::operator=(other);
    m_use_labels = other.m_use_labels;
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
    return *this;
  }

  cross_entropy_layer* copy() const override { return new cross_entropy_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "cross entropy"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
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

    // Check that input dimensions match
    if (this->get_input_dims(0) != this->get_input_dims(1)) {
      const auto& parents = this->get_parent_layers();
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "has input tensors with different dimensions (";
      for (int i = 0; i < this->get_num_parents(); ++i) {
        const auto& dims = this->get_input_dims(i);
        err << (i > 0 ? ", " : "")
            << "layer \"" << parents[i]->get_name() << "\" outputs ";
        for (size_t j = 0; j < dims.size(); ++j) {
          err << (j > 0 ? " x " : "") << dims[j];
        }
      }
      err << ")";
      LBANN_ERROR(err.str());
    }

  }

  void setup_data(size_t max_mini_batch_size) override {
    data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

    // Initialize workspace
    const auto& prediction = this->get_prev_activations(0);
    switch (this->get_data_layout()) {
    case data_layout::DATA_PARALLEL:
      m_workspace.reset(new StarVCMatDT<TensorDataType, Dev>(
                          prediction.Grid(),
                          prediction.Root()));
      break;
    case data_layout::MODEL_PARALLEL:
      m_workspace.reset(new StarMRMatDT<TensorDataType, Dev>(
                          prediction.Grid(),
                          prediction.Root()));
      break;
    default: LBANN_ERROR("invalid data layout");
    }
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB

  }

  void fp_compute() override {

#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      fp_compute_distconv();
      return;
    } else {
      if(m_use_labels) {
        LBANN_ERROR("Cross-entropy layers without Distconv don't support use_labels.");
      }
    }
#else // LBANN_HAS_DISTCONV
    if(m_use_labels) {
      LBANN_ERROR("Cross-entropy layers without Distconv don't support use_labels.");
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

  void bp_compute() override {

#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      bp_compute_distconv();
      return;
    } else {
      if(m_use_labels) {
        LBANN_ERROR("Cross-entropy layers without Distconv don't support use_labels.");
      }
    }
#else // LBANN_HAS_DISTCONV
    if(m_use_labels) {
      LBANN_ERROR("Cross-entropy layers without Distconv don't support use_labels.");
    }
#endif // LBANN_HAS_DISTCONV

    // Initialize workspace
    const auto& prediction = this->get_prev_activations(0);
    m_workspace->AlignWith(prediction.DistData());
    El::Copy(this->get_prev_error_signals(), *m_workspace);

    // Compute local gradients
    local_bp_compute();
  }

protected:

  friend class cereal::access;
  cross_entropy_layer()
    : cross_entropy_layer(nullptr, false)
  {}

private:

  /** Compute local contributions to cross entropy loss. */
  void local_fp_compute();
  /** Compute local gradients. */
  void local_bp_compute();

  /** Use interger label tensors as ground-truth. */
  bool m_use_labels;

  /** Workspace matrix. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

#ifdef LBANN_HAS_DISTCONV
  friend class cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>;
 protected:
  bool is_distconv_supported() const override {
    return Dev == El::Device::GPU && T_layout == data_layout::DATA_PARALLEL;
  }

  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override {
    this->get_distconv_adapter_ptr() = std::make_unique<
      cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>>(*this, m_use_labels);
  }

  cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() override;
  const cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() const override;

  void fp_compute_distconv() {
    assert_always(this->distconv_enabled());
    get_distconv_adapter().m_cross_entropy->forward(this->get_distconv_adapter().get_prev_activations(0),
                                  this->get_distconv_adapter().get_prev_activations(1),
                                  this->get_distconv_adapter().get_activations());
  }

  void bp_compute_distconv() {
    assert_always(this->distconv_enabled());
    get_distconv_adapter().m_cross_entropy->backward(this->get_distconv_adapter().get_prev_activations(0),
                                   this->get_distconv_adapter().get_prev_activations(1),
                                   this->get_distconv_adapter().get_prev_error_signals(0),
                                   this->get_distconv_adapter().get_error_signals(0),
                                   this->get_distconv_adapter().get_error_signals(1));
  }
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void cross_entropy_layer<T, L, D>::fill_onnx_node(
  onnx::GraphProto& graph) const {
  auto const parents = this->get_parent_layers();
  //z = Log(input=x)
  auto* log = graph.add_node();
  size_t idx = parents[0]->find_child_layer_index(*this);
  log->add_input(parents[0]->get_name() + "_" + std::to_string(idx));
  log->add_output(this->get_name() + "_log");
  log->set_name(this->get_name() + "_log");
  log->set_op_type("Log");
  log->set_domain("");
  log->set_doc_string("Log node for Cross Entropy Layer");

  //z = Mul(A=y, B=z)
  auto* mul = graph.add_node();
  idx = parents[1]->find_child_layer_index(*this);
  mul->add_input(parents[1]->get_name() + "_" + std::to_string(idx));
  mul->add_input(log->output(0));
  mul->add_output(this->get_name() + "_mul");
  mul->set_name(this->get_name() + "_mul");
  mul->set_op_type("Mul");
  mul->set_domain("");
  mul->set_doc_string("Multiply node for Cross Entropy Layer");

  //z = Reshape(data=z, shape=[0,-1])
  auto* shape = graph.add_initializer();
  shape->set_name(this->get_name() + "_mul_shape");
  shape->set_data_type(onnx::TensorProto::INT64);
  shape->add_dims(2);
  shape->add_int64_data(0);
  shape->add_int64_data(-1);
  shape->set_doc_string(this->get_name() +
                        " shape to reshape multiply");

  auto* reshape = graph.add_node();
  reshape->add_input(mul->output(0));
  reshape->add_input(shape->name());
  reshape->add_output(this->get_name() + "_mul_reshape");
  reshape->set_name(this->get_name() + "_mul_reshape");
  reshape->set_op_type("Reshape");
  reshape->set_domain("");
  reshape->set_doc_string("Reshape muultiply result for Cross Entropy Layer");

  //z = ReduceSum(data=z, axes=-1)

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
#endif //LBANN_HAS_ONNX

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>&
cross_entropy_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const cross_entropy_distconv_adapter<
    TensorDataType, T_layout, Dev>&>(data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>&
cross_entropy_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      static_cast<const cross_entropy_layer<TensorDataType, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>::
get_prev_activations_shape(int index) const {
  // Assumes both of the two input tensors have the equal shape.
  return data_type_distconv_adapter<TensorDataType>::get_prev_activations_shape(0);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>::
get_activations_shape(int output_index) const {
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
get_activations_local_shape(int index) const {
  assert_eq(index, 0);
  auto input_shape = this->get_prev_activations().get_local_shape();
  for (int i = 0; i < input_shape.length() - 1; ++i) {
    input_shape[i] = 1;
  }
  return input_shape;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_distributions(tensor_overlap_constraints &constraints) {
  data_type_distconv_adapter<TensorDataType>::setup_distributions(
      constraints);
  // Output tensors share all dimensions except for the sample dimension
  auto activations_split = this->get_activations_dist().get_split_shape();
  auto prev_error_signals_split = this->get_prev_error_signals_dist().get_split_shape();
  for (int i = 0; i < activations_split.length() - 1; ++i) {
    activations_split[i] = 1;
    prev_error_signals_split[i] = 1;
  }
  this->get_activations_dist().set_split_shape(activations_split);
  this->get_prev_error_signals_dist().set_split_shape(prev_error_signals_split);

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

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void cross_entropy_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
    size_t workspace_capacity) {
  m_cross_entropy = std::make_unique<dc::CrossEntropy>(dc::get_backend(), m_use_labels);
  m_cross_entropy->setup(this->get_prev_activations(0),
                         this->get_prev_activations(1),
                         this->get_activations(0));
}
#endif // LBANN_HAS_DISTCONV

#ifndef LBANN_CROSS_ENTROPY_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)              \
  extern template class cross_entropy_layer< \
    T, data_layout::DATA_PARALLEL, Device>;  \
  extern template class cross_entropy_layer< \
    T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CROSS_ENTROPY_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED
