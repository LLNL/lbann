////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class mean_squared_error_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  mean_squared_error_distconv_adapter(Layer& layer)
      : data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~mean_squared_error_distconv_adapter() = default;
  void setup_distributions(tensor_overlap_constraints &constraints) override;
  dc::Shape get_prev_activations_shape(int index) const override;
  dc::Shape get_activations_shape(int index) const override;
  dc::Shape get_activations_local_shape(int index) const override;
  void setup_layer(size_t workspace_capacity) override;
  std::unique_ptr<dc::MeanSquaredError> m_mean_squared_error;
};
#endif // LBANN_HAS_DISTCONV

/** @brief
 *
 *  Given a prediction @f$y@f$ and ground truth @f$\hat{y}@f$,
 *  @f[
 *    MSE(y,\hat{y})
 *      = \frac{1}{n} \sum\limits_{i=1}^{n} (y_i - \hat{y}_i)^2
 *  @f]
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class mean_squared_error_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  mean_squared_error_layer(lbann_comm *comm) : data_type_layer<TensorDataType>(comm) {
    this->m_expected_num_parent_layers = 2;
  }

  mean_squared_error_layer(const mean_squared_error_layer& other)
    : data_type_layer<TensorDataType>(other) {
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
  }

  mean_squared_error_layer& operator=(const mean_squared_error_layer& other) {
    data_type_layer<TensorDataType>::operator=(other);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
    return *this;
  }

  mean_squared_error_layer* copy() const override { return new mean_squared_error_layer(*this); }
  std::string get_type() const override { return "mean squared error"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
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
    const auto& input_dist = this->get_prev_activations(0).DistData();
    m_workspace.reset(AbsDistMatrixType::Instantiate(*input_dist.grid,
                                              input_dist.root,
                                              El::STAR,
                                              input_dist.rowDist,
                                              (input_dist.blockHeight == 1
                                               && input_dist.blockWidth == 1 ?
                                               El::ELEMENT : El::BLOCK),
                                              input_dist.device));
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

  void bp_compute() override {

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

private:

  /** Compute local contributions to mean squared error loss. */
  void local_fp_compute();
  /** Compute local gradients. */
  void local_bp_compute();

  /** Workspace matrix. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

#ifdef LBANN_HAS_DISTCONV
  friend class mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>;
 protected:
  bool is_distconv_supported() const override {
    return Dev == El::Device::GPU && T_layout == data_layout::DATA_PARALLEL;
  }

  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override {
    this->get_distconv_adapter_ptr() = make_unique<
      mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>>(*this);
  }

  mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() override;
  const mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() const override;

  void fp_compute_distconv() {
    assert_always(this->distconv_enabled());
    get_distconv_adapter().m_mean_squared_error->forward(this->get_distconv_adapter().get_prev_activations(0),
                                                         this->get_distconv_adapter().get_prev_activations(1),
                                                         this->get_distconv_adapter().get_activations());
  }

  void bp_compute_distconv() {
    assert_always(this->distconv_enabled());
    get_distconv_adapter().m_mean_squared_error->backward(this->get_distconv_adapter().get_prev_activations(0),
                                                          this->get_distconv_adapter().get_prev_activations(1),
                                                          this->get_distconv_adapter().get_prev_error_signals(0),
                                                          this->get_distconv_adapter().get_error_signals(0),
                                                          this->get_distconv_adapter().get_error_signals(1));
  }
#endif // LBANN_HAS_DISTCONV

};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>&
mean_squared_error_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const mean_squared_error_distconv_adapter<
    TensorDataType, T_layout, Dev>&>(data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>&
mean_squared_error_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      static_cast<const mean_squared_error_layer<TensorDataType, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::
get_prev_activations_shape(int index) const {
  // Assumes both of the two input tensors have the equal shape.
  return data_type_distconv_adapter<TensorDataType>::get_prev_activations_shape(0);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::
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
dc::Shape mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::
get_activations_local_shape(int index) const {
  assert_eq(index, 0);
  auto input_shape = this->get_prev_activations().get_local_shape();
  for (int i = 0; i < input_shape.length() - 1; ++i) {
    input_shape[i] = 1;
  }
  return input_shape;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::
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
void mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
    size_t workspace_capacity) {
  m_mean_squared_error = make_unique<dc::MeanSquaredError>(dc::get_backend());
  m_mean_squared_error->setup(this->get_prev_activations(0),
                         this->get_prev_activations(1),
                         this->get_activations(0));
}
#endif // LBANN_HAS_DISTCONV

#ifndef LBANN_MEAN_SQUARED_ERROR_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                     \
  extern template class mean_squared_error_layer<   \
    T, data_layout::DATA_PARALLEL, Device>;         \
  extern template class mean_squared_error_layer<   \
    T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_MEAN_SQUARED_ERROR_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP_INCLUDED
