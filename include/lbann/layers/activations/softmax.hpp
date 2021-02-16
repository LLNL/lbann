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

#ifndef LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/dnn_enums.hpp"
#if defined LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/dnn_lib/softmax.hpp"
#endif // defined LBANN_HAS_DNN_LIB

// Threshold outputs to a minimum value.

// If enabled, the minimum output value is sqrt(min), where min is the
// minimum, normalized, positive value (~1e-19 for float and ~1e-154
// for double). During backprop, gradients are computed as if
// thresholding did not occur, so there will be a discrepancy for
// values that are thresholded.
#define LBANN_ENABLE_SOFTMAX_THRESHOLD

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class softmax_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;

  softmax_distconv_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~softmax_distconv_adapter() = default;

  void setup_distributions(tensor_overlap_constraints &constraints) override;
  void setup_layer(size_t workspace_capacity) override;

  std::unique_ptr<dc::Softmax> m_softmax;
};
#endif // LBANN_HAS_DISTCONV

/**
 *  @f[ \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} @f]
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class softmax_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  softmax_layer(lbann_comm *comm,
                softmax_mode mode)
    : data_type_layer<TensorDataType>(comm),
      m_mode(mode)
#ifdef LBANN_HAS_DNN_LIB
    , m_tensors_dnn_desc(this)
#endif // LBANN_HAS_DNN_LIB
  {
    if(mode == softmax_mode::INVALID) {
      LBANN_ERROR("invalid softmax mode");
    }
  }

  softmax_layer(const softmax_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_mode(other.m_mode),
      m_workspace(other.m_workspace ?
                  other.m_workspace->Copy() : nullptr)
#ifdef LBANN_HAS_DNN_LIB
    , m_tensors_dnn_desc(other.m_tensors_dnn_desc)
#endif // LBANN_HAS_DNN_LIB
  {
#ifdef LBANN_HAS_DNN_LIB
    m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
  }

  ~softmax_layer() = default;

  softmax_layer* copy() const override { return new softmax_layer(*this); }
  std::string get_type() const override { return "softmax"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims(this->get_input_dims());
  }

  void setup_matrices(const El::Grid& grid) override {
    data_type_layer<TensorDataType>::setup_matrices(grid);
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    m_workspace.reset(AbsDistMatrixType::Instantiate(dist));
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB
#ifdef LBANN_HAS_DNN_LIB
    if (!m_tensors_dnn_desc.get_layer())
      m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    data_type_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);
    const auto& dist_data = this->get_prev_activations().DistData();
    m_workspace->Empty(false);
    m_workspace->AlignWith(dist_data);
    m_workspace->Resize(1, mini_batch_size);
  }

  void fp_compute() override;
  void bp_compute() override;

  template <typename U>
  friend void fp_compute_impl(softmax_layer<U, Layout, Device>& l);
  template <typename U>
  friend void bp_compute_impl(softmax_layer<U, Layout, Device>& l);

  template <typename ArchiveT>
  void serialize(ArchiveT& ar)
  {
    using DataTypeLayer = data_type_layer<TensorDataType>;
    ar(::cereal::make_nvp("DataTypeLayer",
                          ::cereal::base_class<DataTypeLayer>(this)),
       CEREAL_NVP(m_mode),
       CEREAL_NVP(threshold_val));
  }

private:
  friend cereal::access;
  softmax_layer() : data_type_layer<TensorDataType>(nullptr) {}

  /** Softmax mode. */
  softmax_mode m_mode;

  /** Workspace for column-wise reductions. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

#ifdef LBANN_HAS_DNN_LIB
  /** Tensor DNN library descriptors. */
  dnn_lib::data_parallel_layer_tensor_manager<TensorDataType> m_tensors_dnn_desc;
#endif // LBANN_HAS_DNN_LIB

// Minimum output value to avoid denormalized floats
#ifdef LBANN_ENABLE_SOFTMAX_THRESHOLD
  TensorDataType threshold_val = static_cast<TensorDataType>(El::Sqrt(std::numeric_limits<TensorDataType>::min()));
#else
  TensorDataType threshold_val = El::TypeTraits<TensorDataType>::Zero();
#endif // LBANN_ENABLE_SOFTMAX_THRESHOLD

#ifdef LBANN_HAS_DISTCONV
  friend class softmax_distconv_adapter<TensorDataType, Layout, Device>;
 protected:
  bool is_distconv_supported() const override {
    return Device == El::Device::GPU && Layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override {
    this->get_distconv_adapter_ptr() = make_unique<softmax_distconv_adapter<
      TensorDataType, Layout, Device>>(*this);
  }
  softmax_distconv_adapter<TensorDataType, Layout, Device>& get_distconv_adapter() override;
  const softmax_distconv_adapter<TensorDataType, Layout, Device>& get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
softmax_distconv_adapter<TensorDataType, T_layout, Dev>&
softmax_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<softmax_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      static_cast<const softmax_layer<TensorDataType, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const softmax_distconv_adapter<TensorDataType, T_layout, Dev>&
softmax_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const softmax_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void softmax_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_distributions(tensor_overlap_constraints &constraints) {
  data_type_distconv_adapter<TensorDataType>::setup_distributions(
      constraints);
  // No overlap supported yet
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
void softmax_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
    size_t workspace_capacity) {
  auto &l = dynamic_cast<softmax_layer<TensorDataType, T_layout, Dev>&>(
      this->layer());
  m_softmax = make_unique<dc::Softmax>(dc::get_backend());
  auto mode = l.m_mode == softmax_mode::INSTANCE ?
                          ::distconv::SoftmaxMode::INSTANCE :
      ::distconv::SoftmaxMode::CHANNEL;
  m_softmax->setup(this->get_prev_activations(), mode);
}
#endif // LBANN_HAS_DISTCONV


LBANN_DEFINE_LAYER_BUILDER(softmax);

#ifndef LBANN_SOFTMAX_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class softmax_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class softmax_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SOFTMAX_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED
