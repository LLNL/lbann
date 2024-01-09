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

#ifndef LBANN_LAYERS_REGULARIZERS_LAYER_NORM_HPP_INCLUDED
#define LBANN_LAYERS_REGULARIZERS_LAYER_NORM_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/datatype_helpers.hpp"

#include "lbann/proto/layers.pb.h"
#include <memory>

namespace lbann {

/** @brief Normalize over data samples
 *
 *  Each data sample is normalized to have zero mean and unit standard
 *  deviation. See:
 *
 *  Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer
 *  normalization." arXiv preprint arXiv:1607.06450 (2016).
 *
 *  Note that this layer does not apply an entry-wise scale and bias
 *  like in the paper. Use the entry-wise scale/bias layer to
 *  reproduce that functionality.
 *
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class layer_norm_layer : public data_type_layer<TensorDataType>
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:
  /**
   *  @param epsilon    Small number to avoid division by zero
   */
  layer_norm_layer(TensorDataType epsilon = El::To<TensorDataType>(1e-5),
                   bool scale = false,
                   bool bias = false,
                   int start_dim = 0);

  layer_norm_layer(const layer_norm_layer& other);
  layer_norm_layer& operator=(const layer_norm_layer& other);
  layer_norm_layer* copy() const override;

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  description get_description() const override;
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void setup_dims() override;
  void setup_data(size_t max_mini_batch_size) override;

  void fp_compute() override;
  void bp_compute() override;

private:
  using AbsDistMatType = El::AbstractDistMatrix<TensorDataType>;

  /** Small number to avoid division by zero. */
  TensorDataType m_epsilon;

  /** @brief Apply elementwise scale after normalization (learned weights). */
  bool m_scale;

  /** @brief Apply elementwise bias after normalization (learned weights). */
  bool m_bias;

  /** @brief The tensor dimension to start normalizing from. */
  int m_start_dim;

  /** @brief Per-sample statistics.
   *
   *  The means and variances are fused for performance.
   */
  std::unique_ptr<AbsDistMatType> m_statistics;
  /** @brief Gradients w.r.t. per-sample statistics.
   *
   *  The means and variances are fused for performance.
   */
  std::unique_ptr<AbsDistMatType> m_statistics_gradient;

  /** @brief Gradient w.r.t. scale. */
  std::unique_ptr<AbsDistMatType> m_scale_gradient;

  /** @brief Gradient w.r.t. bias. */
  std::unique_ptr<AbsDistMatType> m_bias_gradient;

  /** @brief Helper function to obtain normalization parameters. */
  void get_normdims(El::Int& normalization_size,
                    El::Int& num_normalized,
                    El::Int& normalization_stride);
};

// =========================================================
// Implementation
// =========================================================

template <typename T, data_layout L, El::Device D>
void layer_norm_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_layer_norm();
  msg->mutable_epsilon()->set_value(m_epsilon);
  msg->set_scale(m_scale);
  msg->set_bias(m_bias);
  msg->set_start_dim(m_start_dim);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType, Layout, Device>::layer_norm_layer(
  TensorDataType epsilon,
  bool scale,
  bool bias,
  int start_dim)
  : data_type_layer<TensorDataType>(nullptr),
    m_epsilon(epsilon),
    m_scale(scale),
    m_bias(bias),
    m_start_dim(start_dim)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType, Layout, Device>::layer_norm_layer(
  const layer_norm_layer<TensorDataType, Layout, Device>& other)
  : data_type_layer<TensorDataType>(other),
    m_epsilon(other.m_epsilon),
    m_scale(other.m_scale),
    m_bias(other.m_bias),
    m_start_dim(other.m_start_dim),
    m_statistics(other.m_statistics ? other.m_statistics->Copy() : nullptr),
    m_statistics_gradient(other.m_statistics_gradient
                            ? other.m_statistics_gradient->Copy()
                            : nullptr),
    m_scale_gradient(other.m_scale_gradient ? other.m_scale_gradient->Copy()
                                            : nullptr),
    m_bias_gradient(other.m_bias_gradient ? other.m_bias_gradient->Copy()
                                          : nullptr)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType, Layout, Device>&
layer_norm_layer<TensorDataType, Layout, Device>::operator=(
  const layer_norm_layer<TensorDataType, Layout, Device>& other)
{
  data_type_layer<TensorDataType>::operator=(other);
  m_epsilon = other.m_epsilon;
  m_scale = other.m_scale;
  m_bias = other.m_bias;
  m_start_dim = other.m_start_dim;
  m_statistics.reset(other.m_statistics ? other.m_statistics->Copy() : nullptr);
  m_statistics_gradient.reset(other.m_statistics_gradient
                                ? other.m_statistics_gradient->Copy()
                                : nullptr);
  m_scale_gradient.reset(other.m_scale_gradient ? other.m_scale_gradient->Copy()
                                                : nullptr);
  m_bias_gradient.reset(other.m_bias_gradient ? other.m_bias_gradient->Copy()
                                              : nullptr);
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType, Layout, Device>*
layer_norm_layer<TensorDataType, Layout, Device>::copy() const
{
  return new layer_norm_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string layer_norm_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "layer norm";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
layer_norm_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
layer_norm_layer<TensorDataType, Layout, Device>::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
layer_norm_layer<TensorDataType, Layout, Device>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Epsilon", m_epsilon);
  desc.add("Affine Scale", m_scale);
  desc.add("Affine Bias", m_bias);
  desc.add("Start dimension", m_start_dim);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();
  this->set_output_dims(this->get_input_dims());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType, Layout, Device>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);
  const auto& output_dims = this->get_output_dims();
  std::vector<size_t> out_dims{output_dims.begin(), output_dims.end()};

  int start_dim;
  if (m_start_dim >= 0) {
    start_dim = m_start_dim;
  }
  else {
    start_dim = static_cast<int>(out_dims.size()) + m_start_dim;
  }
  if (start_dim < 0 || start_dim >= static_cast<int>(output_dims.size())) {
    LBANN_ERROR("Layer normalization \"",
                this->get_name(),
                "\" start dimension ",
                m_start_dim,
                "does not match the input "
                "tensor dimensionality of ",
                output_dims.size());
  }
  std::vector<size_t> normalized_dims{output_dims.begin() + start_dim,
                                      output_dims.end()};

  auto dist = this->get_prev_activations().DistData();
  dist.colDist = El::STAR;
  m_statistics.reset(AbsDistMatrixType::Instantiate(dist));
  m_statistics_gradient.reset(AbsDistMatrixType::Instantiate(dist));

  // Setup weights
  using WeightsType = data_type_weights<TensorDataType>;
  if ((m_scale && m_bias && this->num_weights() > 2) ||
      (!m_scale && !m_bias && this->num_weights() > 0) ||
      (m_scale && !m_bias && this->num_weights() > 1) ||
      (!m_scale && m_bias && this->num_weights() > 1)) {
    LBANN_ERROR("attempted to setup ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "with an invalid number of weights ",
                "(",
                this->num_weights(),
                ") and scale = ",
                m_scale,
                ", bias = ",
                m_bias);
  }
  this->set_num_weights((m_scale ? 1 : 0) + (m_bias ? 1 : 0));

  // Setup default weights if not given
  int weight_idx = 0;

  // Replicate weights across minibatch
  dist = this->get_prev_activations().DistData();
  dist.rowDist = El::STAR;

  if (m_scale) {
    if (!this->has_weights(weight_idx)) {
      auto w = std::make_shared<WeightsType>(*this->get_comm());
      auto init = std::make_unique<constant_initializer<TensorDataType>>(
        El::TypeTraits<TensorDataType>::One());
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_scale_weights");
      w->set_optimizer(std::move(opt));
      w->set_initializer(std::move(init));
      this->set_weights(weight_idx, w);
      this->m_model->add_weights(std::move(w));
    }
    auto& weights = this->get_weights(weight_idx);
    weights.set_dims(normalized_dims);
    weights.set_matrix_distribution(dist);
    m_scale_gradient.reset(AbsDistMatrixType::Instantiate(dist));
    m_scale_gradient->AlignWith(dist);
    m_scale_gradient->Resize(weights.get_matrix_height(),
                             weights.get_matrix_width());
    ++weight_idx;
  }
  if (m_bias) {
    if (!this->has_weights(weight_idx)) {
      auto w = std::make_shared<WeightsType>(*this->get_comm());
      auto init = std::make_unique<constant_initializer<TensorDataType>>(
        El::TypeTraits<TensorDataType>::Zero());
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_bias_weights");
      w->set_optimizer(std::move(opt));
      w->set_initializer(std::move(init));
      this->set_weights(weight_idx, w);
      this->m_model->add_weights(std::move(w));
    }
    auto& weights = this->get_weights(weight_idx);
    weights.set_dims(normalized_dims);
    weights.set_matrix_distribution(dist);
    m_bias_gradient.reset(AbsDistMatrixType::Instantiate(dist));
    m_bias_gradient->AlignWith(dist);
    m_bias_gradient->Resize(weights.get_matrix_height(),
                            weights.get_matrix_width());
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType, Layout, Device>::get_normdims(
  El::Int& normalization_size,
  El::Int& num_normalized,
  El::Int& normalization_stride)
{
  auto const& dims = this->get_output_dims();
  unsigned int start_dim;
  if (m_start_dim >= 0) {
    start_dim = static_cast<unsigned int>(m_start_dim);
  }
  else {
    start_dim = static_cast<unsigned int>(dims.size() + m_start_dim);
  }

  num_normalized = 1;
  normalization_size = 1;
  for (unsigned int i = 0; i < start_dim; ++i) {
    num_normalized *= dims[i];
  }
  for (unsigned int i = start_dim; i < dims.size(); ++i) {
    normalization_size *= dims[i];
  }
  // Assuming contiguous tensors for now
  normalization_stride = normalization_size;
}

LBANN_DEFINE_LAYER_BUILDER(layer_norm);

// =========================================================
// Explicit template instantiation
// =========================================================

#ifndef LBANN_LAYER_NORM_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class layer_norm_layer<T,                                    \
                                         data_layout::DATA_PARALLEL,           \
                                         Device>;                              \
  extern template class layer_norm_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_LAYER_NORM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_REGULARIZERS_LAYER_NORM_HPP_INCLUDED
