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

#ifndef LBANN_LAYER_LEARNING_CHANNELWISE_SCALE_BIAS_HPP_INCLUDED
#define LBANN_LAYER_LEARNING_CHANNELWISE_SCALE_BIAS_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Apply per-channel scale and bias
 *
 *  The input tensor is sliced along the first tensor dimension (the
 *  "channel" dimension, assuming image data in CHW format) and scale
 *  and bias terms are applied independently to each slice. More
 *  precisely, given input and output tensors
 *  @f$ X,Y\in\mathbb{R}^{d_1\times\cdots\times d_n} @f$
 *  and scale and bias vectors @f$ a,b\in\mathbb{R}^{d_1} @f$:
 *  @f[
 *    Y_{i,j,\cdots} = a_i X_{i,j,\cdots} + b_i
 *  @f]
 *
 *  The scale and bias vectors are fused into a single weights tensor
 *  to reduce the number of gradient allreduces during backprop. In
 *  particular, the weights tensor is a
 *  @f$ \text{num\_channels} \times 2 @f$ matrix, where the first
 *  column correspond to scale terms and the second column to bias
 *  terms.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class channelwise_scale_bias_layer : public data_type_layer<TensorDataType>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "channelwise_mean_layer only supports "
                "data-parallel data layout");

public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  /** @brief The concrete optimizer type used by this object. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  ///@}

public:
  channelwise_scale_bias_layer(lbann_comm* comm = nullptr);
  channelwise_scale_bias_layer(const channelwise_scale_bias_layer& other);
  channelwise_scale_bias_layer&
  operator=(const channelwise_scale_bias_layer& other);

  channelwise_scale_bias_layer* copy() const override
  {
    return new channelwise_scale_bias_layer(*this);
  }

  std::string get_type() const override { return "channel-wise scale/bias"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | WEIGHTS | PREV_ACTIVATIONS;
  }

  void setup_data(size_t max_mini_batch_size) override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void fp_compute() override;
  void bp_compute() override;

private:
  /** @brief Objective function gradient w.r.t. weights. */
  std::unique_ptr<AbsDistMatrixType> m_weights_gradient;
};

// Implementation

template <typename T, data_layout L, El::Device D>
void channelwise_scale_bias_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_channelwise_scale_bias();
}

template <typename TensorDataType, data_layout Layout, El::Device Dev>
channelwise_scale_bias_layer<TensorDataType, Layout, Dev>::
  channelwise_scale_bias_layer(lbann_comm* comm)
  : data_type_layer<TensorDataType>(comm)
{}

template <typename TensorDataType, data_layout Layout, El::Device Dev>
channelwise_scale_bias_layer<TensorDataType, Layout, Dev>::
  channelwise_scale_bias_layer(const channelwise_scale_bias_layer& other)
  : data_type_layer<TensorDataType>(other),
    m_weights_gradient(
      other.m_weights_gradient ? other.m_weights_gradient->Copy() : nullptr)
{}

template <typename TensorDataType, data_layout Layout, El::Device Dev>
auto channelwise_scale_bias_layer<TensorDataType, Layout, Dev>::operator=(
  const channelwise_scale_bias_layer& other) -> channelwise_scale_bias_layer&
{
  data_type_layer<TensorDataType>::operator=(other);
  m_weights_gradient.reset(
    other.m_weights_gradient ? other.m_weights_gradient->Copy() : nullptr);
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Dev>
void channelwise_scale_bias_layer<TensorDataType, Layout, Dev>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);
  const El::Int num_channels = this->get_output_dims()[0];

  // Construct default weights if needed
  // Note: Scale is initialized to 1 and bias to 0
  if (!this->has_weights()) {
    auto w = std::make_shared<WeightsType>(*this->get_comm());
    std::vector<TensorDataType> vals(2 * num_channels,
                                     El::TypeTraits<TensorDataType>::Zero());
    std::fill(vals.begin(),
              vals.begin() + num_channels,
              El::TypeTraits<TensorDataType>::One());
    auto init = std::make_unique<value_initializer<TensorDataType>>(vals);
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->add_weights(w);
    this->m_model->add_weights(std::move(w));
  }
  if (this->num_weights() != 1) {
    LBANN_ERROR("attempted to setup ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "with an invalid number of weights ",
                "(expected 1, found ",
                this->num_weights(),
                ")");
  }

  // Setup weights
  auto dist = this->get_prev_activations().DistData();
  dist.colDist = El::STAR;
  dist.rowDist = El::STAR;
  this->get_weights(0).set_dims({static_cast<size_t>(num_channels)}, {2});
  this->get_weights(0).set_matrix_distribution(dist);

  // Setup gradient w.r.t. weights
  m_weights_gradient.reset(AbsDistMatrixType::Instantiate(dist));
  m_weights_gradient->AlignWith(dist);
  m_weights_gradient->Resize(num_channels, 2);
}

LBANN_DEFINE_LAYER_BUILDER(channelwise_scale_bias);

#ifndef LBANN_CHANNELWISE_SCALE_BIAS_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class channelwise_scale_bias_layer<                          \
    T,                                                                         \
    data_layout::DATA_PARALLEL,                                                \
    Device>;

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CHANNELWISE_SCALE_BIAS_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_LEARNING_CHANNELWISE_SCALE_BIAS_HPP_INCLUDED
