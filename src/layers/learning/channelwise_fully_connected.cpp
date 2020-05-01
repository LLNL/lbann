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

#define LBANN_CHANNELWISE_FULLY_CONNECTED_LAYER_INSTANTIATE
#include "lbann/layers/learning/channelwise_fully_connected.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"
#include <layers.pb.h>

namespace lbann
{

// =========================================================
// Class member functions
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::channelwise_fully_connected_layer(
  lbann_comm* comm,
  std::vector<size_t> output_channel_dims,
  bool bias,
  bool transpose)
  : data_type_layer<TensorDataType>(comm),
    m_has_bias{bias},
    m_transpose{transpose}
{

  // Initialize output tensor dimensions
  if (output_channel_dims.empty()) {
    output_channel_dims.push_back(1);
  }
  std::vector<int> output_dims;
  output_dims.push_back(1);
  output_dims.insert(
    output_dims.end(),
    output_channel_dims.begin(),
    output_channel_dims.end());
  this->set_output_dims(output_dims);

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
channelwise_fully_connected_layer<TensorDataType,Layout,Device>*
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::copy() const
{
  return new channelwise_fully_connected_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::get_type() const
{
  return "channel-wise fully-connected";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Bias", m_has_bias);
  desc.add("Transpose", m_transpose);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();

  // Make sure input and output dimensions are valid
  const auto& input_dims = this->get_input_dims();
  auto output_dims = this->get_output_dims();
  if (input_dims.size() <= 1) {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "expects an input tensor with >1 dimensions, ",
                "but parent layer ",
                "\"",this->get_parent_layers()[0]->get_name(),"\" ",
                "outputs a ",input_dims.size(),"-D tensor");
  }
  if (output_dims.size() <= 1) {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "expects an output tensor with >1 dimensions,",
                "but it has been configured ",
                "as a ",output_dims.size(),"-D tensor");
  }

  // Input and output tensors have same number of channels
  output_dims[0] = input_dims[0];
  this->set_output_dims(output_dims);

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::setup_data()
{
  data_type_layer<TensorDataType>::setup_data();

  // Tensor dimensions
  const auto& input_dims = this->get_input_dims();
  const auto& output_dims = this->get_output_dims();
  const std::vector<int> input_channel_dims(
    input_dims.begin()+1, input_dims.end());
  const std::vector<int> output_channel_dims(
    output_dims.begin()+1, output_dims.end());
  const auto& input_channel_size = std::accumulate(
    input_channel_dims.begin(), input_channel_dims.end(),
    1, std::multiplies<size_t>());
  const auto& output_channel_size = std::accumulate(
    output_channel_dims.begin(), output_channel_dims.end(),
    1, std::multiplies<size_t>());

  // Set number of weights
  using WeightsType = data_type_weights<TensorDataType>;
  if ((m_has_bias && this->num_weights() > 2)
      || (!m_has_bias && this->num_weights() > 1)) {
    LBANN_ERROR(
      "attempted to setup ",
      this->get_type()," layer \"",this->get_name(),"\" ",
      "with an invalid number of weights ",
      "(",this->num_weights(),")");
  }
  this->set_num_data_type_weights(m_has_bias ? 2 : 1);

  // Create default linearity weights if needed
  if (!this->has_data_type_weights(0)) {
    auto w = make_unique<WeightsType>(this->get_comm());
    auto init = make_unique<he_initializer<TensorDataType>>(probability_distribution::gaussian);
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_linearity_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->set_data_type_weights(0, w.get());
    this->m_model->add_weights(std::move(w));
  }

  // Setup linearity weights
  {
    auto& linearity_weights = this->get_data_type_weights(0);
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    auto* cast_initializer = dynamic_cast<variance_scaling_initializer<TensorDataType>*>(linearity_weights.get_initializer());
    if (cast_initializer != nullptr) {
      cast_initializer->set_fan_in(input_channel_size);
      cast_initializer->set_fan_out(output_channel_size);
    }
    linearity_weights.set_dims(
      m_transpose ? input_channel_dims : output_channel_dims,
      m_transpose ? output_channel_dims : input_channel_dims);
    linearity_weights.set_matrix_distribution(dist);
  }

  // Setup bias weights if needed
  if (m_has_bias) {
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    if (!this->has_data_type_weights(1)) {
      auto w = make_unique<WeightsType>(this->get_comm());
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_bias_weights");
      w->set_optimizer(std::move(opt));
      this->set_data_type_weights(1, w.get());
      this->m_model->add_weights(std::move(w));
    }
    auto& bias_weights = this->get_data_type_weights(1);
    bias_weights.set_dims(output_channel_dims);
    bias_weights.set_matrix_distribution(dist);
  }

  // Initialize freeze state
  for (auto&& w : this->get_data_type_weights()) {
    if (this->is_frozen()) {
      w->freeze();
    } else {
      w->unfreeze();
    }
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::fp_compute()
{
  const auto& zero = El::TypeTraits<TensorDataType>::Zero();
  const auto& one = El::TypeTraits<TensorDataType>::One();

  // Data tensors
  using LocalMat = El::Matrix<TensorDataType,Device>;
  const auto& linearity = this->get_data_type_weights(0).get_values();
  const auto& local_input = dynamic_cast<const LocalMat&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<LocalMat&>(this->get_local_activations());
  const auto& local_linearity = dynamic_cast<const LocalMat&>(linearity.LockedMatrix());

  // Tensor dimensions
  const auto& local_mini_batch_size = local_input.Width();
  const auto& input_dims = this->get_input_dims();
  const auto& output_dims = this->get_output_dims();
  const auto& num_channels = input_dims[0];
  const auto& input_channel_size = std::accumulate(
    input_dims.begin()+1, input_dims.end(),
    1, std::multiplies<size_t>());
  const auto& output_channel_size = std::accumulate(
    output_dims.begin()+1, output_dims.end(),
    1, std::multiplies<size_t>());

  // Reshape input and output tensors
  // Note: [mini_batch_size,num_channels,*] -> [mini_batch_size*num_channels,*]
  LocalMat local_input_reshaped, local_output_reshaped;
  if (local_input.Contiguous()) {
    local_input_reshaped.LockedAttach(
      input_channel_size,
      local_mini_batch_size * num_channels,
      local_input.LockedBuffer(),
      input_channel_size);
  }
  else {
    El::Copy(local_input, local_input_reshaped);
    local_input_reshaped.Resize(
      input_channel_size,
      local_mini_batch_size * num_channels);
  }
  if (local_output.Contiguous()) {
    local_output_reshaped.Attach(
      output_channel_size,
      local_mini_batch_size * num_channels,
      local_output.Buffer(),
      output_channel_size);
  }
  else {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "has a non-contiguous output tensor");
  }

  // Apply linearity
  El::Gemm(
    m_transpose ? El::TRANSPOSE : El::NORMAL,
    El::NORMAL,
    one, local_linearity, local_input_reshaped,
    zero, reinterpret_cast<LocalMat&>(local_output_reshaped));

  // Apply bias
  if (m_has_bias) {
    const auto& bias = this->get_data_type_weights(1).get_values();
    LocalMat ones(local_mini_batch_size * num_channels, 1);
    El::Fill(ones, one);
    El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      one, reinterpret_cast<const LocalMat&>(bias.LockedMatrix()), ones,
      one, reinterpret_cast<LocalMat&>(local_output_reshaped));
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void
channelwise_fully_connected_layer<TensorDataType,Layout,Device>
::bp_compute()
{
  const auto& zero = El::TypeTraits<TensorDataType>::Zero();
  const auto& one = El::TypeTraits<TensorDataType>::One();

  // Weights
  auto& linearity_weights = this->get_data_type_weights(0);

  // Data tensors
  using LocalMat = El::Matrix<TensorDataType,Device>;
  const auto& linearity = linearity_weights.get_values();
  const auto& local_input = dynamic_cast<const LocalMat&>(this->get_local_prev_activations());
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(this->get_local_prev_error_signals());
  auto& local_input_grad = dynamic_cast<LocalMat&>(this->get_local_error_signals());
  const auto& local_linearity = dynamic_cast<const LocalMat&>(linearity.LockedMatrix());

  // Tensor dimensions
  const auto& local_mini_batch_size = local_input.Width();
  const auto& input_dims = this->get_input_dims();
  const auto& output_dims = this->get_output_dims();
  const auto& num_channels = input_dims[0];
  const auto& input_channel_size = std::accumulate(
    input_dims.begin()+1, input_dims.end(),
    1, std::multiplies<size_t>());
  const auto& output_channel_size = std::accumulate(
    output_dims.begin()+1, output_dims.end(),
    1, std::multiplies<size_t>());

  // Reshape input and output tensors
  // Note: [mini_batch_size,num_channels,*] -> [mini_batch_size*num_channels,*]
  LocalMat local_input_reshaped, local_output_grad_reshaped, local_input_grad_reshaped;
  if (local_input.Contiguous()) {
    local_input_reshaped.LockedAttach(
      input_channel_size,
      local_mini_batch_size * num_channels,
      local_input.LockedBuffer(),
      input_channel_size);
  }
  else {
    El::Copy(local_input, local_input_reshaped);
    local_input_reshaped.Resize(
      input_channel_size,
      local_mini_batch_size * num_channels);
  }
  if (local_output_grad.Contiguous()) {
    local_output_grad_reshaped.LockedAttach(
      output_channel_size,
      local_mini_batch_size * num_channels,
      local_output_grad.LockedBuffer(),
      output_channel_size);
  }
  else {
    El::Copy(local_output_grad, local_output_grad_reshaped);
    local_output_grad_reshaped.Resize(
      output_channel_size,
      local_mini_batch_size * num_channels);
  }
  if (local_input_grad.Contiguous()) {
    local_input_grad_reshaped.Attach(
      input_channel_size,
      local_mini_batch_size * num_channels,
      local_input_grad.Buffer(),
      input_channel_size);
  }
  else {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "has a non-contiguous gradient w.r.t. input tensor");
  }

  // Compute gradient w.r.t. input
  El::Gemm(
    m_transpose ? El::NORMAL : El::TRANSPOSE,
    El::NORMAL,
    one, local_linearity, local_output_grad_reshaped,
    zero, reinterpret_cast<LocalMat&>(local_input_grad_reshaped));

  // Compute gradient w.r.t. linearity
  auto* linearity_optimizer = linearity_weights.get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale, gradient_scale;
    auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, true);
    if (m_transpose) {
      El::Gemm(
        El::NORMAL, El::TRANSPOSE,
        gradient_scale, local_input_reshaped, local_output_grad_reshaped,
        dst_scale, linearity_gradient.Matrix());
    }
    else {
      El::Gemm(
        El::NORMAL, El::TRANSPOSE,
        gradient_scale, local_output_grad_reshaped, local_input_reshaped,
        dst_scale, linearity_gradient.Matrix());
    }
  }

  // Compute gradient w.r.t. bias
  if (m_has_bias) {
    auto& bias_weights = this->get_data_type_weights(1);
    auto* bias_optimizer = bias_weights.get_optimizer();
    if (bias_optimizer != nullptr) {
      TensorDataType dst_scale, gradient_scale;
      auto& bias_gradient = bias_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
      LocalMat ones(local_mini_batch_size * num_channels, 1);
      El::Fill(ones, one);
      El::Gemv(
        El::NORMAL,
        gradient_scale, local_output_grad_reshaped, ones,
        dst_scale, bias_gradient.Matrix());
    }
  }

}

// =========================================================
// Builder function
// =========================================================

namespace
{

template <typename T, data_layout L, El::Device D>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR(
      "Attempted to construct channelwise_fully_connected_layer ",
      "with invalid parameters ",
      "(TensorDataType=",TypeName<T>(),", ",
      "Layout=",to_string(L),", ",
      "Device=",to_string(D),")");
    return nullptr;
  }
};

template <typename TensorDataType, El::Device Device>
struct Builder<TensorDataType,data_layout::DATA_PARALLEL,Device>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType = channelwise_fully_connected_layer<
      TensorDataType,
      data_layout::DATA_PARALLEL,
      Device>;
    return make_unique<LayerType>(std::forward<Args>(args)...);
  }
};

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_channelwise_fully_connected_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const& proto_layer)
{
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, channelwise_fully_connected);
  const auto& params = proto_layer.channelwise_fully_connected();
  std::vector<size_t> output_channel_dims;
  const size_t num_output_channel_dims = params.output_channel_dims_size();
  for (size_t i=0; i<num_output_channel_dims; ++i) {
    output_channel_dims.push_back(params.output_channel_dims(i));
  }
  const bool has_bias = (params.has_bias()
                         ? params.bias().value()
                         : true);
  const bool transpose = (params.has_transpose()
                          ? params.transpose().value()
                          : false);
  return BuilderType::Build(comm, output_channel_dims, has_bias, transpose);
}

// =========================================================
// Explicit template instantiation
// =========================================================

#define PROTO_DEVICE(T, Device)                                         \
  template class channelwise_fully_connected_layer<                     \
    T,data_layout::DATA_PARALLEL,Device>;                               \
  LBANN_LAYER_BUILDER_ETI(channelwise_fully_connected, T, Device)
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
