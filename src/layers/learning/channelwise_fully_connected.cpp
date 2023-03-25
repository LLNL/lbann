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

#define LBANN_CHANNELWISE_FULLY_CONNECTED_LAYER_INSTANTIATE
#include "lbann/layers/learning/channelwise_fully_connected.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"

#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

template <typename T, data_layout L, El::Device D>
void channelwise_fully_connected_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_channelwise_fully_connected();
  auto const& dims = this->get_output_dims();
  for (size_t ii = 1; ii < dims.size(); ii++)
    msg->add_output_channel_dims(dims[ii]);
  msg->mutable_bias()->set_value(m_has_bias);
  msg->mutable_transpose()->set_value(m_transpose);
}

// =========================================================
// DistConv-Adapter member functions
// =========================================================

#ifdef LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_distconv_adapter<
  TensorDataType,
  Layout,
  Device>::setup_distributions(tensor_overlap_constraints& constraints)
{

  data_type_distconv_adapter<TensorDataType>::setup_distributions(constraints);

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

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_distconv_adapter<
  TensorDataType,
  Layout,
  Device>::setup_layer(size_t workspace_capacity)
{
  data_type_distconv_adapter<TensorDataType>::setup_layer(workspace_capacity);

  m_linear_operator =
    std::make_unique<dc::ChannelwiseFullyConnected<TensorDataType>>(
      dc::get_backend());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                  Layout,
                                                  Device>::setup_fp_tensors()
{
  data_type_distconv_adapter<TensorDataType>::setup_fp_tensors();

  // dc::MPIRootPrintStreamInfo() << "STARTING SETTING UP FP TENSORS " <<
  // std::endl;

  auto& layer = dynamic_cast<
    channelwise_fully_connected_layer<TensorDataType, Layout, Device>&>(
    this->layer());
  // Setup up forward pass tensors here

  // Get shape of the linear weights

  const auto& linearity_dims = layer.get_linearity_dims();
  const auto& input_dist = this->get_prev_activations_dist();
  dc::Shape linearity_shape(linearity_dims);

  // Create distribution from distconv
  auto shared_dist =
    dc::Dist::make_shared_distribution(input_dist.get_locale_shape());
  // Create LocaleMPI via distconv

  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  // Create new distconv tensor using distribution

  m_linear = std::make_unique<TensorDevType>(linearity_shape, loc, shared_dist);

  // This distconv tensor m_linear will be Viewed during forward compute

  // Apply bias
  if (layer.m_has_bias) {
    // get bias shape
    const auto& bias_dims = layer.get_bias_dims();
    dc::Shape bias_shape(bias_dims);
    m_bias = std::make_unique<TensorDevType>(bias_shape, loc, shared_dist);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                  Layout,
                                                  Device>::setup_bp_tensors()
{
  data_type_distconv_adapter<TensorDataType>::setup_bp_tensors();

  auto& layer = dynamic_cast<
    channelwise_fully_connected_layer<TensorDataType, Layout, Device>&>(
    this->layer());

  //  Setup backward pass tensors here

  // create LocaleMPI from distconv
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  // Get shape of the linear weights

  const auto& linearity_dims = layer.get_linearity_dims();
  dc::Shape linearity_shape(linearity_dims);

  const auto shared_dist = dc::Dist::make_shared_distribution(
    this->get_prev_error_signals_dist().get_locale_shape());
  m_linearity_gradient =
    std::make_unique<TensorDevType>(linearity_shape, loc, shared_dist);

  auto* linearity_optimizer = static_cast<data_type_optimizer<TensorDataType>*>(
    layer.get_weights(0).get_optimizer());

  assert0(dc::tensor::View(*m_linearity_gradient,
                           linearity_optimizer->get_gradient().Buffer()));
  if (layer.m_has_bias) {
    // Get bias optimizer
    auto* bias_optimizer = static_cast<data_type_optimizer<TensorDataType>*>(
      layer.get_weights(1).get_optimizer());

    if (bias_optimizer != nullptr) {
      // create shape for bias grad
      const auto& bias_dims = layer.get_bias_dims();
      dc::Shape bias_shape(bias_dims);
      m_bias_gradient =
        std::make_unique<TensorDevType>(bias_shape, loc, shared_dist);

      // Copy over bias gradients
      assert0(dc::tensor::View(*m_bias_gradient,
                               bias_optimizer->get_gradient().Buffer()));
    }
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                  Layout,
                                                  Device>::fp_compute()
{

  auto& layer = dynamic_cast<
    channelwise_fully_connected_layer<TensorDataType, Layout, Device>&>(
    this->layer());

  const auto& linearity = layer.weights_values(0);

  // TO DO: Check if input and output tensors are contiguous

  assert0(dc::tensor::View(*m_linear, linearity.LockedBuffer()));

  m_linear_operator->forward(layer.m_transpose,
                             this->get_prev_activations(),
                             *m_linear,
                             this->get_activations());

  if (layer.m_has_bias) {
    const auto& bias = layer.weights_values(1);
    assert0(dc::tensor::View(*m_bias, bias.LockedBuffer()));

    m_linear_operator->apply_bias(*m_bias, this->get_activations());
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                  Layout,
                                                  Device>::bp_compute()
{
  auto& layer = dynamic_cast<
    channelwise_fully_connected_layer<TensorDataType, Layout, Device>&>(
    this->layer());

  const auto& linearity = layer.weights_values(0);

  // Assuming the matrices are contiguous (May need to add checks or reshape)

  TensorDataType dst_scale, gradient_scale;

  assert0(dc::tensor::View(*m_linear, linearity.LockedBuffer()));

  m_linear_operator->backward_wrt_input(layer.m_transpose,
                                        this->get_prev_error_signals(),
                                        *m_linear,
                                        this->get_error_signals());
  auto* linearity_optimizer = static_cast<data_type_optimizer<TensorDataType>*>(
    layer.get_weights(0).get_optimizer());

  if (linearity_optimizer == nullptr) {
    dc::MPIRootPrintStreamInfo()
      << "Weights optimizer null. Exiting ...." << std::endl;
    return;
  }
  auto& linearity_gradient =
    linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);

  assert0(dc::tensor::View(*m_linearity_gradient, linearity_gradient.Buffer()));
  m_linear_operator->backward_wrt_weight(layer.m_transpose,
                                         dst_scale,
                                         gradient_scale,
                                         this->get_prev_activations(),
                                         this->get_prev_error_signals(),
                                         *m_linearity_gradient);

  if (layer.m_has_bias) {
    auto* bias_optimizer = static_cast<data_type_optimizer<TensorDataType>*>(
      layer.get_weights(1).get_optimizer());
    if (bias_optimizer == nullptr)
      return;

    auto& bias_gradient =
      bias_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);

    assert0(dc::tensor::View(*m_bias_gradient, bias_gradient.Buffer()));

    m_linear_operator->backward_wrt_bias(gradient_scale,
                                         dst_scale,
                                         this->get_prev_error_signals(),
                                         *m_bias_gradient);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape
channelwise_fully_connected_distconv_adapter<TensorDataType, Layout, Device>::
  get_activations_local_shape(int index) const
{

  // The default case assumes that the local is shape is the same as
  // the local shape of the first previous activations

  // Need to update such that the height and width dimensions match
  // match the output dimensions expected

  const auto& layer = dynamic_cast<
    const channelwise_fully_connected_layer<TensorDataType, Layout, Device>&>(
    this->layer());

  auto linearity_dims = layer.get_linearity_dims();

  std::reverse(std::begin(linearity_dims), std::end(linearity_dims));
  const auto output_shape =
    ::distconv::get_fc_output_local_tensor_shape(this->get_prev_activations(),
                                                 linearity_dims,
                                                 layer.m_transpose);
  return output_shape;
}

// =============================================================
// DistConv-enabled Channelwise fullyconnected member functions
// =============================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
bool channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  is_distconv_supported() const
{
  return Device == El::Device::GPU && Layout == data_layout::DATA_PARALLEL;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  setup_distconv_adapter(const DataReaderMetaData& dr_metadata)
{
  this->get_distconv_adapter_ptr() = std::make_unique<
    channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                 Layout,
                                                 Device>>(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
const channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                   Layout,
                                                   Device>&
channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  get_distconv_adapter() const
{
  return dynamic_cast<
    const channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                       Layout,
                                                       Device>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
channelwise_fully_connected_distconv_adapter<TensorDataType, Layout, Device>&
channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  get_distconv_adapter()
{
  return const_cast<channelwise_fully_connected_distconv_adapter<TensorDataType,
                                                                 Layout,
                                                                 Device>&>(
    static_cast<
      const channelwise_fully_connected_layer<TensorDataType, Layout, Device>&>(
      *this)
      .get_distconv_adapter());
}

#endif //  LBANN_HAS_DISTCONV

// =========================================================
// Class member functions
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  channelwise_fully_connected_layer(std::vector<size_t> output_channel_dims,
                                    bool bias,
                                    bool transpose)
  : data_type_layer<TensorDataType>(nullptr),
    m_has_bias{bias},
    m_transpose{transpose}
{

  // Initialize output tensor dimensions
  if (output_channel_dims.empty()) {
    output_channel_dims.push_back(1);
  }
  std::vector<int> output_dims;
  output_dims.push_back(1);
  output_dims.insert(output_dims.end(),
                     output_channel_dims.begin(),
                     output_channel_dims.end());
  this->set_output_dims(output_dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  channelwise_fully_connected_layer()
  : channelwise_fully_connected_layer({}, false, false)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
channelwise_fully_connected_layer<TensorDataType, Layout, Device>*
channelwise_fully_connected_layer<TensorDataType, Layout, Device>::copy() const
{
  return new channelwise_fully_connected_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string
channelwise_fully_connected_layer<TensorDataType, Layout, Device>::get_type()
  const
{
  return "channel-wise fully-connected";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Bias", m_has_bias);
  desc.add("Transpose", m_transpose);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::vector<int>
channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  get_linearity_dims() const
{
  const auto& input_dims = this->get_input_dims();
  const auto& output_dims = this->get_output_dims();

  const std::vector<size_t> input_channel_dims(input_dims.begin() + 1,
                                               input_dims.end());

  const std::vector<size_t> output_channel_dims(output_dims.begin() + 1,
                                                output_dims.end());

  const auto& input_channel_size = std::accumulate(input_channel_dims.begin(),
                                                   input_channel_dims.end(),
                                                   1,
                                                   std::multiplies<size_t>());

  const auto& output_channel_size = std::accumulate(output_channel_dims.begin(),
                                                    output_channel_dims.end(),
                                                    1,
                                                    std::multiplies<size_t>());

  const auto linearity_dim_rows =
    this->m_transpose ? output_channel_size : input_channel_size;
  const auto linearity_dims_cols =
    this->m_transpose ? input_channel_size : output_channel_size;
  std::vector<int> linearity_dims{1,
                                  1,
                                  linearity_dim_rows,
                                  linearity_dims_cols};
  return linearity_dims;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::vector<int>
channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  get_bias_dims() const
{
  const auto& output_dims = this->get_output_dims();

  const std::vector<size_t> output_channel_dims(output_dims.begin() + 1,
                                                output_dims.end());

  const auto& bias_size = std::accumulate(output_channel_dims.begin(),
                                          output_channel_dims.end(),
                                          1,
                                          std::multiplies<size_t>());

  std::vector<int> bias_dims{1, 1, bias_size, 1};
  return bias_dims;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  setup_dims(DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Make sure input and output dimensions are valid
  const auto& input_dims = this->get_input_dims();
  auto output_dims = this->get_output_dims();
  if (input_dims.size() <= 1) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "expects an input tensor with >1 dimensions, ",
                "but parent layer ",
                "\"",
                this->get_parent_layers()[0]->get_name(),
                "\" ",
                "outputs a ",
                input_dims.size(),
                "-D tensor");
  }
  if (output_dims.size() <= 1) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "expects an output tensor with >1 dimensions,",
                "but it has been configured ",
                "as a ",
                output_dims.size(),
                "-D tensor");
  }

#ifdef LBANN_HAS_DISTCONV

  if (this->distconv_enabled()) {
    if (input_dims.size() != 3 || output_dims.size() != 3) {
      LBANN_ERROR(
        this->get_type(),
        " layer \"",
        this->get_name(),
        "\" ",
        "expects an input and output tensor with 3 dimensions (channel, *, *), "
        "but it has been configured as a ",
        input_dims.size(),
        "-D input tensor and ",
        output_dims.size(),
        "-D output tensor");
    }
  }
#endif // LBANN_HAS_DISTCONV

  // Input and output tensors have same number of channels
  output_dims[0] = input_dims[0];
  this->set_output_dims(output_dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  setup_data(size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Tensor dimensions
  const auto& input_dims = this->get_input_dims();
  const auto& output_dims = this->get_output_dims();
  const std::vector<size_t> input_channel_dims(input_dims.begin() + 1,
                                               input_dims.end());
  const std::vector<size_t> output_channel_dims(output_dims.begin() + 1,
                                                output_dims.end());
  const auto& input_channel_size = std::accumulate(input_channel_dims.begin(),
                                                   input_channel_dims.end(),
                                                   1,
                                                   std::multiplies<size_t>());
  const auto& output_channel_size = std::accumulate(output_channel_dims.begin(),
                                                    output_channel_dims.end(),
                                                    1,
                                                    std::multiplies<size_t>());

  // Set number of weights
  using WeightsType = data_type_weights<TensorDataType>;
  if ((m_has_bias && this->num_weights() > 2) ||
      (!m_has_bias && this->num_weights() > 1)) {
    LBANN_ERROR("attempted to setup ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "with an invalid number of weights ",
                "(",
                this->num_weights(),
                ")");
  }
  this->set_num_weights(m_has_bias ? 2 : 1);

  // Create default linearity weights if needed
  if (!this->has_weights(0)) {
    auto w = std::make_shared<WeightsType>(*this->get_comm());
    auto init = std::make_unique<he_initializer<TensorDataType>>(
      probability_distribution::gaussian);
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_linearity_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->set_weights(0, w);
    this->m_model->add_weights(std::move(w));
  }

  // Setup linearity weights
  {
    auto& linearity_weights = this->get_weights(0);
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    if (auto* initializer = linearity_weights.get_initializer()) {
      set_fan_in(*initializer, input_channel_size);
      set_fan_out(*initializer, output_channel_size);
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
    if (!this->has_weights(1)) {
      auto w = std::make_shared<WeightsType>(*this->get_comm());
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_bias_weights");
      w->set_optimizer(std::move(opt));
      this->set_weights(1, w);
      this->m_model->add_weights(std::move(w));
    }
    auto& bias_weights = this->get_weights(1);
    bias_weights.set_dims(output_channel_dims);
    bias_weights.set_matrix_distribution(dist);
  }

  // Initialize freeze state
  auto const num_weights = this->num_weights();
  for (size_t ii = 0; ii < num_weights; ++ii) {
    auto& w = this->get_weights(ii);
    if (this->m_frozen) {
      w.freeze();
    }
    else {
      w.unfreeze();
    }
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  fp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  // We are guaranteed to have
  if (this->distconv_enabled()) {
    this->get_distconv_adapter().fp_compute();
    return;
  }

#endif // LBANN_HAS_DISTCONV

  const auto& zero = El::TypeTraits<TensorDataType>::Zero();
  const auto& one = El::TypeTraits<TensorDataType>::One();

  // Data tensors
  using LocalMat = El::Matrix<TensorDataType, Device>;
  const auto& linearity = this->weights_values(0);
  const auto& local_input =
    dynamic_cast<const LocalMat&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<LocalMat&>(this->get_local_activations());
  const auto& local_linearity =
    dynamic_cast<const LocalMat&>(linearity.LockedMatrix());

  // Tensor dimensions
  const auto& local_mini_batch_size = local_input.Width();
  const auto& input_dims = this->get_input_dims();
  const auto& output_dims = this->get_output_dims();
  const auto& num_channels = input_dims[0];
  const auto& input_channel_size = std::accumulate(input_dims.begin() + 1,
                                                   input_dims.end(),
                                                   1,
                                                   std::multiplies<size_t>());
  const auto& output_channel_size = std::accumulate(output_dims.begin() + 1,
                                                    output_dims.end(),
                                                    1,
                                                    std::multiplies<size_t>());

  // Reshape input and output tensors
  // Note: [mini_batch_size,num_channels,*] -> [mini_batch_size*num_channels,*]
  LocalMat local_input_reshaped, local_output_reshaped;
  if (local_input.Contiguous()) {
    local_input_reshaped.LockedAttach(input_channel_size,
                                      local_mini_batch_size * num_channels,
                                      local_input.LockedBuffer(),
                                      input_channel_size);
  }
  else {
    El::Copy(local_input, local_input_reshaped);
    local_input_reshaped.Resize(input_channel_size,
                                local_mini_batch_size * num_channels);
  }
  if (local_output.Contiguous()) {
    local_output_reshaped.Attach(output_channel_size,
                                 local_mini_batch_size * num_channels,
                                 local_output.Buffer(),
                                 output_channel_size);
  }
  else {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has a non-contiguous output tensor");
  }
  // Apply linearity
  El::Gemm(m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           one,
           local_linearity,
           local_input_reshaped,
           zero,
           reinterpret_cast<LocalMat&>(local_output_reshaped));

  // Apply bias
  if (m_has_bias) {
    const auto& bias = this->weights_values(1);
    LocalMat ones(local_mini_batch_size * num_channels, 1);
    El::Fill(ones, one);
    El::Gemm(El::NORMAL,
             El::TRANSPOSE,
             one,
             reinterpret_cast<const LocalMat&>(bias.LockedMatrix()),
             ones,
             one,
             reinterpret_cast<LocalMat&>(local_output_reshaped));
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_fully_connected_layer<TensorDataType, Layout, Device>::
  bp_compute()
{
#ifdef LBANN_HAS_DISTCONV

  if (this->distconv_enabled()) {
    this->get_distconv_adapter().bp_compute();

    return;
  }
#endif // LBANN_HAS_DISTCONV

  const auto& zero = El::TypeTraits<TensorDataType>::Zero();
  const auto& one = El::TypeTraits<TensorDataType>::One();

  // Data tensors
  using LocalMat = El::Matrix<TensorDataType, Device>;
  const auto& linearity = this->weights_values(0);
  const auto& local_input =
    dynamic_cast<const LocalMat&>(this->get_local_prev_activations());
  const auto& local_output_grad =
    dynamic_cast<const LocalMat&>(this->get_local_prev_error_signals());
  auto& local_input_grad =
    dynamic_cast<LocalMat&>(this->get_local_error_signals());
  const auto& local_linearity =
    dynamic_cast<const LocalMat&>(linearity.LockedMatrix());

  // Tensor dimensions
  const auto& local_mini_batch_size = local_input.Width();
  const auto& input_dims = this->get_input_dims();
  const auto& output_dims = this->get_output_dims();
  const auto& num_channels = input_dims[0];
  const auto& input_channel_size = std::accumulate(input_dims.begin() + 1,
                                                   input_dims.end(),
                                                   1,
                                                   std::multiplies<size_t>());
  const auto& output_channel_size = std::accumulate(output_dims.begin() + 1,
                                                    output_dims.end(),
                                                    1,
                                                    std::multiplies<size_t>());

  // Reshape input and output tensors
  // Note: [mini_batch_size,num_channels,*] -> [mini_batch_size*num_channels,*]
  LocalMat local_input_reshaped, local_output_grad_reshaped,
    local_input_grad_reshaped;
  if (local_input.Contiguous()) {
    local_input_reshaped.LockedAttach(input_channel_size,
                                      local_mini_batch_size * num_channels,
                                      local_input.LockedBuffer(),
                                      input_channel_size);
  }
  else {
    El::Copy(local_input, local_input_reshaped);
    local_input_reshaped.Resize(input_channel_size,
                                local_mini_batch_size * num_channels);
  }
  if (local_output_grad.Contiguous()) {
    local_output_grad_reshaped.LockedAttach(output_channel_size,
                                            local_mini_batch_size *
                                              num_channels,
                                            local_output_grad.LockedBuffer(),
                                            output_channel_size);
  }
  else {
    El::Copy(local_output_grad, local_output_grad_reshaped);
    local_output_grad_reshaped.Resize(output_channel_size,
                                      local_mini_batch_size * num_channels);
  }
  if (local_input_grad.Contiguous()) {
    local_input_grad_reshaped.Attach(input_channel_size,
                                     local_mini_batch_size * num_channels,
                                     local_input_grad.Buffer(),
                                     input_channel_size);
  }
  else {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has a non-contiguous gradient w.r.t. input tensor");
  }

  // Compute gradient w.r.t. input
  El::Gemm(m_transpose ? El::NORMAL : El::TRANSPOSE,
           El::NORMAL,
           one,
           local_linearity,
           local_output_grad_reshaped,
           zero,
           reinterpret_cast<LocalMat&>(local_input_grad_reshaped));

  // Compute gradient w.r.t. linearity
  auto* linearity_optimizer = this->get_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale, gradient_scale;
    auto& linearity_gradient =
      linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
    if (m_transpose) {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_input_reshaped,
               local_output_grad_reshaped,
               dst_scale,
               linearity_gradient.Matrix());
    }
    else {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_output_grad_reshaped,
               local_input_reshaped,
               dst_scale,
               linearity_gradient.Matrix());
    }
  }

  // Compute gradient w.r.t. bias
  if (m_has_bias) {
    auto& bias_weights = this->get_weights(1);
    auto* bias_optimizer = bias_weights.get_optimizer();
    if (bias_optimizer != nullptr) {
      TensorDataType dst_scale, gradient_scale;
      auto& bias_gradient =
        bias_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
      LocalMat ones(local_mini_batch_size * num_channels, 1);
      El::Fill(ones, one);
      El::Gemv(El::NORMAL,
               gradient_scale,
               local_output_grad_reshaped,
               ones,
               dst_scale,
               bias_gradient.Matrix());
    }
  }
}

// =========================================================
// Builder function
// =========================================================

namespace {

template <typename T, data_layout L, El::Device D>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR("Attempted to construct channelwise_fully_connected_layer ",
                "with invalid parameters ",
                "(TensorDataType=",
                TypeName<T>(),
                ", ",
                "Layout=",
                to_string(L),
                ", ",
                "Device=",
                to_string(D),
                ")");
    return nullptr;
  }
};

template <typename TensorDataType, El::Device Device>
struct Builder<TensorDataType, data_layout::DATA_PARALLEL, Device>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType =
      channelwise_fully_connected_layer<TensorDataType,
                                        data_layout::DATA_PARALLEL,
                                        Device>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_channelwise_fully_connected_layer_from_pbuf(
  lbann_comm* comm,
  lbann_data::Layer const& proto_layer)
{
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, channelwise_fully_connected);
  const auto& params = proto_layer.channelwise_fully_connected();
  std::vector<size_t> output_channel_dims;
  const size_t num_output_channel_dims = params.output_channel_dims_size();
  for (size_t i = 0; i < num_output_channel_dims; ++i) {
    output_channel_dims.push_back(params.output_channel_dims(i));
  }
  const bool has_bias = (params.has_bias() ? params.bias().value() : true);
  const bool transpose =
    (params.has_transpose() ? params.transpose().value() : false);
  return BuilderType::Build(output_channel_dims, has_bias, transpose);
}

// =========================================================
// Explicit template instantiation
// =========================================================

#ifdef LBANN_HAS_DISTCONV

#define PROTO_DEVICE(T, Device)                                                \
  template class channelwise_fully_connected_distconv_adapter<                 \
    T,                                                                         \
    data_layout::DATA_PARALLEL,                                                \
    Device>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif //  LBANN_HAS_DISTCONV

#define PROTO_DEVICE(T, Device)                                                \
  template class channelwise_fully_connected_layer<T,                          \
                                                   data_layout::DATA_PARALLEL, \
                                                   Device>;                    \
  LBANN_LAYER_BUILDER_ETI(channelwise_fully_connected, T, Device)
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
