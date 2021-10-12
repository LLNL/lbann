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

#include <google/protobuf/stubs/port.h>
#define LBANN_CONVOLUTION_LAYER_INSTANTIATE
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/learning/convolution.hpp"

#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/layers.pb.h"

#ifdef LBANN_HAS_ONNX
#include <onnx/onnx_pb.h>
#endif // LBANN_HAS_ONNX

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
convolution_layer<TensorDataType, Layout, Device>::convolution_layer(
  int num_data_dims,
  int num_output_channels,
  int conv_dim,
  int pad,
  int stride,
  int dilation,
  int groups,
  bool has_bias)
  : convolution_layer(num_data_dims,
                      num_output_channels,
                      std::vector<int>(num_data_dims, conv_dim),
                      std::vector<int>(num_data_dims, pad),
                      std::vector<int>(num_data_dims, stride),
                      std::vector<int>(num_data_dims, dilation),
                      groups,
                      has_bias)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
convolution_layer<TensorDataType, Layout, Device>::convolution_layer(
  int num_data_dims,
  int num_output_channels,
  std::vector<int> conv_dims,
  std::vector<int> pads,
  std::vector<int> strides,
  std::vector<int> dilations,
  int groups,
  bool has_bias)
  : base_convolution_layer<TensorDataType, Device>(num_data_dims,
                                                   num_output_channels,
                                                   std::move(conv_dims),
                                                   std::move(pads),
                                                   std::move(strides),
                                                   std::move(dilations),
                                                   groups,
                                                   has_bias)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
convolution_layer<TensorDataType, Layout, Device>::convolution_layer()
  : convolution_layer(0, 0, {}, {}, {}, {}, 0, false)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  base_convolution_layer<TensorDataType, Device>::setup_dims(dr_metadata);

  // Get tensor dimensions
  const auto& input_dims = this->get_input_dims();
  auto output_dims = input_dims;

  // Initialize output tensor dimensions
  output_dims[0] = this->m_output_channels;
  for (size_t i = 0; i < output_dims.size() - 1; ++i) {
    const auto& input_dim = input_dims[i + 1];
    const auto& kernel_dim = this->m_conv_dims[i];
    const auto& stride = this->m_strides[i];
    const auto& pad = this->m_pads[i];
    const auto& dilation = this->m_dilations[i];
    const auto& effective_dim =
      (input_dim + 2 * pad - dilation * (kernel_dim - 1));
    output_dims[i + 1] = (effective_dim + stride - 1) / stride;
  }
  this->set_output_dims(output_dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::vector<int>
convolution_layer<TensorDataType, Layout, Device>::get_kernel_dims() const
{
  std::vector<int> dims;
  dims.push_back(this->m_output_channels);
  dims.push_back(this->get_input_dims()[0] / this->m_groups);
  dims.insert(dims.end(), this->m_conv_dims.begin(), this->m_conv_dims.end());
  return dims;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_layer<TensorDataType, Layout, Device>::fp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("convolution_layer::fp_compute");
  using BaseConvLayer = base_convolution_layer<TensorDataType, Device>;
  if (this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      this->get_distconv_adapter().fp_compute_convolution();
      this->get_distconv_adapter().fp_apply_bias();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    BaseConvLayer::apply_convolution_dnn(true);
    BaseConvLayer::apply_bias_dnn();
  }
  else {
    BaseConvLayer::apply_convolution_im2col(true);
    BaseConvLayer::apply_bias_cpu();
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_layer<TensorDataType, Layout, Device>::bp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("convolution_layer::bp_compute");
  using BaseConvLayer = base_convolution_layer<TensorDataType, Device>;
  if (this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      if (this->get_distconv_adapter()
            .m_conv->is_overlap_bwd_halo_exchange_enabled()) {
        this->get_distconv_adapter().m_conv->backward_data_exchange_halo(
          this->get_distconv_adapter().get_prev_error_signals());
      }
      this->get_distconv_adapter().bp_compute_convolution_filter();
      this->get_distconv_adapter().bp_compute_convolution_data();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    BaseConvLayer::compute_gradients_dnn(false);
    BaseConvLayer::apply_transposed_convolution_dnn(false);
  }
  else {
    BaseConvLayer::compute_gradients_im2col(false);
    BaseConvLayer::apply_transposed_convolution_im2col(false);
  }
}

#ifdef LBANN_HAS_ONNX
template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_layer<TensorDataType, Layout, Device>::fill_onnx_node(
  onnx::GraphProto& graph) const
{
  Layer::fill_onnx_node(graph);
  onnx::NodeProto* conv = nullptr;
  // We need the node whose name matches the layer name. The node
  // *SHOULD* be the most recently added node (i.e., the highest
  // index), but the loop isn't so hard.
  {
    auto const this_name = this->get_name();
    auto const num_nodes = graph.node_size();
    for (int i = num_nodes; i != 0; --i) {
      if (graph.node(i - 1).name() == this_name) {
        conv = graph.mutable_node(i - 1);
        break;
      }
    }
  }
  if (!conv)
    LBANN_ERROR("Bad assumptions about node names.");

  if (!this->m_strides.empty()) {
    auto* strides = conv->add_attribute();
    strides->set_name("strides");
    strides->set_type(onnx::AttributeProto::INTS);
    for (auto const& s : this->m_strides)
      strides->add_ints(s);
  }
  if (!this->m_pads.empty()) {
    auto* pads = conv->add_attribute();
    pads->set_name("pads");
    pads->set_type(onnx::AttributeProto::INTS);
    for (auto const& p : this->m_pads) {
      pads->add_ints(p);
      pads->add_ints(p);
    }
  }
  if (!this->m_dilations.empty()) {
    auto* dilations = conv->add_attribute();
    dilations->set_name("dilations");
    dilations->set_type(onnx::AttributeProto::INTS);
    for (auto const& p : this->m_dilations) {
      dilations->add_ints(p);
    }
  }
  if (this->m_groups > 1) {
    auto* group = conv->add_attribute();
    group->set_name("group");
    group->set_type(onnx::AttributeProto::INT);
    group->set_i(this->m_groups);
  }
}
#endif // LBANN_HAS_ONNX

template <typename T, data_layout L, El::Device D>
void convolution_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_convolution();
  msg->set_num_dims(this->get_conv_dims().size());
  msg->set_out_channels(this->m_output_channels);
  protobuf::assign_to_repeated(*msg->mutable_kernel_size(),
                               this->get_kernel_dims());
  protobuf::assign_to_repeated(*msg->mutable_stride(), this->get_strides());
  protobuf::assign_to_repeated(*msg->mutable_padding(), this->get_pads());
  msg->mutable_groups()->set_value(this->m_groups);
  auto const has_bias = (this->num_weights() > 1UL);
  msg->mutable_has_bias()->set_value(has_bias);
  protobuf::assign_to_repeated(*msg->mutable_dilation(), this->get_dilations());
#ifdef LBANN_HAS_DNN_LIB
  msg->set_conv_tensor_op_mode(
    dnn_lib::convert_to_proto_math_type(this->m_convolution_math_type));
#endif // LBANN_HAS_DNN_LIB
}

#if defined LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_layer<TensorDataType, Layout, Device>::setup_distconv_adapter(
  const DataReaderMetaData& dr_metadata)
{
  this->get_distconv_adapter_ptr() = std::make_unique<
    convolution_distconv_adapter<TensorDataType, Layout, Device>>(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
bool convolution_layer<TensorDataType, Layout, Device>::is_distconv_supported()
  const
{
  const auto& kernel_dims = get_kernel_dims();
  for (int i = 0; i < dc::get_num_spatial_dims(*this); i++) {
    if (kernel_dims[2 + i] != kernel_dims[2]) {
      dc::MPIRootPrintStreamDebug() << "Nonsymmetric kernel not supported";
      return false;
    }
    if (kernel_dims[2 + i] != this->m_pads[i] / this->m_dilations[i] * 2 + 1) {
      dc::MPIRootPrintStreamDebug()
        << "Unsupported as padding does not match the kernel size";
      return false;
    }
  }
  return true;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void convolution_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_distributions(tensor_overlap_constraints& constraints)
{
  base_convolution_adapter<TensorDataType, Dev>::setup_distributions(
    constraints);
  auto& l = dynamic_cast<convolution_layer<TensorDataType, T_layout, Dev>&>(
    this->layer());
  auto kernel_dims = l.get_kernel_dims();
  std::reverse(kernel_dims.begin(), kernel_dims.end());
  auto dilations = l.m_dilations;
  std::reverse(dilations.begin(), dilations.end());
  dc::IntVector overlap(dc::get_num_dims(l), 0);
  const auto& ps = l.get_parallel_strategy();
  // i=0 -> width; i=1 -> height; i=2: -> depth;
  for (int i = 0; i < dc::get_num_spatial_dims(l); i++) {
    int splits = 0;
    switch (i) {
    case 0:
      splits = ps.width_splits;
      break;
    case 1:
      splits = ps.height_splits;
      break;
    case 2:
      splits = ps.depth_splits;
      break;
    }
    if (splits > 1) {
      overlap[i] = (kernel_dims[i] - 1) / 2 * dilations[i];
    }
  }
  auto& prev_activations_dist = this->get_prev_activations_dist();
  prev_activations_dist.set_overlap(overlap);
  constraints.mark_updated(prev_activations_dist);
  constraints.mark_invariant(prev_activations_dist);
  auto& prev_error_signals_dist = this->get_prev_error_signals_dist();
  prev_error_signals_dist.set_overlap(overlap);
  constraints.mark_updated(prev_error_signals_dist);
  constraints.mark_invariant(prev_error_signals_dist);
  // To deal with strides, error signals must have the same size
  // of overlap
  auto& error_signals_dist = this->get_error_signals_dist();
  error_signals_dist.set_overlap(overlap);
  constraints.mark_updated(error_signals_dist);
  constraints.mark_invariant(error_signals_dist);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape convolution_distconv_adapter<TensorDataType, Layout, Device>::
  get_activations_local_shape(int index) const
{
  assert_eq(index, 0);
  const auto& layer =
    dynamic_cast<const convolution_layer<TensorDataType, Layout, Device>&>(
      this->layer());
  auto filter_dims = layer.get_kernel_dims();
  std::reverse(std::begin(filter_dims), std::end(filter_dims));
  auto strides = layer.m_strides;
  std::reverse(std::begin(strides), std::end(strides));
  auto dilations = layer.m_dilations;
  std::reverse(std::begin(dilations), std::end(dilations));
  const auto output_spatial_local_shape =
    ::distconv::get_convolution_output_local_tensor_shape(
      this->get_prev_activations(),
      filter_dims,
      strides,
      true,
      dilations,
      layer.m_groups);
  return output_spatial_local_shape;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_distconv_adapter<TensorDataType, Layout, Device>::setup_layer(
  size_t workspace_capacity)
{
  base_convolution_adapter<TensorDataType, Device>::setup_layer(
    workspace_capacity);
  auto& layer =
    dynamic_cast<convolution_layer<TensorDataType, Layout, Device>&>(
      this->layer());

  if (dc::is_deterministic()) {
    dc::MPIRootPrintStreamDebug()
      << "Using deterministic convolution algorithms";
    this->m_fwd_algo = "DETERMINISTIC";
    this->m_bwd_data_algo = "DETERMINISTIC";
    this->m_bwd_filter_algo = "DETERMINISTIC";
  }
  else {
    this->m_fwd_algo = dc::get_convolution_fwd_algorithm();
    this->m_bwd_data_algo = dc::get_convolution_bwd_data_algorithm();
    this->m_bwd_filter_algo = dc::get_convolution_bwd_filter_algorithm();
  }

  std::vector<int> pads = layer.m_pads;
  std::reverse(pads.begin(), pads.end());
  std::vector<int> strides = layer.m_strides;
  std::reverse(strides.begin(), strides.end());
  std::vector<int> dilations = layer.m_dilations;
  std::reverse(dilations.begin(), dilations.end());

  // Allocate temporary buffer for kernel gradient buffer, if needed
  // Note: Needed for autotuning the convolution algorithm
  El::simple_buffer<TensorDataType, Device> temp;
  TensorDataType* kernel_gradient_buffer =
    this->m_kernel_gradient->get_buffer();
  if (kernel_gradient_buffer == nullptr) {
    temp.allocate(this->m_kernel_gradient->get_local_size());
    assert0(dc::tensor::View(*this->m_kernel_gradient, temp.data()));
  }

  // Setup
  this->m_conv->setup(this->get_prev_activations(),
                      *(this->m_kernel),
                      this->get_activations(),
                      this->get_error_signals(),
                      *this->m_kernel_gradient,
                      this->get_prev_error_signals(),
                      pads,
                      strides,
                      dilations,
                      layer.m_groups,
                      this->m_fwd_algo,
                      this->m_bwd_data_algo,
                      this->m_bwd_filter_algo,
                      workspace_capacity);

  // Clean up
  assert0(dc::tensor::View(*this->m_kernel_gradient, kernel_gradient_buffer));
}
#endif // defined LBANN_HAS_DISTCONV

// Builder helper stuff
namespace {

template <typename TensorDataType, data_layout Layout, El::Device Device>
struct ConvLayerBuilder
{
  static std::unique_ptr<Layer> Build(lbann_data::Layer const& proto_layer)
  {
    auto const& params = proto_layer.convolution();
    int const num_dims = params.num_dims();
    int const num_output_channels = params.out_channels();
    int const num_groups = params.has_groups() ? params.groups().value() : 1;
    bool const bias = params.has_has_bias() ? params.has_bias().value() : true;

    // Fill in a repeated field. If it's empty, a vector of the default
    // value will be used. If it's length 1, the remaining values will
    // be filled in with the value of that entry.
    auto const ensure_dims =
      [&num_dims](auto const& repeated_field,
                  int const default_value) -> std::vector<int> {
      auto vec = repeated_field.size()
                   ? protobuf::to_vector<int>(repeated_field)
                   : std::vector<int>(num_dims, default_value);
      if (vec.size() == 1)
        vec.assign(num_dims, vec.front());
      return vec;
    };

    auto ret =
      std::make_unique<convolution_layer<TensorDataType, Layout, Device>>(
        num_dims,
        num_output_channels,
        ensure_dims(params.kernel_size(), /*NOTUSED=*/-1),
        ensure_dims(params.padding(), /*default=*/0),
        ensure_dims(params.stride(), /*default=*/1),
        ensure_dims(params.dilation(), /*default=*/1),
        num_groups,
        bias);
#ifdef LBANN_HAS_DNN_LIB
    ret->set_dnn_math_mode(
      dnn_lib::convert_to_dnn_math_type(params.conv_tensor_op_mode()));
#endif
    return ret;
  }
};

template <typename TensorDataType, El::Device Device>
struct ConvLayerBuilder<TensorDataType, data_layout::MODEL_PARALLEL, Device>
{
  static std::unique_ptr<Layer> Build(lbann_data::Layer const& proto_layer)
  {
    LBANN_ERROR("convolution layer is only supported with "
                "a data-parallel layout");
  }
};

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_convolution_layer_from_pbuf(lbann_comm* /*comm*/,
                                  const lbann_data::Layer& proto_layer)
{
  using Builder = ConvLayerBuilder<TensorDataType, Layout, Device>;
  return Builder::Build(proto_layer);
}

#define PROTO_DEVICE(T, Device)                                                \
  template class convolution_layer<T, data_layout::DATA_PARALLEL, Device>;     \
  template std::unique_ptr<Layer>                                              \
  build_convolution_layer_from_pbuf<T, data_layout::DATA_PARALLEL, Device>(    \
    lbann_comm*,                                                               \
    lbann_data::Layer const&);                                                 \
  template std::unique_ptr<Layer>                                              \
  build_convolution_layer_from_pbuf<T, data_layout::MODEL_PARALLEL, Device>(   \
    lbann_comm*,                                                               \
    lbann_data::Layer const&)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
