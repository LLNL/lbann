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

#include "lbann/utils/protobuf/decl.hpp"
#include <google/protobuf/repeated_field.h>
#define LBANN_CONVOLUTION_LAYER_INSTANTIATE
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/learning/deconvolution.hpp"

#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/layers.pb.h"

#include <sstream>
#include <string>

namespace lbann {

// =========================================================
// Layer member functions
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
deconvolution_layer<TensorDataType, Layout, Device>::deconvolution_layer(
  int num_data_dims,
  int num_output_channels,
  std::vector<int> conv_dims,
  std::vector<int> pads,
  std::vector<int> output_pads,
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
                                                   has_bias),
    m_output_pads{std::move(output_pads)}
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
deconvolution_layer<TensorDataType, Layout, Device>::deconvolution_layer()
  : deconvolution_layer(0, 0, {}, {}, {}, {}, {}, 0, false)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void deconvolution_layer<TensorDataType, Layout, Device>::setup_dims()
{
  base_convolution_layer<TensorDataType, Device>::setup_dims();

  // Check for unsupported features
  /// @todo Implement dilated and grouped deconvolution
  if (std::any_of(this->m_dilations.begin(),
                  this->m_dilations.end(),
                  [](int d) { return d != 1; })) {
    std::ostringstream ss;
    for (size_t i = 0; i < this->m_dilations.size(); ++i) {
      ss << (i > 0 ? "," : "") << this->m_dilations[i];
    }
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has non-unit dilations (",
                ss.str(),
                ")");
  }
  if (this->m_groups != 1) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has non-unit groups (",
                this->m_groups,
                ")");
  }

  // Check that output tensor padding is valid
  const size_t num_dims = this->m_conv_dims.size();
  if (this->m_output_pads.empty()) {
    this->m_output_pads.assign(num_dims, 0);
  }
  if (this->m_output_pads.size() == 1) {
    this->m_output_pads.assign(num_dims, this->m_output_pads.front());
  }
  if (this->m_output_pads.size() != num_dims) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has an invalid number of output pads ",
                "(expected ",
                num_dims,
                ", found ",
                this->m_output_pads.size(),
                ")");
  }

  // Compute output tensor dimensions
  const auto& input_dims = this->get_input_dims();
  auto output_dims = input_dims;
  output_dims[0] = this->m_output_channels;
  for (size_t i = 0; i < output_dims.size() - 1; ++i) {

    // Compute output dim
    /// @todo Dilated deconvolution
    const auto& input_dim = input_dims[i + 1];
    const auto& kernel_dim = this->m_conv_dims[i];
    const auto& stride = this->m_strides[i];
    const auto& pad = this->m_pads[i];
    const auto& dilation = this->m_dilations[i];
    const auto& output_pad = this->m_output_pads[i];
    const int output_dim =
      (input_dim - 1) * stride + kernel_dim - 2 * pad + output_pad;
    output_dims[i + 1] = output_dim;

    // Check that output dim is valid
    const int effective_output_dim =
      output_dim + 2 * pad - dilation * (kernel_dim - 1);
    const int expected_input_dim = (effective_output_dim + stride - 1) / stride;
    if (output_dim <= 0 || expected_input_dim != input_dim) {
      LBANN_ERROR(this->get_type(),
                  " layer \"",
                  this->get_name(),
                  "\" ",
                  "has invalid convolution parameters along dim ",
                  i + 1,
                  " ",
                  "(input dim = ",
                  input_dim,
                  ", kernel dim = ",
                  kernel_dim,
                  ", ",
                  "stride = ",
                  stride,
                  ", pad = ",
                  pad,
                  ", dilation = ",
                  dilation,
                  " ",
                  "output pad = ",
                  output_pad,
                  ", ",
                  "computed output dim = ",
                  output_dim,
                  ")");
    }
  }
  this->set_output_dims(output_dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::vector<int>
deconvolution_layer<TensorDataType, Layout, Device>::get_kernel_dims() const
{
  std::vector<int> dims;
  dims.push_back(this->get_input_dims()[0]);
  dims.push_back(this->m_output_channels);
  dims.insert(dims.end(), this->m_conv_dims.begin(), this->m_conv_dims.end());
  return dims;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void deconvolution_layer<TensorDataType, Layout, Device>::fp_compute()
{
  using BaseConvLayer = base_convolution_layer<TensorDataType, Device>;
  if (this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      this->get_distconv_adapter().fp_compute_convolution();
      this->get_distconv_adapter().fp_apply_bias();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    BaseConvLayer::apply_transposed_convolution_dnn(true);
    BaseConvLayer::apply_bias_dnn();
  }
  else {
    BaseConvLayer::apply_transposed_convolution_im2col(true);
    BaseConvLayer::apply_bias_cpu();
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void deconvolution_layer<TensorDataType, Layout, Device>::bp_compute()
{
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
    BaseConvLayer::compute_gradients_dnn(true);
    BaseConvLayer::apply_convolution_dnn(false);
  }
  else {
    BaseConvLayer::compute_gradients_im2col(true);
    BaseConvLayer::apply_convolution_im2col(false);
  }
}

#if defined LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout Layout, El::Device Device>
bool deconvolution_layer<TensorDataType, Layout, Device>::
  is_distconv_supported() const
{
  const auto& kernel_dims = get_kernel_dims();
  for (int i = 0; i < dc::get_num_spatial_dims(*this); i++) {
    auto pad = this->m_pads[i];
    if (pad != 0) {
      dc::MPIPrintStreamDebug()
        << this->get_name() << " unsupported as padding must be zero";
      return false;
    }
    auto stride_size = this->m_strides[i];
    auto filter_size = kernel_dims[2 + i];
    if (!(filter_size % 2 == 0 && filter_size == stride_size)) {
      dc::MPIPrintStreamDebug()
        << this->get_name() << " unsupported due to filter and stride sizes";
      return false;
    }
  }
  return true;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void deconvolution_layer<TensorDataType, Layout, Device>::
  setup_distconv_adapter()
{
  this->get_distconv_adapter_ptr() = std::make_unique<
    deconvolution_distconv_adapter<TensorDataType, Layout, Device>>(*this);
}

#endif // LBANN_HAS_DISTCONV

// =========================================================
// Distconv adapter member functions
// =========================================================

#ifdef LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void deconvolution_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_distributions(tensor_overlap_constraints& constraints)
{
  base_convolution_adapter<TensorDataType, Dev>::setup_distributions(
    constraints);

  // Assumes zero halo all tensor for now
  // prev activations
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
dc::Shape deconvolution_distconv_adapter<TensorDataType, Layout, Device>::
  get_activations_local_shape(int index) const
{
  assert_eq(index, 0);
  const auto& layer =
    dynamic_cast<const deconvolution_layer<TensorDataType, Layout, Device>&>(
      this->layer());
  auto filter_dims = layer.get_kernel_dims();
  std::reverse(std::begin(filter_dims), std::end(filter_dims));
  auto strides = layer.m_strides;
  std::reverse(std::begin(strides), std::end(strides));
  auto dilations = layer.m_dilations;
  std::reverse(std::begin(dilations), std::end(dilations));
  const auto output_spatial_local_shape =
    ::distconv::get_deconvolution_output_local_tensor_shape(
      this->get_prev_activations(),
      filter_dims,
      strides,
      false,
      dilations,
      layer.m_groups);
  return output_spatial_local_shape;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void deconvolution_distconv_adapter<TensorDataType, Layout, Device>::
  setup_layer(size_t workspace_capacity)
{
  base_convolution_adapter<TensorDataType, Device>::setup_layer(
    workspace_capacity);
  auto& layer =
    dynamic_cast<deconvolution_layer<TensorDataType, Layout, Device>&>(
      this->layer());

  if (dc::is_deterministic()) {
    dc::MPIRootPrintStreamDebug()
      << "Using deterministic convolution algorithms";
    this->m_fwd_algo = "DETERMINISTIC";
    this->m_bwd_data_algo = "DETERMINISTIC";
    this->m_bwd_filter_algo = "DETERMINISTIC";
  }
  else {
    this->m_fwd_algo = dc::get_convolution_bwd_data_algorithm();
    this->m_bwd_data_algo = dc::get_convolution_fwd_algorithm();
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
                      workspace_capacity,
                      false,
                      true);

  // Clean up
  assert0(dc::tensor::View(*this->m_kernel_gradient, kernel_gradient_buffer));
}

#endif // LBANN_HAS_DISTCONV

// =========================================================
// Builder function
// =========================================================

namespace {

template <typename T, data_layout L, El::Device D>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(lbann_data::Layer const&, Args&&...)
  {
    LBANN_ERROR("Attempted to instantiate layer \"deconvolution\""
                "with Layout=",
                to_string(L),
                ".\nThis layer is only "
                "supported with DATA_PARALLEL data layout.");
    return nullptr;
  }
};

template <typename TensorDataType, El::Device Device>
struct Builder<TensorDataType, data_layout::DATA_PARALLEL, Device>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(lbann_data::Layer const& proto_layer,
                                      Args&&... args)
  {
    using LayerType =
      deconvolution_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>;
    auto ret = make_unique<LayerType>(std::forward<Args>(args)...);
#ifdef LBANN_HAS_DNN_LIB
    const auto& params = proto_layer.deconvolution();
    ret->set_dnn_math_mode(
      dnn_lib::convert_to_dnn_math_type(params.conv_tensor_op_mode()));
#endif // LBANN_HAS_DNN_LIB
    return ret;
  }
};

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_deconvolution_layer_from_pbuf(lbann_comm* comm,
                                    lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, deconvolution);
  const auto& params = proto_layer.deconvolution();

  int const num_dims = params.num_dims();
  int const out_channels = params.out_channels();
  int const groups = params.has_groups() ? params.groups().value() : 1;
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

  // Kernel dimensions must be given - NO default. Must exist.
  LBANN_ASSERT(params.kernel_size_size() > 0);

  // Construct layer
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  return BuilderType::Build(proto_layer,
                            num_dims,
                            out_channels,
                            ensure_dims(params.kernel_size(), /*NOTUSED=*/-1),
                            ensure_dims(params.padding(), /*default=*/0),
                            ensure_dims(params.output_padding(), /*default=*/0),
                            ensure_dims(params.stride(), /*default=*/1),
                            ensure_dims(params.dilation(), /*default=*/1),
                            groups,
                            bias);
}

template <typename T, data_layout L, El::Device D>
void deconvolution_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_deconvolution();
  msg->set_num_dims(this->get_conv_dims().size());
  msg->set_out_channels(this->m_output_channels);
  protobuf::assign_to_repeated(*msg->mutable_kernel_size(),
                               this->get_conv_dims());
  protobuf::assign_to_repeated(*msg->mutable_stride(), this->get_strides());
  protobuf::assign_to_repeated(*msg->mutable_padding(), this->get_pads());
  protobuf::assign_to_repeated(*msg->mutable_output_padding(),
                               this->m_output_pads);
  msg->mutable_groups()->set_value(this->m_groups);
  auto const has_bias = (this->num_weights() > 1UL);
  msg->mutable_has_bias()->set_value(has_bias);
  protobuf::assign_to_repeated(*msg->mutable_dilation(), this->get_dilations());
#ifdef LBANN_HAS_DNN_LIB
  msg->set_conv_tensor_op_mode(
    dnn_lib::convert_to_proto_math_type(this->m_convolution_math_type));
#endif // LBANN_HAS_DNN_LIB
}

// =========================================================
// Explicit template instantiation
// =========================================================

#define PROTO_DEVICE(T, Device)                                                \
  template class deconvolution_layer<T, data_layout::DATA_PARALLEL, Device>;   \
  LBANN_LAYER_BUILDER_ETI(deconvolution, T, Device)
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
