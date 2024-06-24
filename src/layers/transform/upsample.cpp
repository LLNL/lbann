////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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

#define LBANN_UPSAMPLE_LAYER_INSTANTIATE
#include "lbann/layers/transform/upsample.hpp"

#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
using dc_backend = ::distconv::GPUDNNBackend;
#endif // LBANN_HAS_DISTCONV

#include "lbann/proto/layers.pb.h"
#include "lbann/proto/lbann.pb.h"

namespace lbann {
namespace {

template <typename T, data_layout L, El::Device D>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR("Attempted to instantiate layer \"upsample\" with "
                "Layout=",
                to_string(L),
                ", Device=",
                El::DeviceName<D>(),
                ".\nThis layer is only "
                "supported on GPU with DATA_PARALLEL data layout.");
    return nullptr;
  }
};

template <typename TensorDataType>
struct Builder<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType = upsample_layer<TensorDataType,
                                     data_layout::DATA_PARALLEL,
                                     El::Device::GPU>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};
} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void upsample_layer<TensorDataType, Layout, Device>::fp_compute()
{
  if (this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      const auto& mode =
        this->m_model->get_execution_context().get_execution_mode();
      get_distconv_adapter().fp_compute(mode == execution_mode::training);
      return;
    }
#endif // LBANN_HAS_DISTCONV
    fp_compute_dnn();
  }
  else {
    LBANN_ERROR("Upsampling with CPU is not implemented.");
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void upsample_layer<TensorDataType, Layout, Device>::bp_compute()
{
  if (this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      get_distconv_adapter().bp_compute();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    bp_compute_dnn();
  }
  else {
    LBANN_ERROR("Upsampling with CPU is not implemented.");
  }
}

/// Pooling forward propagation with DNN library
template <typename TensorDataType, data_layout Layout, El::Device Device>
void upsample_layer<TensorDataType, Layout, Device>::fp_compute_dnn()
{
#ifndef LBANN_HAS_DNN_LIB
  LBANN_ERROR("DNN library not detected");
#else
  using ScalingType = dnn_lib::ScalingParamType<TensorDataType>;
  const auto& local_input = this->get_local_prev_activations();
  auto& local_output = this->get_local_activations();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
    const auto zero = El::TypeTraits<ScalingType>::Zero();
    const auto alpha = El::To<ScalingType>(get_linear_size(m_scale_factors));
    dnn_lib::upsample_nearest_forward(m_pooling_dnn_desc,
                                      alpha,
                                      m_tensors_dnn_desc.get_prev_activations(),
                                      local_input,
                                      zero,
                                      m_tensors_dnn_desc.get_activations(),
                                      local_output);
  }
#endif // #ifndef LBANN_HAS_DNN_LIB
}

/// Pooling backward propagation with DNN library
template <typename TensorDataType, data_layout Layout, El::Device Device>
void upsample_layer<TensorDataType, Layout, Device>::bp_compute_dnn()
{
#ifndef LBANN_HAS_DNN_LIB
  LBANN_ERROR("DNN library not detected");
#else
  using ScalingType = dnn_lib::ScalingParamType<TensorDataType>;
  const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
  auto& local_gradient_wrt_input = this->get_local_error_signals();
  if (local_gradient_wrt_output.Height() > 0 &&
      local_gradient_wrt_output.Width() > 0) {

    // Useful constants
    const auto alpha = El::To<ScalingType>(get_linear_size(m_scale_factors));
    const auto zero = El::TypeTraits<ScalingType>::Zero();

    // Perform backprop on GPU
    dnn_lib::upsample_nearest_backward(
      m_pooling_dnn_desc,
      alpha,
      m_tensors_dnn_desc.get_prev_error_signals(),
      local_gradient_wrt_output,
      zero,
      m_tensors_dnn_desc.get_error_signals(),
      local_gradient_wrt_input);
  }
#endif // #ifndef LBANN_HAS_DNN_LIB
}

template <typename T, data_layout L, El::Device D>
void upsample_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_upsample();
  switch (m_upsample_mode) {
  case upsample_mode::NEAREST:
    msg->set_upsample_mode("nearest");
    break;
  default:
    LBANN_ERROR("Invalid upsample mode requested.");
  }
  msg->set_num_dims(m_scale_factors.size());
  msg->set_has_vectors(true);
  protobuf::assign_to_repeated(*msg->mutable_scale_factors(), m_scale_factors);
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
upsample_distconv_adapter<TensorDataType, T_layout, Dev>&
upsample_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<upsample_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<const upsample_layer<TensorDataType, T_layout, Dev>&>(*this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const upsample_distconv_adapter<TensorDataType, T_layout, Dev>&
upsample_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const
{
  return dynamic_cast<
    const upsample_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
bool upsample_layer<TensorDataType, T_layout, Dev>::is_distconv_supported()
  const
{
  return Dev == El::Device::GPU && T_layout == data_layout::DATA_PARALLEL;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape upsample_distconv_adapter<TensorDataType, Layout, Device>::
  get_activations_local_shape(int index) const
{
  assert_eq(index, 0);
  const auto& layer =
    dynamic_cast<const upsample_layer<TensorDataType, Layout, Device>&>(
      this->layer());
  auto scale_factors = layer.m_scale_factors;
  std::reverse(std::begin(scale_factors), std::end(scale_factors));
  auto output_spatial_local_shape =
    this->get_prev_activations(index).get_local_shape();
  for (size_t i = 0; i < scale_factors.size(); i++) {
    output_spatial_local_shape[i] *= scale_factors[i];
  }
  return output_spatial_local_shape;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void upsample_distconv_adapter<TensorDataType, Layout, Device>::setup_layer(
  size_t workspace_capacity)
{
  m_xdesc.create();
  m_ydesc.create();
  m_dxdesc.create();
  m_dydesc.create();

  auto& l = dynamic_cast<upsample_layer<TensorDataType, Layout, Device>&>(
    this->layer());

  std::string mode;
  switch (l.m_upsample_mode) {
  case upsample_mode::NEAREST:
    mode = "nearest";
    break;
  default:
    LBANN_ERROR("upsample_layer: no DISTCONV implementation for upsample mode");
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void upsample_distconv_adapter<TensorDataType, Layout, Device>::fp_compute(
  bool const training)
{
  auto& l = dynamic_cast<upsample_layer<TensorDataType, Layout, Device>&>(
    this->layer());

  auto& prev_activations = this->get_prev_activations();
  auto& activations = this->get_activations();

  auto xdesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_xdesc.get());
  auto ydesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_ydesc.get());
  dc_backend::setup_tensor_descriptor(xdesc,
                                      prev_activations,
                                      prev_activations.get_local_shape());
  dc_backend::setup_tensor_descriptor(ydesc,
                                      activations,
                                      activations.get_local_shape());

  using ScalingType = dnn_lib::ScalingParamType<TensorDataType>;
  const auto zero = El::TypeTraits<ScalingType>::Zero();
  const auto alpha = El::To<ScalingType>(get_linear_size(l.m_scale_factors));

  dnn_lib::upsample_nearest_forward(l.m_pooling_dnn_desc,
                                    alpha,
                                    m_xdesc,
                                    prev_activations.get_const_base_ptr(),
                                    zero,
                                    m_ydesc,
                                    activations.get_base_ptr(),
                                    dc::get_backend().get_handle());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void upsample_distconv_adapter<TensorDataType, Layout, Device>::bp_compute()
{
  auto& l = dynamic_cast<upsample_layer<TensorDataType, Layout, Device>&>(
    this->layer());

  auto& prev_error_signals = this->get_prev_error_signals();
  auto& error_signals = this->get_error_signals();

  auto dxdesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_dxdesc.get());
  auto dydesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_dydesc.get());
  dc_backend::setup_tensor_descriptor(dxdesc,
                                      error_signals,
                                      error_signals.get_local_shape());
  dc_backend::setup_tensor_descriptor(dydesc,
                                      prev_error_signals,
                                      prev_error_signals.get_local_shape());

  using ScalingType = dnn_lib::ScalingParamType<TensorDataType>;
  const auto zero = El::TypeTraits<ScalingType>::Zero();
  const auto alpha = El::To<ScalingType>(get_linear_size(l.m_scale_factors));

  dnn_lib::upsample_nearest_backward(l.m_pooling_dnn_desc,
                                     alpha,
                                     m_dydesc,
                                     prev_error_signals.get_const_base_ptr(),
                                     zero,
                                     m_dxdesc,
                                     error_signals.get_base_ptr(),
                                     dc::get_backend().get_handle());
}
#endif // LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_upsample_layer_from_pbuf(lbann_comm* comm,
                               lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, upsample);

  using BuilderType = Builder<TensorDataType, Layout, Device>;
  const auto& params = proto_layer.upsample();
  upsample_mode const mode = to_upsample_mode(params.upsample_mode());
  if (params.has_vectors()) {
    return BuilderType::Build(comm,
                              params.scale_factors_size(),
                              protobuf::to_vector<int>(params.scale_factors()),
                              mode);
  }
  else {
    return BuilderType::Build(comm,
                              params.num_dims(),
                              params.scale_factors_i(),
                              mode);
  }
}

#define PROTO_DEVICE(T, Device)                                                \
  template class upsample_layer<T, data_layout::DATA_PARALLEL, Device>;        \
  LBANN_LAYER_BUILDER_ETI(upsample, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
