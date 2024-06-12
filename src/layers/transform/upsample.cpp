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
                ".\nThis layer is only "
                "supported with DATA_PARALLEL data layout.");
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
      upsample_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>;
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
    // fp_compute_im2col();
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
    // bp_compute_im2col();
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
  if (local_gradient_wrt_output.Height() > 0 && local_gradient_wrt_output.Width() > 0) {

    // Useful constants
    const auto alpha = El::To<ScalingType>(get_linear_size(m_scale_factors));
    const auto zero = El::TypeTraits<ScalingType>::Zero();

    // Perform backprop on GPU
    dnn_lib::upsample_nearest_backward(m_pooling_dnn_desc,
                                       alpha,
                                       m_tensors_dnn_desc.get_prev_error_signals(),
                                       local_gradient_wrt_output,
                                       zero,
                                       m_tensors_dnn_desc.get_error_signals(),
                                       local_gradient_wrt_input);
  }
#endif // #ifndef LBANN_HAS_DNN_LIB
}

/// Pooling forward propagation with im2col
template <typename TensorDataType, data_layout Layout, El::Device Dev>
void upsample_layer<TensorDataType, Layout, Dev>::fp_compute_im2col()
{
  // if (m_pool_mode != upsample_mode::MAX &&
  //     m_pool_mode != upsample_mode::MAX_DETERMINISTIC &&
  //     m_pool_mode != upsample_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
  //   LBANN_ERROR("CPU upsample layer only supports max and average upsample");
  // }

  // // Local matrices
  // const auto& local_input = this->get_local_prev_activations();
  // auto& local_output = this->get_local_activations();

  // // Pool parameters
  // const int local_width = local_input.Width();
  // const auto& input_dims = this->get_input_dims();
  // const int num_channels = input_dims[0];
  // const int num_per_output_channel = this->get_output_size() / num_channels;

  // // Initialize max pool indices if needed
  // if (m_pool_mode == upsample_mode::MAX ||
  //     m_pool_mode == upsample_mode::MAX_DETERMINISTIC) {
  //   m_max_pool_indices.assign(this->get_output_size() * local_width, 0);
  // }

  // // Initialize matrices
  // El::Matrix<TensorDataType, Dev> im2col_mat(m_pool_size * num_channels,
  //                                            num_per_output_channel);
  // El::Matrix<TensorDataType, Dev> input_mat;

  // // Iterate through data samples
  // for (int sample = 0; sample < local_width; ++sample) {

  //   // Construct im2col matrix from input
  //   El::LockedView(input_mat, local_input, El::ALL, El::IR(sample));
  //   im2col<TensorDataType>(input_mat,
  //                          im2col_mat,
  //                          num_channels,
  //                          input_dims.size() - 1,
  //                          &input_dims[1],
  //                          m_pads.data(),
  //                          m_pool_dims.data(),
  //                          m_strides.data());

  //   if (m_pool_mode == upsample_mode::MAX ||
  //       m_pool_mode == upsample_mode::MAX_DETERMINISTIC) {
  //     // Apply max upsample
  //     TensorDataType* output_buffer = local_output.Buffer(0, sample);
  //     int* indices_buffer =
  //       &m_max_pool_indices[sample * this->get_output_size()];
  //     LBANN_OMP_PARALLEL_FOR
  //     for (int channel = 0; channel < num_channels; ++channel) {
  //       for (int j = 0; j < num_per_output_channel; ++j) {
  //         TensorDataType* im2col_buffer =
  //           im2col_mat.Buffer(channel * m_pool_size, j);
  //         TensorDataType max_entry = im2col_buffer[0];
  //         int max_index = 0;
  //         for (int i = 1; i < m_pool_size; ++i) {
  //           const TensorDataType current_entry = im2col_buffer[i];
  //           if (current_entry > max_entry) {
  //             max_entry = current_entry;
  //             max_index = i;
  //           }
  //         }
  //         const int output_index = j + channel * num_per_output_channel;
  //         output_buffer[output_index] = max_entry;
  //         indices_buffer[output_index] = max_index;
  //       }
  //     }
  //   }

  //   if (m_pool_mode == upsample_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
  //     // Apply average upsample
  //     TensorDataType* output_buffer = local_output.Buffer(0, sample);
  //     LBANN_OMP_PARALLEL_FOR
  //     for (int channel = 0; channel < num_channels; ++channel) {
  //       for (int j = 0; j < num_per_output_channel; ++j) {
  //         const TensorDataType* im2col_buffer =
  //           im2col_mat.LockedBuffer(channel * m_pool_size, j);
  //         TensorDataType output_entry = El::TypeTraits<TensorDataType>::Zero();
  //         for (int i = 0; i < m_pool_size; ++i) {
  //           output_entry += im2col_buffer[i];
  //         }
  //         output_entry /= m_pool_size;
  //         const int output_index = j + channel * num_per_output_channel;
  //         output_buffer[output_index] = output_entry;
  //       }
  //     }
  //   }
  // }
}

/// Pooling forward propagation with im2col
template <typename TensorDataType, data_layout Layout, El::Device Dev>
void upsample_layer<TensorDataType, Layout, Dev>::bp_compute_im2col()
{
  // using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;
  // if (m_pool_mode != upsample_mode::MAX &&
  //     m_pool_mode != upsample_mode::MAX_DETERMINISTIC &&
  //     m_pool_mode != upsample_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
  //   LBANN_ERROR("CPU upsample layer only supports max and average upsample");
  // }

  // // Local matrices
  // const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
  // auto& local_gradient_wrt_input = this->get_local_error_signals();

  // // Pool parameters
  // const int local_width = local_gradient_wrt_output.Width();
  // const auto& input_dims = this->get_input_dims();
  // const int num_channels = input_dims[0];
  // const int num_per_input_channel = this->get_output_size() / num_channels;

  // // Initialize matrices
  // CPUMatType im2col_mat(m_pool_size * num_channels, num_per_input_channel);
  // CPUMatType gradient_wrt_input_col;

  // // Iterate through data samples
  // for (int sample = 0; sample < local_width; ++sample) {

  //   // Compute gradient w.r.t. im2col matrix for max upsample
  //   if (m_pool_mode == upsample_mode::MAX ||
  //       m_pool_mode == upsample_mode::MAX_DETERMINISTIC) {

  //     // Clear im2col matrix
  //     El::Zero(im2col_mat);

  //     // Copy previous error signal to im2col matrix entries
  //     // corresponding to max
  //     const TensorDataType* gradient_wrt_output_buffer =
  //       local_gradient_wrt_output.LockedBuffer(0, sample);
  //     const int* indices_buffer =
  //       &m_max_pool_indices[sample * this->get_output_size()];
  //     LBANN_OMP_PARALLEL_FOR
  //     for (int channel = 0; channel < num_channels; ++channel) {
  //       for (int j = 0; j < num_per_input_channel; ++j) {
  //         const int input_index = j + channel * num_per_input_channel;
  //         const int max_index = indices_buffer[input_index];
  //         TensorDataType* im2col_buffer =
  //           im2col_mat.Buffer(channel * m_pool_size, j);
  //         im2col_buffer[max_index] = gradient_wrt_output_buffer[input_index];
  //       }
  //     }
  //   }

  //   // Compute gradient w.r.t. im2col matrix for average upsample
  //   if (m_pool_mode == upsample_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
  //     const TensorDataType* gradient_wrt_output_buffer =
  //       local_gradient_wrt_output.LockedBuffer(0, sample);
  //     LBANN_OMP_PARALLEL_FOR
  //     for (int channel = 0; channel < num_channels; ++channel) {
  //       for (int j = 0; j < num_per_input_channel; ++j) {
  //         TensorDataType* im2col_buffer =
  //           im2col_mat.Buffer(channel * m_pool_size, j);
  //         const int input_index = j + channel * num_per_input_channel;
  //         const TensorDataType output_entry =
  //           gradient_wrt_output_buffer[input_index] /
  //           El::To<TensorDataType>(m_pool_size);
  //         for (int i = 0; i < m_pool_size; ++i) {
  //           im2col_buffer[i] = output_entry;
  //         }
  //       }
  //     }
  //   }

  //   // Compute error signal (i.e. gradient w.r.t. input)
  //   El::View(gradient_wrt_input_col,
  //            local_gradient_wrt_input,
  //            El::ALL,
  //            El::IR(sample));
  //   col2im<TensorDataType>(im2col_mat,
  //                          gradient_wrt_input_col,
  //                          num_channels,
  //                          input_dims.size() - 1,
  //                          &input_dims[1],
  //                          m_pads.data(),
  //                          m_pool_dims.data(),
  //                          m_strides.data());
  // }
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
bool upsample_layer<TensorDataType, T_layout, Dev>::is_distconv_supported() const
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
  auto output_spatial_local_shape = this->get_prev_activations(index).get_local_shape();
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

  auto& l =
    dynamic_cast<upsample_layer<TensorDataType, Layout, Device>&>(this->layer());

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
  auto& l =
    dynamic_cast<upsample_layer<TensorDataType, Layout, Device>&>(this->layer());

  auto& prev_activations = this->get_prev_activations();
  auto& activations = this->get_activations();

  auto xdesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_xdesc.get());
  auto ydesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_ydesc.get());
  dc_backend::setup_tensor_descriptor(xdesc, prev_activations,
                                      prev_activations.get_local_shape());
  dc_backend::setup_tensor_descriptor(ydesc, activations,
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
  auto& l =
    dynamic_cast<upsample_layer<TensorDataType, Layout, Device>&>(this->layer());

  auto& prev_error_signals = this->get_prev_error_signals();
  auto& error_signals = this->get_error_signals();

  auto dxdesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_dxdesc.get());
  auto dydesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_dydesc.get());
  dc_backend::setup_tensor_descriptor(dxdesc, error_signals,
                                      error_signals.get_local_shape());
  dc_backend::setup_tensor_descriptor(dydesc, prev_error_signals,
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
  template class upsample_layer<T, data_layout::DATA_PARALLEL, Device>;         \
  LBANN_LAYER_BUILDER_ETI(upsample, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
