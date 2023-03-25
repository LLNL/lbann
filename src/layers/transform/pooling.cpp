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

#define LBANN_POOLING_LAYER_INSTANTIATE
#include "lbann/layers/transform/pooling.hpp"

#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
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
    LBANN_ERROR("Attempted to instantiate layer \"pooling\" with "
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
      pooling_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};
} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_layer<TensorDataType, Layout, Device>::fp_compute()
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
    fp_compute_im2col();
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_layer<TensorDataType, Layout, Device>::bp_compute()
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
    bp_compute_im2col();
  }
}

/// Pooling forward propagation with DNN library
template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_layer<TensorDataType, Layout, Device>::fp_compute_dnn()
{
#ifndef LBANN_HAS_DNN_LIB
  LBANN_ERROR("DNN library not detected");
#else
  // Initialize GPU workspace
  El::Matrix<TensorDataType, El::Device::GPU> workspace;
  size_t workspace_size =
    dnn_lib::get_pooling_ws_size(m_pooling_dnn_desc,
                                 m_tensors_dnn_desc.get_activations());
  workspace.Resize(workspace_size / sizeof(TensorDataType), 1);

  using ScalingType = dnn_lib::ScalingParamType<TensorDataType>;
  const auto& local_input = this->get_local_prev_activations();
  auto& local_output = this->get_local_activations();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
    const auto zero = El::TypeTraits<ScalingType>::Zero();
    const auto one = El::TypeTraits<ScalingType>::One();
    dnn_lib::pooling_forward(m_pooling_dnn_desc,
                             one,
                             m_tensors_dnn_desc.get_prev_activations(),
                             local_input,
                             zero,
                             m_tensors_dnn_desc.get_activations(),
                             local_output,
                             workspace);
  }
#endif // #ifndef LBANN_HAS_DNN_LIB
}

/// Pooling backward propagation with DNN library
template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_layer<TensorDataType, Layout, Device>::bp_compute_dnn()
{
#ifndef LBANN_HAS_DNN_LIB
  LBANN_ERROR("DNN library not detected");
#else
  // Initialize GPU workspace
  El::Matrix<TensorDataType, El::Device::GPU> workspace;
  size_t workspace_size =
    dnn_lib::get_pooling_ws_size(m_pooling_dnn_desc,
                                 m_tensors_dnn_desc.get_activations());
  workspace.Resize(workspace_size / sizeof(TensorDataType), 1);

  using ScalingType = dnn_lib::ScalingParamType<TensorDataType>;
  const auto& local_input = this->get_local_prev_activations();
  const auto& local_output = this->get_local_activations();
  const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
  auto& local_gradient_wrt_input = this->get_local_error_signals();
  if (local_input.Height() > 0 && local_input.Width() > 0) {

    // Useful constants
    const auto one = El::TypeTraits<ScalingType>::One();
    const auto zero = El::TypeTraits<ScalingType>::Zero();

    // Perform backprop on GPU
    dnn_lib::pooling_backward(m_pooling_dnn_desc,
                              one,
                              m_tensors_dnn_desc.get_activations(),
                              local_output,
                              m_tensors_dnn_desc.get_prev_error_signals(),
                              local_gradient_wrt_output,
                              m_tensors_dnn_desc.get_prev_activations(),
                              local_input,
                              zero,
                              m_tensors_dnn_desc.get_error_signals(),
                              local_gradient_wrt_input,
                              workspace);
  }
#endif // #ifndef LBANN_HAS_DNN_LIB
}

/// Pooling forward propagation with im2col
template <typename TensorDataType, data_layout Layout, El::Device Dev>
void pooling_layer<TensorDataType, Layout, Dev>::fp_compute_im2col()
{
  if (m_pool_mode != pooling_mode::MAX &&
      m_pool_mode != pooling_mode::MAX_DETERMINISTIC &&
      m_pool_mode != pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
    LBANN_ERROR("CPU pooling layer only supports max and average pooling");
  }

  // Local matrices
  const auto& local_input = this->get_local_prev_activations();
  auto& local_output = this->get_local_activations();

  // Pool parameters
  const int local_width = local_input.Width();
  const auto& input_dims = this->get_input_dims();
  const int num_channels = input_dims[0];
  const int num_per_output_channel = this->get_output_size() / num_channels;

  // Initialize max pool indices if needed
  if (m_pool_mode == pooling_mode::MAX ||
      m_pool_mode == pooling_mode::MAX_DETERMINISTIC) {
    m_max_pool_indices.assign(this->get_output_size() * local_width, 0);
  }

  // Initialize matrices
  El::Matrix<TensorDataType, Dev> im2col_mat(m_pool_size * num_channels,
                                             num_per_output_channel);
  El::Matrix<TensorDataType, Dev> input_mat;

  // Iterate through data samples
  for (int sample = 0; sample < local_width; ++sample) {

    // Construct im2col matrix from input
    El::LockedView(input_mat, local_input, El::ALL, El::IR(sample));
    im2col<TensorDataType>(input_mat,
                           im2col_mat,
                           num_channels,
                           input_dims.size() - 1,
                           &input_dims[1],
                           m_pads.data(),
                           m_pool_dims.data(),
                           m_strides.data());

    if (m_pool_mode == pooling_mode::MAX ||
        m_pool_mode == pooling_mode::MAX_DETERMINISTIC) {
      // Apply max pooling
      TensorDataType* output_buffer = local_output.Buffer(0, sample);
      int* indices_buffer =
        &m_max_pool_indices[sample * this->get_output_size()];
      LBANN_OMP_PARALLEL_FOR
      for (int channel = 0; channel < num_channels; ++channel) {
        for (int j = 0; j < num_per_output_channel; ++j) {
          TensorDataType* im2col_buffer =
            im2col_mat.Buffer(channel * m_pool_size, j);
          TensorDataType max_entry = im2col_buffer[0];
          int max_index = 0;
          for (int i = 1; i < m_pool_size; ++i) {
            const TensorDataType current_entry = im2col_buffer[i];
            if (current_entry > max_entry) {
              max_entry = current_entry;
              max_index = i;
            }
          }
          const int output_index = j + channel * num_per_output_channel;
          output_buffer[output_index] = max_entry;
          indices_buffer[output_index] = max_index;
        }
      }
    }

    if (m_pool_mode == pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
      // Apply average pooling
      TensorDataType* output_buffer = local_output.Buffer(0, sample);
      LBANN_OMP_PARALLEL_FOR
      for (int channel = 0; channel < num_channels; ++channel) {
        for (int j = 0; j < num_per_output_channel; ++j) {
          const TensorDataType* im2col_buffer =
            im2col_mat.LockedBuffer(channel * m_pool_size, j);
          TensorDataType output_entry = El::TypeTraits<TensorDataType>::Zero();
          for (int i = 0; i < m_pool_size; ++i) {
            output_entry += im2col_buffer[i];
          }
          output_entry /= m_pool_size;
          const int output_index = j + channel * num_per_output_channel;
          output_buffer[output_index] = output_entry;
        }
      }
    }
  }
}

/// Pooling forward propagation with im2col
template <typename TensorDataType, data_layout Layout, El::Device Dev>
void pooling_layer<TensorDataType, Layout, Dev>::bp_compute_im2col()
{
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;
  if (m_pool_mode != pooling_mode::MAX &&
      m_pool_mode != pooling_mode::MAX_DETERMINISTIC &&
      m_pool_mode != pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
    LBANN_ERROR("CPU pooling layer only supports max and average pooling");
  }

  // Local matrices
  const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
  auto& local_gradient_wrt_input = this->get_local_error_signals();

  // Pool parameters
  const int local_width = local_gradient_wrt_output.Width();
  const auto& input_dims = this->get_input_dims();
  const int num_channels = input_dims[0];
  const int num_per_input_channel = this->get_output_size() / num_channels;

  // Initialize matrices
  CPUMatType im2col_mat(m_pool_size * num_channels, num_per_input_channel);
  CPUMatType gradient_wrt_input_col;

  // Iterate through data samples
  for (int sample = 0; sample < local_width; ++sample) {

    // Compute gradient w.r.t. im2col matrix for max pooling
    if (m_pool_mode == pooling_mode::MAX ||
        m_pool_mode == pooling_mode::MAX_DETERMINISTIC) {

      // Clear im2col matrix
      El::Zero(im2col_mat);

      // Copy previous error signal to im2col matrix entries
      // corresponding to max
      const TensorDataType* gradient_wrt_output_buffer =
        local_gradient_wrt_output.LockedBuffer(0, sample);
      const int* indices_buffer =
        &m_max_pool_indices[sample * this->get_output_size()];
      LBANN_OMP_PARALLEL_FOR
      for (int channel = 0; channel < num_channels; ++channel) {
        for (int j = 0; j < num_per_input_channel; ++j) {
          const int input_index = j + channel * num_per_input_channel;
          const int max_index = indices_buffer[input_index];
          TensorDataType* im2col_buffer =
            im2col_mat.Buffer(channel * m_pool_size, j);
          im2col_buffer[max_index] = gradient_wrt_output_buffer[input_index];
        }
      }
    }

    // Compute gradient w.r.t. im2col matrix for average pooling
    if (m_pool_mode == pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
      const TensorDataType* gradient_wrt_output_buffer =
        local_gradient_wrt_output.LockedBuffer(0, sample);
      LBANN_OMP_PARALLEL_FOR
      for (int channel = 0; channel < num_channels; ++channel) {
        for (int j = 0; j < num_per_input_channel; ++j) {
          TensorDataType* im2col_buffer =
            im2col_mat.Buffer(channel * m_pool_size, j);
          const int input_index = j + channel * num_per_input_channel;
          const TensorDataType output_entry =
            gradient_wrt_output_buffer[input_index] /
            El::To<TensorDataType>(m_pool_size);
          for (int i = 0; i < m_pool_size; ++i) {
            im2col_buffer[i] = output_entry;
          }
        }
      }
    }

    // Compute error signal (i.e. gradient w.r.t. input)
    El::View(gradient_wrt_input_col,
             local_gradient_wrt_input,
             El::ALL,
             El::IR(sample));
    col2im<TensorDataType>(im2col_mat,
                           gradient_wrt_input_col,
                           num_channels,
                           input_dims.size() - 1,
                           &input_dims[1],
                           m_pads.data(),
                           m_pool_dims.data(),
                           m_strides.data());
  }
}

template <typename T, data_layout L, El::Device D>
void pooling_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_pooling();
  switch (m_pool_mode) {
  case pooling_mode::MAX_DETERMINISTIC:
    msg->set_pool_mode("max");
    break;
  case pooling_mode::MAX:
    msg->set_pool_mode("max");
    break;
  case pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING:
    msg->set_pool_mode("average");
    break;
  case pooling_mode::AVERAGE_COUNT_EXCLUDE_PADDING:
    msg->set_pool_mode("average_no_pad");
    break;
  default:
    LBANN_ERROR("Invalid pooling mode requested.");
  }
  msg->set_num_dims(m_pool_dims.size());
  msg->set_has_vectors(true);
  protobuf::assign_to_repeated(*msg->mutable_pool_dims(), m_pool_dims);
  protobuf::assign_to_repeated(*msg->mutable_pool_pads(), m_pads);
  protobuf::assign_to_repeated(*msg->mutable_pool_strides(), m_strides);
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
pooling_distconv_adapter<TensorDataType, T_layout, Dev>&
pooling_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<pooling_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<const pooling_layer<TensorDataType, T_layout, Dev>&>(*this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const pooling_distconv_adapter<TensorDataType, T_layout, Dev>&
pooling_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const
{
  return dynamic_cast<
    const pooling_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
bool pooling_layer<TensorDataType, T_layout, Dev>::is_distconv_supported() const
{
  if (Dev != El::Device::GPU || T_layout != data_layout::DATA_PARALLEL) {
    return false;
  }

  bool cond = true;
  for (int i = 0; i < dc::get_num_spatial_dims(*this); i++) {
    cond &= (m_pool_dims[i] % 2 != 0) || (m_pool_dims[i] == m_strides[i]);
  }
  if (!cond) {
    dc::MPIPrintStreamDebug() << "pooling: unsupported due to window shape: "
                              << dc::util::join_xd_array(m_pool_dims);
    return false;
  }

  for (int i = 0; i < dc::get_num_spatial_dims(*this); i++) {
    bool odd = m_pool_dims[i] % 2;
    if (odd) {
      int stencil = (m_pool_dims[i] - 1) / 2;
      if (!(m_pads[i] == 0 || m_pads[i] == stencil)) {
        dc::MPIPrintStreamDebug()
          << "pooling: unsupported due to padding: " << m_pads[i];
        return false;
      }
      if (!(m_strides[i] == 1 || m_strides[i] == stencil + 1)) {
        dc::MPIPrintStreamDebug() << "pooling: unsupported due to strides";
        return false;
      }
    }
    else {
      if (m_pads[i] != 0)
        return false;
      if (m_pool_dims[i] != m_strides[i])
        return false;
    }
  }

  return true;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void pooling_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_distributions(tensor_overlap_constraints& constraints)
{
  data_type_distconv_adapter<TensorDataType>::setup_distributions(constraints);
  const auto& l =
    dynamic_cast<const pooling_layer<TensorDataType, T_layout, Dev>&>(
      this->layer());
  dc::IntVector overlap(dc::get_num_dims(l), 0);
  const auto& ps = l.get_parallel_strategy();
  auto pool_dims = l.m_pool_dims;
  std::reverse(pool_dims.begin(), pool_dims.end());
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
    if (splits == 1)
      continue;
    int ov = 0;
    if (pool_dims[i] % 2) {
      ov = (pool_dims[i] - 1) / 2;
    }
    else {
      // no halo dependency is assumed for now
      ov = 0;
    }
    overlap[i] = ov;
  }
  auto& prev_activations_dist = this->get_prev_activations_dist();
  auto& activations_dist = this->get_activations_dist();
  auto& error_signals_dist = this->get_error_signals_dist();
  auto& prev_error_signals_dist = this->get_prev_error_signals_dist();
  prev_activations_dist.set_overlap(overlap);
  constraints.mark_updated(prev_activations_dist);
  constraints.mark_invariant(prev_activations_dist);
  // cudnnPoolingBackward requires activations and
  // prev_error_signals must have the same stride
  constraints.mark_equivalent(activations_dist, prev_error_signals_dist);
  // cudnnPoolingBackward requires prev_activations and
  // error_signals must have the same stride
  constraints.mark_equivalent(error_signals_dist, prev_activations_dist);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape pooling_distconv_adapter<TensorDataType, Layout, Device>::
  get_activations_local_shape(int index) const
{
  assert_eq(index, 0);
  const auto& layer =
    dynamic_cast<const pooling_layer<TensorDataType, Layout, Device>&>(
      this->layer());
  auto filter_dims = layer.m_pool_dims;
  std::reverse(std::begin(filter_dims), std::end(filter_dims));
  auto strides = layer.m_strides;
  std::reverse(std::begin(strides), std::end(strides));
  const std::vector<int> dilations(dc::get_num_spatial_dims(layer), 1);
  bool use_padding = layer.m_pads[0] != 0;
  auto output_spatial_local_shape =
    ::distconv::get_pooling_output_local_tensor_shape(
      this->get_prev_activations(),
      filter_dims,
      strides,
      use_padding,
      dilations);
  return output_spatial_local_shape;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_distconv_adapter<TensorDataType, Layout, Device>::setup_layer(
  size_t workspace_capacity)
{
  auto& l =
    dynamic_cast<pooling_layer<TensorDataType, Layout, Device>&>(this->layer());

  // Init the dc::Pooling layer
  m_pooling = std::make_unique<dc::Pooling<TensorDataType>>(
    dc::get_backend(),
    dc::get_num_dims(l),
    dc::get_halo_exchange_method());

  std::string mode;
  switch (l.m_pool_mode) {
  case pooling_mode::MAX:
    mode = "MAX";
    break;
  case pooling_mode::MAX_DETERMINISTIC:
    mode = "MAX";
    break;
  case pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING:
    mode = "AVERAGE";
    break;
  case pooling_mode::AVERAGE_COUNT_EXCLUDE_PADDING:
    mode = "AVERAGE_NO_PAD";
    break;
  default:
    LBANN_ERROR("pooling_layer: no DISTCONV implementation for pooling mode");
  }

  std::vector<int> pool_dims = l.m_pool_dims;
  std::reverse(pool_dims.begin(), pool_dims.end());
  std::vector<int> pads = l.m_pads;
  std::reverse(pads.begin(), pads.end());
  std::vector<int> strides = l.m_strides;
  std::reverse(strides.begin(), strides.end());

  m_pooling->setup(this->get_prev_activations(),
                   this->get_activations(),
                   this->get_error_signals(),
                   this->get_prev_error_signals(),
                   pool_dims,
                   pads,
                   strides,
                   mode);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_distconv_adapter<TensorDataType, Layout, Device>::fp_compute(
  bool const training)
{
  m_pooling->forward(El::To<TensorDataType>(1),
                     this->get_prev_activations(),
                     El::To<TensorDataType>(0),
                     this->get_activations(),
                     training);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_distconv_adapter<TensorDataType, Layout, Device>::bp_compute()
{
  m_pooling->backward(El::To<TensorDataType>(1),
                      this->get_activations(),
                      this->get_prev_error_signals(),
                      this->get_prev_activations(),
                      El::To<TensorDataType>(0),
                      this->get_error_signals());
}
#endif // LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_pooling_layer_from_pbuf(lbann_comm* comm,
                              lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, pooling);

  using BuilderType = Builder<TensorDataType, Layout, Device>;
  const auto& params = proto_layer.pooling();
  pooling_mode const mode = to_pool_mode(params.pool_mode());
  if (params.has_vectors()) {
    return BuilderType::Build(comm,
                              params.pool_dims_size(),
                              protobuf::to_vector<int>(params.pool_dims()),
                              protobuf::to_vector<int>(params.pool_pads()),
                              protobuf::to_vector<int>(params.pool_strides()),
                              mode);
  }
  else {
    return BuilderType::Build(comm,
                              params.num_dims(),
                              params.pool_dims_i(),
                              params.pool_pads_i(),
                              params.pool_strides_i(),
                              mode);
  }
}

#define PROTO_DEVICE(T, Device)                                                \
  template class pooling_layer<T, data_layout::DATA_PARALLEL, Device>;         \
  LBANN_LAYER_BUILDER_ETI(pooling, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
