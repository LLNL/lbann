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

#include <hydrogen/utils/SimpleBuffer.hpp>
#define LBANN_GRU_LAYER_INSTANTIATE
#include "lbann/layers/learning/gru.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/hash.hpp"
#include "lbann/utils/sync_info_helpers.hpp"
#include "lbann/weights/initializer.hpp"

#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

// =========================================================
// Life cycle
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>::gru_layer(size_t hidden_size,
                                                     size_t num_layers)
  : data_type_layer<TensorDataType>(nullptr),
    m_hidden_size{hidden_size},
    m_num_layers{num_layers}
{
  this->m_expected_num_parent_layers = 2;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>::gru_layer(const gru_layer& other)
  : data_type_layer<TensorDataType>(other),
    m_hidden_size{other.m_hidden_size},
    m_num_layers{other.m_num_layers}
{
#ifdef LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
  m_onednn_cpu_objects.reset();
#endif // LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
#ifdef LBANN_GRU_LAYER_CUDNN_SUPPORTED
  m_cudnn_objects.reset();
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>&
gru_layer<TensorDataType, Layout, Device>::operator=(const gru_layer& other)
{
  data_type_layer<TensorDataType>::operator=(other);
  m_hidden_size = other.m_hidden_size;
  m_num_layers = other.m_num_layers;
#ifdef LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
  m_onednn_cpu_objects.reset();
#endif // LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
#ifdef LBANN_GRU_LAYER_CUDNN_SUPPORTED
  m_cudnn_objects.reset();
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>*
gru_layer<TensorDataType, Layout, Device>::copy() const
{
  return new gru_layer(*this);
}

// =========================================================
// Query functions
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string gru_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "GRU";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout gru_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
gru_layer<TensorDataType, Layout, Device>::get_device_allocation() const
{
  return Device;
}

template <typename T, data_layout L, El::Device D>
void gru_layer<T, L, D>::write_specific_proto(lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_gru();
  msg->set_hidden_size(m_hidden_size);
  msg->mutable_num_layers()->set_value(m_num_layers);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description gru_layer<TensorDataType, Layout, Device>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Hidden size", m_hidden_size);
  desc.add("Num layers", m_num_layers);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
const hydrogen::simple_buffer<El::byte, Device>&
gru_layer<TensorDataType, Layout, Device>::get_reserve_space() const
{
#ifdef LBANN_GRU_LAYER_CUDNN_SUPPORTED
  if constexpr (Device == El::Device::GPU) {
    return m_cudnn_objects->reserve_space;
  }
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED
  LBANN_ERROR("GRU layers' reserve space is not available without cuDNN.");
  static hydrogen::simple_buffer<El::byte, Device> invalid;
  return invalid; // silence compiler warnings
}

// =========================================================
// Setup
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Check parameters
  if (m_hidden_size <= 0) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has an invalid hidden state size (",
                m_hidden_size,
                ")");
  }
  if (m_num_layers <= 0) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has an invalid number of layers (",
                m_num_layers,
                ")");
  }

  // Check input dims
  const auto& input0_dims = this->get_input_dims(0);
  const auto& input1_dims = this->get_input_dims(1);
  auto dims_to_str = [](const std::vector<int>& dims) -> std::string {
    std::ostringstream ss;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i > 0) {
        ss << " x ";
      }
      ss << dims[i];
    }
    return ss.str();
  };
  if (input0_dims.size() != 2) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "expected a 2D input tensor for the input sequence, "
                "but recieved a tensor with ",
                "dimensions of ",
                dims_to_str(input0_dims));
  }
  if (input1_dims.size() != 2 ||
      static_cast<size_t>(input1_dims[0]) != m_num_layers ||
      static_cast<size_t>(input1_dims[1]) != m_hidden_size) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "expected a ",
                m_num_layers,
                " x ",
                m_hidden_size,
                " input tensor ",
                "for the initial hidden state, ",
                "but recieved a tensor with ",
                "dimensions of ",
                dims_to_str(input1_dims));
  }

  // Set output dims
  const std::vector<int> output_dims = {input0_dims[0],
                                        static_cast<int>(m_hidden_size)};
  this->set_output_dims(output_dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  const size_t sequence_length = this->get_input_dims()[0];
  const size_t input_size = this->get_input_size(0) / sequence_length;

  // Construct default weights if needed
  if (!this->has_weights()) {
    const std::vector<std::string> weight_names = {"ih_matrix",
                                                   "hh_matrix",
                                                   "ih_bias",
                                                   "hh_bias"};
    this->set_num_weights(4 * m_num_layers);
    const auto scale = El::To<TensorDataType>(1. / std::sqrt(m_hidden_size));
    for (size_t i = 0; i < m_num_layers; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        auto w = std::make_shared<data_type_weights<TensorDataType>>(
          *this->get_comm());
        auto init =
          std::make_unique<uniform_initializer<TensorDataType>>(-scale, scale);
        auto opt = this->m_model->template create_optimizer<TensorDataType>();
        w->set_name(
          lbann::build_string(this->get_name(), "_", weight_names[j], "_l", i));
        w->set_initializer(std::move(init));
        w->set_optimizer(std::move(opt));
        this->set_weights(4 * i + j, w);
        this->m_model->add_weights(std::move(w));
      }
    }
  }
  if (this->num_weights() != 4 * m_num_layers) {
    LBANN_ERROR("attempted to setup ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "with an invalid number of weights ",
                "(expected ",
                4 * m_num_layers,
                ", found ",
                this->num_weights(),
                ")");
  }

  // Setup weight dimensions and distribution
  for (size_t i = 0; i < m_num_layers; ++i) {
    auto& ih_matrix = this->get_weights(4 * i);
    auto& hh_matrix = this->get_weights(4 * i + 1);
    auto& ih_bias = this->get_weights(4 * i + 2);
    auto& hh_bias = this->get_weights(4 * i + 3);

    ih_matrix.set_dims({3 * m_hidden_size},
                       {i == 0 ? input_size : m_hidden_size});
    hh_matrix.set_dims({3 * m_hidden_size}, {m_hidden_size});
    ih_bias.set_dims({3 * m_hidden_size});
    hh_bias.set_dims({3 * m_hidden_size});
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    ih_matrix.set_matrix_distribution(dist);
    hh_matrix.set_matrix_distribution(dist);
    ih_bias.set_matrix_distribution(dist);
    hh_bias.set_matrix_distribution(dist);
  }

#ifdef LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
  if constexpr (Device == El::Device::CPU) {
    setup_onednn_cpu();
  }
#endif // LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
#ifdef LBANN_GRU_LAYER_CUDNN_SUPPORTED
  if constexpr (Device == El::Device::GPU) {
    setup_cudnn();
  }
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED
}

// =========================================================
// Forward prop and back prop
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::fp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("gru_layer::fp_compute");
  fp_compute_impl(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::bp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("gru_layer::bp_compute");
  bp_compute_impl(*this);
}

// =========================================================
// oneDNN CPU implementation
// =========================================================

#ifdef LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED

// ---------------------------------
// oneDNN CPU setup
// ---------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::setup_onednn_cpu()
{

  // Initialize storage for oneDNN objects
  m_onednn_cpu_objects = std::make_unique<OnednnCpuObjects>();

  // oneDNN objects
  using Backend = onednn_backend<El::Device::CPU>;
  using Memory = ::dnnl::memory;
  const auto data_type = Backend::template data_type<TensorDataType>();
  auto& onednn_objects = *m_onednn_cpu_objects;
  auto& engine = onednn::get_device_engine<El::Device::CPU>();

  // Dimensions
  const int sequence_length = this->get_input_dims(0)[0];
  const int input_size = this->get_input_size(0) / sequence_length;
  const int hidden_size = m_hidden_size;
  const int num_layers = m_num_layers;

  // Check that input and hidden size are identical
  // Note: As of oneDNN 2.1.0, GRUs with num_layers>1 require the
  // input and output channel sizes to be identical.
  if (num_layers > 1 && input_size != hidden_size) {
    LBANN_ERROR("oneDNN requires that multi-layer GRUs ",
                "have consistent input and output dimensions, ",
                "but ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has num_layers=",
                num_layers,
                ", input_size=",
                input_size,
                ", ",
                "and hidden_size=",
                hidden_size);
  }

  // Initialize storage for packed "ih_matrix" weights
  // Note: weights_layer in oneDNN terminology.
  {
    Memory::dims dims{num_layers,
                      /*num_directions=*/1,
                      input_size,
                      /*num_gates=*/3,
                      hidden_size};
    Memory::desc ldigo_desc(dims, data_type, Memory::format_tag::ldigo);
    Memory::desc ldgoi_desc(dims, data_type, Memory::format_tag::ldgoi);
    onednn_objects.forward_ih_matrix_weights.reset(Memory(ldigo_desc, engine));
    onednn_objects.backward_ih_matrix_weights.reset(Memory(ldgoi_desc, engine));
    onednn_objects.ih_matrix_weights_grad.reset(Memory(ldigo_desc, engine));
  }

  // Initialize storage for packed "hh_matrix" weights
  // Note: weights_iter in oneDNN terminology.
  {
    Memory::dims dims{num_layers,
                      /*num_directions=*/1,
                      hidden_size,
                      /*num_gates=*/3,
                      hidden_size};
    Memory::desc ldigo_desc(dims, data_type, Memory::format_tag::ldigo);
    Memory::desc ldgoi_desc(dims, data_type, Memory::format_tag::ldgoi);
    onednn_objects.forward_hh_matrix_weights.reset(Memory(ldigo_desc, engine));
    onednn_objects.backward_hh_matrix_weights.reset(Memory(ldgoi_desc, engine));
    onednn_objects.hh_matrix_weights_grad.reset(Memory(ldigo_desc, engine));
  }

  // Initialize storage for packed biases
  Memory::desc bias_weights_desc(
    {num_layers, /*num_directions=*/1, /*num_gates=*/4, hidden_size},
    data_type,
    Memory::format_tag::ldgo);
  onednn_objects.bias_weights.reset(Memory(bias_weights_desc, engine));
  onednn_objects.bias_weights_grad.reset(Memory(bias_weights_desc, engine));

  // Initialize empty tensors
  onednn_objects.final_hidden_desc.set(data_type, {});
  onednn_objects.final_hidden_grad_desc.set(data_type, {});
  onednn_objects.workspace.set(data_type, {});
}

// ---------------------------------
// oneDNN CPU forward prop
// ---------------------------------

namespace {
/** @brief Pack entries from LBANN weights into oneDNN buffers.
 *
 *  The oneDNN linear-before-reset GRU expects weights to be packed
 *  into three tensors:
 *
 *    weights_layer : num_layers x num_directions x input_size x num_gates x
 * hidden_size
 *
 *    weights_iter : num_layers x num_directions x hidden_size x num_gates x
 * hidden_size
 *
 *    bias: num_layers x num_directions x num_gates x hidden_size
 *
 *  The gates for the matrix weights are ordered {update, reset,
 *  output} and the gates for the bias are ordered {update, reset,
 *  output, before_reset}. Note that this is different than the order
 *  in LBANN and cuDNN, which is {reset, update, output}.
 *
 */
template <typename TensorDataType>
void pack_onednn_weights(
  size_t input_size,
  size_t hidden_size,
  size_t num_layers,
  onednn_backend<El::Device::CPU>::TensorDescriptor& ih_matrix_weights,
  onednn_backend<El::Device::CPU>::TensorDescriptor& hh_matrix_weights,
  onednn_backend<El::Device::CPU>::TensorDescriptor& bias_weights,
  const std::vector<El::Matrix<TensorDataType, El::Device::CPU>>& weights_list)
{

  // Typedefs
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;

  // Copy from "ih_matrix" weights to packed buffer
  auto make_ih_matrix_src_view = [&](size_t layer_id,
                                     size_t gate_id) -> const LocalMat {
    const auto& ih_matrix = weights_list[4 * layer_id];
    return ih_matrix(El::IR(gate_id * hidden_size, (gate_id + 1) * hidden_size),
                     El::ALL);
  };
  auto make_ih_matrix_dst_view = [&](size_t layer_id,
                                     size_t gate_id) -> LocalMat {
    auto* buffer = reinterpret_cast<TensorDataType*>(
      ih_matrix_weights.get().get_data_handle());
    const size_t layer_stride = 1 * input_size * 3 * hidden_size;
    const size_t gate_stride = hidden_size;
    return LocalMat(hidden_size,
                    input_size,
                    &buffer[layer_id * layer_stride + gate_id * gate_stride],
                    3 * hidden_size);
  };
  for (size_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    auto dst_update = make_ih_matrix_dst_view(layer_id, 0);
    auto dst_reset = make_ih_matrix_dst_view(layer_id, 1);
    auto dst_output = make_ih_matrix_dst_view(layer_id, 2);
    El::Copy(make_ih_matrix_src_view(layer_id, 1), dst_update);
    El::Copy(make_ih_matrix_src_view(layer_id, 0), dst_reset);
    El::Copy(make_ih_matrix_src_view(layer_id, 2), dst_output);
  }

  // Copy from "hh_matrix" weights to packed buffer
  auto make_hh_matrix_src_view = [&](size_t layer_id,
                                     size_t gate_id) -> const LocalMat {
    const auto& hh_matrix = weights_list[4 * layer_id + 1];
    return hh_matrix(El::IR(gate_id * hidden_size, (gate_id + 1) * hidden_size),
                     El::ALL);
  };
  auto make_hh_matrix_dst_view = [&](size_t layer_id,
                                     size_t gate_id) -> LocalMat {
    auto* buffer = reinterpret_cast<TensorDataType*>(
      hh_matrix_weights.get().get_data_handle());
    const size_t layer_stride = 1 * hidden_size * 3 * hidden_size;
    const size_t gate_stride = hidden_size;
    return LocalMat(hidden_size,
                    hidden_size,
                    &buffer[layer_id * layer_stride + gate_id * gate_stride],
                    3 * hidden_size);
  };
  for (size_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    auto dst_update = make_hh_matrix_dst_view(layer_id, 0);
    auto dst_reset = make_hh_matrix_dst_view(layer_id, 1);
    auto dst_output = make_hh_matrix_dst_view(layer_id, 2);
    El::Copy(make_hh_matrix_src_view(layer_id, 1), dst_update);
    El::Copy(make_hh_matrix_src_view(layer_id, 0), dst_reset);
    El::Copy(make_hh_matrix_src_view(layer_id, 2), dst_output);
  }

  // Copy from bias weights to packed buffer
  auto make_ih_bias_src_view = [&](size_t layer_id,
                                   size_t gate_id) -> const LocalMat {
    const auto& ih_bias = weights_list[4 * layer_id + 2];
    return ih_bias(El::IR(gate_id * hidden_size, (gate_id + 1) * hidden_size),
                   El::ALL);
  };
  auto make_hh_bias_src_view = [&](size_t layer_id,
                                   size_t gate_id) -> const LocalMat {
    const auto& hh_bias = weights_list[4 * layer_id + 3];
    return hh_bias(El::IR(gate_id * hidden_size, (gate_id + 1) * hidden_size),
                   El::ALL);
  };
  auto make_bias_dst_view = [&](size_t layer_id, size_t gate_id) -> LocalMat {
    auto* buffer =
      reinterpret_cast<TensorDataType*>(bias_weights.get().get_data_handle());
    const size_t layer_stride = 1 * 4 * hidden_size;
    const size_t gate_stride = hidden_size;
    return LocalMat(hidden_size,
                    1,
                    &buffer[layer_id * layer_stride + gate_id * gate_stride],
                    hidden_size);
  };
  for (size_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    auto dst_update = make_bias_dst_view(layer_id, 0);
    auto dst_reset = make_bias_dst_view(layer_id, 1);
    auto dst_output = make_bias_dst_view(layer_id, 2);
    auto dst_before_reset = make_bias_dst_view(layer_id, 3);
    El::Copy(make_ih_bias_src_view(layer_id, 1), dst_update);
    El::Axpy(1, make_hh_bias_src_view(layer_id, 1), dst_update);
    El::Copy(make_ih_bias_src_view(layer_id, 0), dst_reset);
    El::Axpy(1, make_hh_bias_src_view(layer_id, 0), dst_reset);
    El::Copy(make_ih_bias_src_view(layer_id, 2), dst_output);
    El::Copy(make_hh_bias_src_view(layer_id, 2), dst_before_reset);
  }
}
} // namespace

template <typename TensorDataType>
void fp_compute_impl(
  gru_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l)
{

  // Matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& input_sequence =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& init_hidden =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  auto& output_sequence = dynamic_cast<LocalMat&>(l.get_local_activations());

  // Dimensions
  const int local_mini_batch_size = input_sequence.Width();
  const int sequence_length = l.get_input_dims(0)[0];
  const int input_size = l.get_input_size(0) / sequence_length;
  const int hidden_size = l.m_hidden_size;
  const int num_layers = l.m_num_layers;

  // Return immediately if there is no local data.
  if (local_mini_batch_size <= 0) {
    return;
  }

  // oneDNN objects
  if (l.m_onednn_cpu_objects == nullptr) {
    LBANN_ERROR(l.get_type(),
                " layer \"",
                l.get_name(),
                "\" ",
                "attempted to run oneDNN CPU implementation ",
                "before initializing oneDNN objects");
  }
  constexpr auto Device = El::Device::CPU;
  using Backend = onednn_backend<Device>;
  using Memory = ::dnnl::memory;
  const auto data_type = Backend::template data_type<TensorDataType>();
  auto& onednn_objects = *l.m_onednn_cpu_objects;
  auto sync_info = force(El::MakeMultiSync(get_sync_info(output_sequence),
                                           get_sync_info(input_sequence),
                                           get_sync_info(init_hidden)));
  auto& engine = onednn::get_device_engine<Device>();
  auto stream = onednn::get_stream<Device>(engine, sync_info);

  // Configure input and output tensor descriptors
  onednn_objects.input_sequence_desc.set(
    data_type,
    {sequence_length, local_mini_batch_size, input_size},
    {input_size, El::To<int>(input_sequence.LDim()), 1});
  onednn_objects.input_sequence_desc.get().set_data_handle(
    const_cast<TensorDataType*>(input_sequence.LockedBuffer()),
    stream);
  onednn_objects.init_hidden_desc.set(
    data_type,
    {num_layers, /*num_directions=*/1, local_mini_batch_size, hidden_size},
    {hidden_size, 1, El::To<int>(init_hidden.LDim()), 1});
  onednn_objects.init_hidden_desc.get().set_data_handle(
    const_cast<TensorDataType*>(init_hidden.LockedBuffer()),
    stream);
  onednn_objects.output_sequence_desc.set(
    data_type,
    {sequence_length, local_mini_batch_size, hidden_size},
    {hidden_size, El::To<int>(output_sequence.LDim()), 1});
  onednn_objects.output_sequence_desc.get().set_data_handle(
    output_sequence.Buffer(),
    stream);

  // Pack weights into workspace buffer
  std::vector<LocalMat> weights_list;
  for (int i = 0; i < 4 * num_layers; ++i) {
    const auto& w =
      dynamic_cast<const LocalMat&>(l.weights_values(i).LockedMatrix());
    weights_list.emplace_back(El::LockedView(w));
  }
  pack_onednn_weights<TensorDataType>(input_size,
                                      hidden_size,
                                      num_layers,
                                      onednn_objects.forward_ih_matrix_weights,
                                      onednn_objects.forward_hh_matrix_weights,
                                      onednn_objects.bias_weights,
                                      weights_list);

  // Construct operation descriptor and primitive descriptor
  ::dnnl::lbr_gru_forward::desc gru_forward_desc(
    ::dnnl::prop_kind::forward_training,
    ::dnnl::rnn_direction::unidirectional_left2right,
    onednn_objects.input_sequence_desc.get().get_desc(),
    onednn_objects.init_hidden_desc.get().get_desc(),
    onednn_objects.forward_ih_matrix_weights.get().get_desc(),
    onednn_objects.forward_hh_matrix_weights.get().get_desc(),
    onednn_objects.bias_weights.get().get_desc(),
    onednn_objects.output_sequence_desc.get().get_desc(),
    onednn_objects.final_hidden_desc.get().get_desc());
  onednn_objects.gru_forward_primitive_desc =
    ::dnnl::lbr_gru_forward::primitive_desc(gru_forward_desc, engine);

  // Allocate workspace, if needed
  const auto& workspace_desc =
    onednn_objects.gru_forward_primitive_desc.workspace_desc();
  if (onednn_objects.workspace.get().get_desc() != workspace_desc) {
    onednn_objects.workspace.reset(Memory(workspace_desc, engine));
  }

  // Execute primitive
  /// @todo Cache primitive and reuse
  onednn_objects.gru_forward_primitive =
    ::dnnl::lbr_gru_forward(onednn_objects.gru_forward_primitive_desc);
  onednn_objects.gru_forward_primitive.execute(
    stream,
    {{DNNL_ARG_SRC_LAYER, onednn_objects.input_sequence_desc},
     {DNNL_ARG_SRC_ITER, onednn_objects.init_hidden_desc},
     {DNNL_ARG_DST_LAYER, onednn_objects.output_sequence_desc},
     {DNNL_ARG_DST_ITER, onednn_objects.final_hidden_desc},
     {DNNL_ARG_WEIGHTS_LAYER, onednn_objects.forward_ih_matrix_weights},
     {DNNL_ARG_WEIGHTS_ITER, onednn_objects.forward_hh_matrix_weights},
     {DNNL_ARG_BIAS, onednn_objects.bias_weights},
     {DNNL_ARG_WORKSPACE, onednn_objects.workspace}});
  stream.wait();
}

// ---------------------------------
// oneDNN CPU back prop
// ---------------------------------

namespace {
/** @brief See @c pack_onednn_weights */
template <typename TensorDataType>
void unpack_onednn_weights(
  size_t input_size,
  size_t hidden_size,
  size_t num_layers,
  const onednn_backend<El::Device::CPU>::TensorDescriptor& ih_matrix_weights,
  const onednn_backend<El::Device::CPU>::TensorDescriptor& hh_matrix_weights,
  const onednn_backend<El::Device::CPU>::TensorDescriptor& bias_weights,
  std::vector<El::Matrix<TensorDataType, El::Device::CPU>>& weights_list)
{

  // Typedefs
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;

  // Copy from packed buffer to "ih_matrix" weights
  auto make_ih_matrix_src_view = [&](size_t layer_id,
                                     size_t gate_id) -> const LocalMat {
    const auto* buffer = reinterpret_cast<const TensorDataType*>(
      ih_matrix_weights.get().get_data_handle());
    const size_t layer_stride = 1 * input_size * 3 * hidden_size;
    const size_t gate_stride = hidden_size;
    return LocalMat(hidden_size,
                    input_size,
                    &buffer[layer_id * layer_stride + gate_id * gate_stride],
                    3 * hidden_size);
  };
  auto make_ih_matrix_dst_view = [&](size_t layer_id,
                                     size_t gate_id) -> LocalMat {
    auto& ih_matrix = weights_list[4 * layer_id];
    return ih_matrix(El::IR(gate_id * hidden_size, (gate_id + 1) * hidden_size),
                     El::ALL);
  };
  for (size_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    auto dst_reset = make_ih_matrix_dst_view(layer_id, 0);
    auto dst_update = make_ih_matrix_dst_view(layer_id, 1);
    auto dst_output = make_ih_matrix_dst_view(layer_id, 2);
    El::Copy(make_ih_matrix_src_view(layer_id, 1), dst_reset);
    El::Copy(make_ih_matrix_src_view(layer_id, 0), dst_update);
    El::Copy(make_ih_matrix_src_view(layer_id, 2), dst_output);
  }

  // Copy from packed buffer to "hh_matrix" weights
  auto make_hh_matrix_src_view = [&](size_t layer_id,
                                     size_t gate_id) -> const LocalMat {
    const auto* buffer = reinterpret_cast<const TensorDataType*>(
      hh_matrix_weights.get().get_data_handle());
    const size_t layer_stride = 1 * hidden_size * 3 * hidden_size;
    const size_t gate_stride = hidden_size;
    return LocalMat(hidden_size,
                    hidden_size,
                    &buffer[layer_id * layer_stride + gate_id * gate_stride],
                    3 * hidden_size);
  };
  auto make_hh_matrix_dst_view = [&](size_t layer_id,
                                     size_t gate_id) -> LocalMat {
    auto& hh_matrix = weights_list[4 * layer_id + 1];
    return hh_matrix(El::IR(gate_id * hidden_size, (gate_id + 1) * hidden_size),
                     El::ALL);
  };
  for (size_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    auto dst_reset = make_hh_matrix_dst_view(layer_id, 0);
    auto dst_update = make_hh_matrix_dst_view(layer_id, 1);
    auto dst_output = make_hh_matrix_dst_view(layer_id, 2);
    El::Copy(make_hh_matrix_src_view(layer_id, 1), dst_reset);
    El::Copy(make_hh_matrix_src_view(layer_id, 0), dst_update);
    El::Copy(make_hh_matrix_src_view(layer_id, 2), dst_output);
  }

  // Copy from packed buffer to bias weights
  auto make_bias_src_view = [&](size_t layer_id,
                                size_t gate_id) -> const LocalMat {
    const auto* buffer = reinterpret_cast<const TensorDataType*>(
      bias_weights.get().get_data_handle());
    const size_t layer_stride = 1 * 4 * hidden_size;
    const size_t gate_stride = hidden_size;
    return LocalMat(hidden_size,
                    1,
                    &buffer[layer_id * layer_stride + gate_id * gate_stride],
                    hidden_size);
  };
  auto make_ih_bias_dst_view = [&](size_t layer_id,
                                   size_t gate_id) -> LocalMat {
    auto& ih_bias = weights_list[4 * layer_id + 2];
    return ih_bias(El::IR(gate_id * hidden_size, (gate_id + 1) * hidden_size),
                   El::ALL);
  };
  auto make_hh_bias_dst_view = [&](size_t layer_id,
                                   size_t gate_id) -> LocalMat {
    auto& hh_bias = weights_list[4 * layer_id + 3];
    return hh_bias(El::IR(gate_id * hidden_size, (gate_id + 1) * hidden_size),
                   El::ALL);
  };
  for (size_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    const auto src_update = make_bias_src_view(layer_id, 0);
    const auto src_reset = make_bias_src_view(layer_id, 1);
    const auto src_output = make_bias_src_view(layer_id, 2);
    const auto src_before_reset = make_bias_src_view(layer_id, 3);
    auto dst_ih_reset = make_ih_bias_dst_view(layer_id, 0);
    auto dst_ih_update = make_ih_bias_dst_view(layer_id, 1);
    auto dst_ih_output = make_ih_bias_dst_view(layer_id, 2);
    auto dst_hh_reset = make_hh_bias_dst_view(layer_id, 0);
    auto dst_hh_update = make_hh_bias_dst_view(layer_id, 1);
    auto dst_hh_output = make_hh_bias_dst_view(layer_id, 2);
    El::Copy(src_reset, dst_ih_reset);
    El::Copy(src_update, dst_ih_update);
    El::Copy(src_output, dst_ih_output);
    El::Copy(src_reset, dst_hh_reset);
    El::Copy(src_update, dst_hh_update);
    El::Copy(src_before_reset, dst_hh_output);
  }
}
} // namespace

template <typename TensorDataType>
void bp_compute_impl(
  gru_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l)
{

  // Matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& output_sequence_grad =
    dynamic_cast<const LocalMat&>(l.get_local_prev_error_signals());
  auto& input_sequence_grad =
    dynamic_cast<LocalMat&>(l.get_local_error_signals(0));
  auto& init_hidden_grad =
    dynamic_cast<LocalMat&>(l.get_local_error_signals(1));

  // Dimensions
  const int local_mini_batch_size = output_sequence_grad.Width();
  const int sequence_length = l.get_input_dims(0)[0];
  const int input_size = l.get_input_size(0) / sequence_length;
  const int hidden_size = l.m_hidden_size;
  const int num_layers = l.m_num_layers;

  // oneDNN objects
  if (l.m_onednn_cpu_objects == nullptr) {
    LBANN_ERROR(l.get_type(),
                " layer \"",
                l.get_name(),
                "\" ",
                "attempted to run oneDNN CPU implementation ",
                "before initializing oneDNN objects");
  }
  constexpr auto Device = El::Device::CPU;
  using Backend = onednn_backend<Device>;
  const auto data_type = Backend::template data_type<TensorDataType>();
  auto& onednn_objects = *l.m_onednn_cpu_objects;
  auto sync_info =
    force(El::MakeMultiSync(get_sync_info(input_sequence_grad),
                            get_sync_info(init_hidden_grad),
                            get_sync_info(output_sequence_grad)));
  auto& engine = onednn::get_device_engine<Device>();
  auto stream = onednn::get_stream<Device>(engine, sync_info);

  // Define closure to send weight gradients to optimizers
  std::vector<LocalMat> weights_grad_list(4 * num_layers);
  for (int i = 0; i < num_layers; ++i) {
    weights_grad_list[4 * i].Resize(3 * hidden_size, input_size);
    weights_grad_list[4 * i + 1].Resize(3 * hidden_size, hidden_size);
    weights_grad_list[4 * i + 2].Resize(3 * hidden_size, 1);
    weights_grad_list[4 * i + 3].Resize(3 * hidden_size, 1);
  }
  auto send_weight_grads_to_optimizers = [&]() {
    TensorDataType buf_scale, in_scale;
    for (int i = 0; i < 4 * num_layers; ++i) {
      auto&& opt = l.get_weights(i).get_optimizer();
      if (opt != nullptr) {
        auto& buf = opt->get_gradient_buffer(buf_scale, in_scale, true);
        El::Scale(buf_scale, buf);
        El::Axpy(in_scale, weights_grad_list[i], buf.Matrix());
      }
    }
  };

  // Return immediately if there is no local data
  if (local_mini_batch_size <= 0) {
    for (auto& dw : weights_grad_list) {
      El::Zero(dw);
    }
    send_weight_grads_to_optimizers();
    return;
  }

  // Configure input grad and output grad tensor descriptors
  // Note: Reuse tensor descriptors from forward prop.
  onednn_objects.output_sequence_grad_desc.set(
    data_type,
    {sequence_length, local_mini_batch_size, hidden_size},
    {hidden_size, El::To<int>(output_sequence_grad.LDim()), 1});
  onednn_objects.output_sequence_grad_desc.get().set_data_handle(
    const_cast<TensorDataType*>(output_sequence_grad.LockedBuffer()),
    stream);
  onednn_objects.input_sequence_grad_desc.set(
    data_type,
    {sequence_length, local_mini_batch_size, input_size},
    {input_size, El::To<int>(input_sequence_grad.LDim()), 1});
  onednn_objects.input_sequence_grad_desc.get().set_data_handle(
    input_sequence_grad.Buffer(),
    stream);
  onednn_objects.init_hidden_grad_desc.set(
    data_type,
    {num_layers, /*num_directions=*/1, local_mini_batch_size, hidden_size},
    {hidden_size, 1, El::To<int>(init_hidden_grad.LDim()), 1});
  onednn_objects.init_hidden_grad_desc.get().set_data_handle(
    init_hidden_grad.Buffer(),
    stream);

  // Reorder matrix weights from LDIGO to LDGOI format
  auto&& forward_ih_matrix_weights =
    onednn_objects.forward_ih_matrix_weights.get();
  auto&& backward_ih_matrix_weights =
    onednn_objects.backward_ih_matrix_weights.get();
  auto&& forward_hh_matrix_weights =
    onednn_objects.forward_hh_matrix_weights.get();
  auto&& backward_hh_matrix_weights =
    onednn_objects.backward_hh_matrix_weights.get();
  ::dnnl::reorder reorder_ih_matrix_weights_primitive(
    forward_ih_matrix_weights,
    backward_ih_matrix_weights);
  ::dnnl::reorder reorder_hh_matrix_weights_primitive(
    forward_hh_matrix_weights,
    backward_hh_matrix_weights);
  reorder_ih_matrix_weights_primitive.execute(stream,
                                              forward_ih_matrix_weights,
                                              backward_ih_matrix_weights);
  reorder_hh_matrix_weights_primitive.execute(stream,
                                              forward_hh_matrix_weights,
                                              backward_hh_matrix_weights);

  // Clear weights gradients
  std::memset(
    onednn_objects.ih_matrix_weights_grad.get().get_data_handle(),
    0,
    onednn_objects.ih_matrix_weights_grad.get().get_desc().get_size());
  std::memset(
    onednn_objects.hh_matrix_weights_grad.get().get_data_handle(),
    0,
    onednn_objects.hh_matrix_weights_grad.get().get_desc().get_size());
  std::memset(onednn_objects.bias_weights_grad.get().get_data_handle(),
              0,
              onednn_objects.bias_weights_grad.get().get_desc().get_size());

  // Construct operation descriptor and primitive descriptor
  ::dnnl::lbr_gru_backward::desc gru_backward_desc(
    ::dnnl::prop_kind::backward,
    ::dnnl::rnn_direction::unidirectional_left2right,
    onednn_objects.input_sequence_desc.get().get_desc(),
    onednn_objects.init_hidden_desc.get().get_desc(),
    backward_ih_matrix_weights.get_desc(),
    backward_hh_matrix_weights.get_desc(),
    onednn_objects.bias_weights.get().get_desc(),
    onednn_objects.output_sequence_desc.get().get_desc(),
    onednn_objects.final_hidden_desc.get().get_desc(),
    onednn_objects.input_sequence_grad_desc.get().get_desc(),
    onednn_objects.init_hidden_grad_desc.get().get_desc(),
    onednn_objects.ih_matrix_weights_grad.get().get_desc(),
    onednn_objects.hh_matrix_weights_grad.get().get_desc(),
    onednn_objects.bias_weights_grad.get().get_desc(),
    onednn_objects.output_sequence_grad_desc.get().get_desc(),
    onednn_objects.final_hidden_grad_desc.get().get_desc());
  onednn_objects.gru_backward_primitive_desc =
    ::dnnl::lbr_gru_backward::primitive_desc(
      gru_backward_desc,
      engine,
      onednn_objects.gru_forward_primitive_desc);

  // Execute backprop primitive
  /// @todo Cache primitives and reuse
  onednn_objects.gru_backward_primitive =
    ::dnnl::lbr_gru_backward(onednn_objects.gru_backward_primitive_desc);
  onednn_objects.gru_backward_primitive.execute(
    stream,
    {{DNNL_ARG_SRC_LAYER, onednn_objects.input_sequence_desc},
     {DNNL_ARG_SRC_ITER, onednn_objects.init_hidden_desc},
     {DNNL_ARG_DST_LAYER, onednn_objects.output_sequence_desc},
     {DNNL_ARG_DST_ITER, onednn_objects.final_hidden_desc},
     {DNNL_ARG_WEIGHTS_LAYER, backward_ih_matrix_weights},
     {DNNL_ARG_WEIGHTS_ITER, backward_hh_matrix_weights},
     {DNNL_ARG_BIAS, onednn_objects.bias_weights},
     {DNNL_ARG_DIFF_SRC_LAYER, onednn_objects.input_sequence_grad_desc},
     {DNNL_ARG_DIFF_SRC_ITER, onednn_objects.init_hidden_grad_desc},
     {DNNL_ARG_DIFF_DST_LAYER, onednn_objects.output_sequence_grad_desc},
     {DNNL_ARG_DIFF_DST_ITER, onednn_objects.final_hidden_grad_desc},
     {DNNL_ARG_DIFF_WEIGHTS_LAYER, onednn_objects.ih_matrix_weights_grad},
     {DNNL_ARG_DIFF_WEIGHTS_ITER, onednn_objects.hh_matrix_weights_grad},
     {DNNL_ARG_DIFF_BIAS, onednn_objects.bias_weights_grad},
     {DNNL_ARG_WORKSPACE, onednn_objects.workspace}});
  stream.wait();

  // Send gradients to optimizers
  unpack_onednn_weights<TensorDataType>(input_size,
                                        hidden_size,
                                        num_layers,
                                        onednn_objects.ih_matrix_weights_grad,
                                        onednn_objects.hh_matrix_weights_grad,
                                        onednn_objects.bias_weights_grad,
                                        weights_grad_list);
  send_weight_grads_to_optimizers();
}

#endif // LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED

// =========================================================
// cuDNN implementation
// =========================================================

#ifdef LBANN_GRU_LAYER_CUDNN_SUPPORTED

// ---------------------------------
// cuDNN setup
// ---------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::setup_cudnn()
{

  // Dimensions
  const size_t sequence_length = this->get_input_dims(0)[0];
  const size_t input_size = this->get_input_size(0) / sequence_length;

  // Initialize storage for cuDNN objects
  m_cudnn_objects = std::make_unique<CudnnObjects>();

  // RNN descriptor
  static dnn_lib::DropoutDescriptor dropout_desc;
  dropout_desc.set(0, nullptr, 0, 0);
  m_cudnn_objects->rnn_desc.set(CUDNN_RNN_ALGO_STANDARD,
                                CUDNN_GRU,
                                CUDNN_RNN_DOUBLE_BIAS,
                                CUDNN_UNIDIRECTIONAL,
                                CUDNN_LINEAR_INPUT,
                                dnn_lib::get_data_type<TensorDataType>(),
                                dnn_lib::get_data_type<TensorDataType>(),
                                dnn_lib::get_default_convolution_math_type(),
                                input_size,
                                m_hidden_size,
                                m_hidden_size, // proj_size
                                m_num_layers,
                                dropout_desc,
                                CUDNN_RNN_PADDED_IO_ENABLED);
}

// ---------------------------------
// cuDNN forward prop
// ---------------------------------

/// @todo Figure out cuDNN bug
// Note (tym 10/3/20): We experience an error in cuDNN with certain
// mini-batch sizes. Hack around it by padding to a minimum batch
// size.
// Note (BVE 10/3/20): Note that the bug seems to be triggered by
// switching mini-batch sizes when the last mini-batch is too small.
// This means that we need a lower bound, which is emperically tested
// to be 128 for WAE. However, if the initial mini-batch size is less
// than 128 and it isn't changed, things seem to be okay. So set the
// threshold to be the smaller of the initial mini-batch size or 128.
#define MIN_WORKSPACE_MINI_BATCH_SIZE 128
size_t active_min_workspace_mini_batch_size = 0;

namespace {
template <typename TensorDataType>
void pack_cudnn_rnn_weights(
  const cudnnHandle_t& handle,
  const dnn_lib::RNNDescriptor& rnn_desc,
  const El::SyncInfo<El::Device::GPU>& sync_info,
  size_t input_size,
  size_t hidden_size,
  size_t num_layers,
  void* packed_weights_buffer,
  size_t packed_weights_size,
  const std::vector<El::Matrix<TensorDataType, El::Device::GPU>>& weights_list)
{

  // Construct objects
  static dnn_lib::TensorDescriptor matrix_desc, bias_desc;
  El::Matrix<TensorDataType, El::Device::GPU> packed_weights_view;
  packed_weights_view.SetSyncInfo(sync_info);

  // Function to get pointers in packed weights buffer
  using PtrPair = std::pair<TensorDataType*, TensorDataType*>;
  auto get_ptrs = [&](size_t i, size_t id) -> PtrPair {
    PtrPair ptrs;
    matrix_desc.create();
    bias_desc.create();
    CHECK_CUDNN(
      cudnnGetRNNWeightParams(handle,
                              rnn_desc,
                              i,
                              packed_weights_size,
                              packed_weights_buffer,
                              id,
                              matrix_desc,
                              reinterpret_cast<void**>(&ptrs.first),
                              bias_desc,
                              reinterpret_cast<void**>(&ptrs.second)));
    return ptrs;
  };

  for (size_t i = 0; i < num_layers; ++i) {

    // Copy from ih_matrix
    const auto& ih_matrix = weights_list[4 * i];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.Attach(i == 0 ? input_size : hidden_size,
                                 hidden_size,
                                 get_ptrs(i, id).first,
                                 i == 0 ? input_size : hidden_size);
      El::Transpose(
        ih_matrix(El::IR(id * hidden_size, (id + 1) * hidden_size), El::ALL),
        packed_weights_view,
        false);
    }

    // Copy from hh_matrix
    const auto& hh_matrix = weights_list[4 * i + 1];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.Attach(hidden_size,
                                 hidden_size,
                                 get_ptrs(i, 3 + id).first,
                                 hidden_size);
      El::Transpose(
        hh_matrix(El::IR(id * hidden_size, (id + 1) * hidden_size), El::ALL),
        packed_weights_view,
        false);
    }

    // Copy from ih_bias
    const auto& ih_bias = weights_list[4 * i + 2];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.Attach(hidden_size,
                                 1,
                                 get_ptrs(i, id).second,
                                 hidden_size);
      El::Copy(
        ih_bias(El::IR(id * hidden_size, (id + 1) * hidden_size), El::ALL),
        packed_weights_view);
    }

    // Copy from hh_bias
    const auto& hh_bias = weights_list[4 * i + 3];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.Attach(hidden_size,
                                 1,
                                 get_ptrs(i, 3 + id).second,
                                 hidden_size);
      El::Copy(
        hh_bias(El::IR(id * hidden_size, (id + 1) * hidden_size), El::ALL),
        packed_weights_view);
    }
  }
}

} // namespace

template <typename TensorDataType>
void fp_compute_impl(
  gru_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l)
{

  // Matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& input_sequence =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& init_hidden =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  auto& output_sequence = dynamic_cast<LocalMat&>(l.get_local_activations());

  // Dimensions
  const size_t sequence_length = l.get_input_dims(0)[0];
  const size_t input_size = l.get_input_size(0) / sequence_length;
  const size_t hidden_size = l.m_hidden_size;
  const size_t num_layers = l.m_num_layers;

  // Configure workspace mini-batch size
  // Note: Return immediately if there is no local data.
  /// @todo Figure out cuDNN bug
  const size_t mini_batch_size = input_sequence.Width();
  if (mini_batch_size <= 0) {
    return;
  }
  if (active_min_workspace_mini_batch_size == 0) {
    // Set the minumum to the smaller of the initial mini-batch size
    // or a predefined minumim
    active_min_workspace_mini_batch_size =
      El::Min(mini_batch_size, MIN_WORKSPACE_MINI_BATCH_SIZE);
  }
  const size_t workspace_mini_batch_size =
    El::Max(mini_batch_size, active_min_workspace_mini_batch_size);

  // GPU objects
  if (l.m_cudnn_objects == nullptr) {
    LBANN_ERROR(l.get_type(),
                " layer \"",
                l.get_name(),
                "\" ",
                "attempted to run cuDNN implementation ",
                "before initializing cuDNN objects");
  }
  auto&& sync_info = input_sequence.GetSyncInfo();
  auto&& stream = sync_info.Stream();
  auto&& handle = dnn_lib::get_handle();
  auto& cudnn_objects = *l.m_cudnn_objects;
  const auto data_type = dnn_lib::get_data_type<TensorDataType>();

  // Configure input and output tensor descriptors
  std::vector<int> sequence_lengths(workspace_mini_batch_size, sequence_length);
  cudnn_objects.input_desc.set(data_type,
                               CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
                               sequence_length,
                               workspace_mini_batch_size,
                               input_size,
                               sequence_lengths.data(),
                               nullptr);
  cudnn_objects.output_desc.set(data_type,
                                CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
                                sequence_length,
                                workspace_mini_batch_size,
                                hidden_size,
                                sequence_lengths.data(),
                                nullptr);
  cudnn_objects.hidden_desc.set(data_type,
                                num_layers,
                                workspace_mini_batch_size,
                                hidden_size);

  // Allocate cuDNN workspace buffers
  size_t cudnn_workspace_size, cudnn_reserve_space_size;
  CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(handle,
                                        cudnn_objects.rnn_desc,
                                        CUDNN_FWD_MODE_TRAINING,
                                        cudnn_objects.input_desc,
                                        &cudnn_workspace_size,
                                        &cudnn_reserve_space_size));
  if (cudnn_objects.workspace.size() < cudnn_workspace_size) {
    /// @todo Handle synchronization
    cudnn_objects.workspace.allocate(cudnn_workspace_size);
  }
  if (cudnn_objects.reserve_space.size() < cudnn_reserve_space_size) {
    /// @todo Handle synchronization
    cudnn_objects.reserve_space.allocate(cudnn_reserve_space_size);
  }
  if (cudnn_objects.gpu_sequence_lengths.size() < workspace_mini_batch_size) {
    /// @todo Handle synchronization
    cudnn_objects.gpu_sequence_lengths.allocate(workspace_mini_batch_size);
    std::vector<int32_t> cpu_sequence_lengths(workspace_mini_batch_size,
                                              sequence_length);
    CHECK_CUDA(cudaMemcpyAsync(cudnn_objects.gpu_sequence_lengths.data(),
                               cpu_sequence_lengths.data(),
                               cpu_sequence_lengths.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice,
                               stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // Make sure tensors are in correct format
  cudnn_objects.input_sequence_workspace.SetSyncInfo(sync_info);
  cudnn_objects.init_hidden_workspace.SetSyncInfo(sync_info);
  cudnn_objects.output_sequence_workspace.SetSyncInfo(sync_info);
  constexpr size_t one{1};
  cudnn_objects.input_sequence_workspace.Resize(sequence_length * input_size,
                                                workspace_mini_batch_size);
  cudnn_objects.init_hidden_workspace.Resize(workspace_mini_batch_size *
                                               hidden_size,
                                             num_layers);
  cudnn_objects.output_sequence_workspace.Resize(sequence_length * hidden_size,
                                                 workspace_mini_batch_size);
  El::Zero(cudnn_objects.input_sequence_workspace);
  El::Zero(cudnn_objects.init_hidden_workspace);
  auto input_sequence_workspace_ =
    cudnn_objects.input_sequence_workspace(El::ALL, El::IR(0, mini_batch_size));
  El::Copy(input_sequence, input_sequence_workspace_);
  cuda::copy_tensor(
    stream,
    {mini_batch_size, num_layers, hidden_size},
    init_hidden.LockedBuffer(),
    {static_cast<size_t>(init_hidden.LDim()), hidden_size, one},
    cudnn_objects.init_hidden_workspace.Buffer(),
    {hidden_size, workspace_mini_batch_size * hidden_size, one});

  // Pack weights into workspace buffer
  /// @todo Handle synchronization
  size_t weights_size;
  CHECK_CUDNN(
    cudnnGetRNNWeightSpaceSize(handle, cudnn_objects.rnn_desc, &weights_size));
  cudnn_objects.weights_workspace.allocate(weights_size);
  std::vector<LocalMat> weights_list;
  for (size_t i = 0; i < 4 * num_layers; ++i) {
    const auto& w =
      dynamic_cast<const LocalMat&>(l.weights_values(i).LockedMatrix());
    weights_list.emplace_back(El::LockedView(w));
  }
  pack_cudnn_rnn_weights<TensorDataType>(handle,
                                         cudnn_objects.rnn_desc,
                                         sync_info,
                                         input_size,
                                         hidden_size,
                                         num_layers,
                                         cudnn_objects.weights_workspace.data(),
                                         cudnn_objects.weights_workspace.size(),
                                         weights_list);

#if !defined(LBANN_DEBUG) // Disable CUDA graphs

  // Compute hash with cuDNN function arguments
  size_t hash{0};
  hash = hash_combine(hash, cudnn_objects.gpu_sequence_lengths.data());
  hash =
    hash_combine(hash, cudnn_objects.input_sequence_workspace.LockedBuffer());
  hash = hash_combine(hash, cudnn_objects.init_hidden_workspace.LockedBuffer());
  hash = hash_combine(hash, cudnn_objects.output_sequence_workspace.Buffer());
  hash = hash_combine(hash, cudnn_objects.weights_workspace.data());
  hash = hash_combine(hash, cudnn_objects.workspace.data());
  hash = hash_combine(hash, cudnn_objects.reserve_space.data());

  // Capture graph if not in cache
  if (cudnn_objects.forward_prop_graph_cache.count(workspace_mini_batch_size) <
        1 ||
      cudnn_objects.forward_prop_graph_cache[workspace_mini_batch_size].first !=
        hash) {
    cuda::Graph::begin_capture(stream);

#endif // !defined(LBANN_DEBUG)

    // cuDNN forward prop
    CHECK_CUDNN(
      cudnnRNNForward(handle,
                      cudnn_objects.rnn_desc,
                      CUDNN_FWD_MODE_TRAINING,
                      cudnn_objects.gpu_sequence_lengths.data(),
                      cudnn_objects.input_desc,
                      cudnn_objects.input_sequence_workspace.LockedBuffer(),
                      cudnn_objects.output_desc,
                      cudnn_objects.output_sequence_workspace.Buffer(),
                      cudnn_objects.hidden_desc,
                      cudnn_objects.init_hidden_workspace.LockedBuffer(),
                      nullptr,                   // hy
                      cudnn_objects.hidden_desc, // cDesc
                      nullptr,                   // cx
                      nullptr,                   // cy
                      cudnn_objects.weights_workspace.size(),
                      cudnn_objects.weights_workspace.data(),
                      cudnn_objects.workspace.size(),
                      cudnn_objects.workspace.data(),
                      cudnn_objects.reserve_space.size(),
                      cudnn_objects.reserve_space.data()));

#if !defined(LBANN_DEBUG) // Disable CUDA graphs

    // Finish capturing graph and update cache
    auto graph = cuda::Graph::end_capture(stream);
    auto& cache_pair =
      cudnn_objects.forward_prop_graph_cache[workspace_mini_batch_size];
    cache_pair.first = hash;
    cache_pair.second.update(graph);
  }

  // Launch CUDA graph
  cudnn_objects.forward_prop_graph_cache[workspace_mini_batch_size]
    .second.launch(stream);

#endif // !defined(LBANN_DEBUG)

  // Output tensor
  El::LockedView(output_sequence,
                 cudnn_objects.output_sequence_workspace,
                 El::ALL,
                 El::IR(0, mini_batch_size));
}

// ---------------------------------
// cuDNN back prop
// ---------------------------------

namespace {
template <typename TensorDataType>
void unpack_cudnn_rnn_weights(
  const cudnnHandle_t& handle,
  const dnn_lib::RNNDescriptor& rnn_desc,
  const El::SyncInfo<El::Device::GPU>& sync_info,
  size_t input_size,
  size_t hidden_size,
  size_t num_layers,
  const void* packed_weights_buffer,
  size_t packed_weights_size,
  std::vector<El::Matrix<TensorDataType, El::Device::GPU>>& weights_list)
{

  // Construct objects
  static dnn_lib::TensorDescriptor matrix_desc, bias_desc;
  El::Matrix<TensorDataType, El::Device::GPU> packed_weights_view;
  packed_weights_view.SetSyncInfo(sync_info);

  // Function to get pointers in packed weights buffer
  using PtrPair = std::pair<TensorDataType*, TensorDataType*>;
  auto get_ptrs = [&](size_t i, size_t id) -> PtrPair {
    PtrPair ptrs;
    matrix_desc.create();
    bias_desc.create();
    CHECK_CUDNN(
      cudnnGetRNNWeightParams(handle,
                              rnn_desc,
                              i,
                              packed_weights_size,
                              packed_weights_buffer,
                              id,
                              matrix_desc,
                              reinterpret_cast<void**>(&ptrs.first),
                              bias_desc,
                              reinterpret_cast<void**>(&ptrs.second)));
    return ptrs;
  };

  for (size_t i = 0; i < num_layers; ++i) {

    // Copy to ih_matrix
    auto& ih_matrix = weights_list[4 * i];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.LockedAttach(i == 0 ? input_size : hidden_size,
                                       hidden_size,
                                       get_ptrs(i, id).first,
                                       i == 0 ? input_size : hidden_size);
      auto ih_matrix_view =
        ih_matrix(El::IR(id * hidden_size, (id + 1) * hidden_size), El::ALL);
      El::Transpose(packed_weights_view, ih_matrix_view, false);
    }

    // Copy to hh_matrix
    auto& hh_matrix = weights_list[4 * i + 1];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.LockedAttach(hidden_size,
                                       hidden_size,
                                       get_ptrs(i, 3 + id).first,
                                       hidden_size);
      auto hh_matrix_view =
        hh_matrix(El::IR(id * hidden_size, (id + 1) * hidden_size), El::ALL);
      El::Transpose(packed_weights_view, hh_matrix_view, false);
    }

    // Copy to ih_bias
    auto& ih_bias = weights_list[4 * i + 2];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.LockedAttach(hidden_size,
                                       1,
                                       get_ptrs(i, id).second,
                                       hidden_size);
      auto ih_bias_view =
        ih_bias(El::IR(id * hidden_size, (id + 1) * hidden_size), El::ALL);
      El::Copy(packed_weights_view, ih_bias_view);
    }

    // Copy to hh_bias
    auto& hh_bias = weights_list[4 * i + 3];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.LockedAttach(hidden_size,
                                       1,
                                       get_ptrs(i, 3 + id).second,
                                       hidden_size);
      auto hh_bias_view =
        hh_bias(El::IR(id * hidden_size, (id + 1) * hidden_size), El::ALL);
      El::Copy(packed_weights_view, hh_bias_view);
    }
  }
}
} // namespace

template <typename TensorDataType>
void bp_compute_impl(
  gru_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l)
{

  // Matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& output_sequence_grad =
    dynamic_cast<const LocalMat&>(l.get_local_prev_error_signals());
  auto& input_sequence_grad =
    dynamic_cast<LocalMat&>(l.get_local_error_signals(0));
  auto& init_hidden_grad =
    dynamic_cast<LocalMat&>(l.get_local_error_signals(1));

  // Dimensions
  const size_t sequence_length = l.get_input_dims(0)[0];
  const size_t input_size = l.get_input_size(0) / sequence_length;
  const size_t hidden_size = l.m_hidden_size;
  const size_t num_layers = l.m_num_layers;

  // Configure workspace mini-batch size
  /// @todo Figure out cuDNN bug
  const size_t mini_batch_size = output_sequence_grad.Width();
  if (mini_batch_size <= 0) {
    return;
  }
  const size_t workspace_mini_batch_size =
    El::Max(mini_batch_size, active_min_workspace_mini_batch_size);

  // GPU objects
  if (l.m_cudnn_objects == nullptr) {
    LBANN_ERROR(l.get_type(),
                " layer \"",
                l.get_name(),
                "\" ",
                "attempted to run cuDNN implementation ",
                "before initializing cuDNN objects");
  }
  auto&& sync_info = output_sequence_grad.GetSyncInfo();
  auto&& stream = sync_info.Stream();
  auto&& handle = dnn_lib::get_handle();
  auto& cudnn_objects = *l.m_cudnn_objects;

  // Define closure to send weight gradients to optimizers
  std::vector<LocalMat> weights_grad_list(4 * num_layers);
  for (auto& dw : weights_grad_list) {
    dw.SetSyncInfo(sync_info);
  }
  for (size_t i = 0; i < num_layers; ++i) {
    weights_grad_list[4 * i].Resize(3 * hidden_size,
                                    i == 0 ? input_size : hidden_size);
    weights_grad_list[4 * i + 1].Resize(3 * hidden_size, hidden_size);
    weights_grad_list[4 * i + 2].Resize(3 * hidden_size, 1);
    weights_grad_list[4 * i + 3].Resize(3 * hidden_size, 1);
  }
  auto send_weight_grads_to_optimizers = [&]() {
    TensorDataType buf_scale, in_scale;
    for (size_t i = 0; i < 4 * num_layers; ++i) {
      auto&& opt = l.get_weights(i).get_optimizer();
      if (opt != nullptr) {
        auto& buf = opt->get_gradient_buffer(buf_scale, in_scale, true);
        El::Scale(buf_scale, buf);
        El::Axpy(in_scale, weights_grad_list[i], buf.Matrix());
      }
    }
  };

  // Return immediately if there is no local data
  if (mini_batch_size <= 0) {
    for (auto& dw : weights_grad_list) {
      El::Zero(dw);
    }
    send_weight_grads_to_optimizers();
    return;
  }

  // Make sure tensors are in correct format
  // Note: input_sequence_workspace and init_hidden_workspace have
  // already been setup in forward prop
  cudnn_objects.output_sequence_grad_workspace.SetSyncInfo(sync_info);
  cudnn_objects.input_sequence_grad_workspace.SetSyncInfo(sync_info);
  cudnn_objects.init_hidden_grad_workspace.SetSyncInfo(sync_info);
  cudnn_objects.output_sequence_grad_workspace.Resize(
    sequence_length * hidden_size,
    workspace_mini_batch_size);
  cudnn_objects.input_sequence_grad_workspace.Resize(sequence_length *
                                                       input_size,
                                                     workspace_mini_batch_size);
  cudnn_objects.init_hidden_grad_workspace.Resize(workspace_mini_batch_size *
                                                    hidden_size,
                                                  num_layers);
  El::Zero(cudnn_objects.output_sequence_grad_workspace);
  auto output_sequence_grad_workspace_ =
    cudnn_objects.output_sequence_grad_workspace(El::ALL,
                                                 El::IR(0, mini_batch_size));
  El::Copy(output_sequence_grad, output_sequence_grad_workspace_);

  // Initialize workspace for weight gradients
  // Note: Weights have already been packed in forward prop
  cudnn_objects.weights_grad_workspace.allocate(
    cudnn_objects.weights_workspace.size());
  CHECK_CUDA(cudaMemsetAsync(cudnn_objects.weights_grad_workspace.data(),
                             0,
                             cudnn_objects.weights_grad_workspace.size(),
                             stream));

#if !defined(LBANN_DEBUG) // Disable CUDA graphs

  // Compute hash with cuDNN function arguments
  size_t hash{0};
  hash = hash_combine(hash, cudnn_objects.gpu_sequence_lengths.data());
  hash =
    hash_combine(hash, cudnn_objects.input_sequence_workspace.LockedBuffer());
  hash =
    hash_combine(hash, cudnn_objects.input_sequence_grad_workspace.Buffer());
  hash = hash_combine(hash, cudnn_objects.init_hidden_workspace.LockedBuffer());
  hash = hash_combine(hash, cudnn_objects.init_hidden_grad_workspace.Buffer());
  hash =
    hash_combine(hash, cudnn_objects.output_sequence_workspace.LockedBuffer());
  hash =
    hash_combine(hash,
                 cudnn_objects.output_sequence_grad_workspace.LockedBuffer());
  hash = hash_combine(hash, cudnn_objects.weights_workspace.data());
  hash = hash_combine(hash, cudnn_objects.weights_grad_workspace.data());
  hash = hash_combine(hash, cudnn_objects.workspace.data());
  hash = hash_combine(hash, cudnn_objects.reserve_space.data());

  // Capture graph if not in cache
  if (cudnn_objects.backward_prop_graph_cache.count(workspace_mini_batch_size) <
        1 ||
      cudnn_objects.backward_prop_graph_cache[workspace_mini_batch_size]
          .first != hash) {
    cuda::Graph::begin_capture(stream);

#endif // !defined(LBANN_DEBUG)

    // cuDNN backward prop
    CHECK_CUDNN(cudnnRNNBackwardData_v8(
      handle,
      cudnn_objects.rnn_desc,
      cudnn_objects.gpu_sequence_lengths.data(),
      cudnn_objects.output_desc,
      cudnn_objects.output_sequence_workspace.LockedBuffer(),
      cudnn_objects.output_sequence_grad_workspace.LockedBuffer(),
      cudnn_objects.input_desc,
      cudnn_objects.input_sequence_grad_workspace.Buffer(),
      cudnn_objects.hidden_desc,
      cudnn_objects.init_hidden_workspace.LockedBuffer(),
      nullptr, // dhy
      cudnn_objects.init_hidden_grad_workspace.Buffer(),
      cudnn_objects.hidden_desc, // cDesc
      nullptr,                   // cx
      nullptr,                   // dcy
      nullptr,                   // dcx
      cudnn_objects.weights_workspace.size(),
      cudnn_objects.weights_workspace.data(),
      cudnn_objects.workspace.size(),
      cudnn_objects.workspace.data(),
      cudnn_objects.reserve_space.size(),
      cudnn_objects.reserve_space.data()));
    CHECK_CUDNN(cudnnRNNBackwardWeights_v8(
      handle,
      cudnn_objects.rnn_desc,
      CUDNN_WGRAD_MODE_ADD,
      cudnn_objects.gpu_sequence_lengths.data(),
      cudnn_objects.input_desc,
      cudnn_objects.input_sequence_workspace.LockedBuffer(),
      cudnn_objects.hidden_desc,
      cudnn_objects.init_hidden_workspace.LockedBuffer(),
      cudnn_objects.output_desc,
      cudnn_objects.output_sequence_workspace.LockedBuffer(),
      cudnn_objects.weights_grad_workspace.size(),
      cudnn_objects.weights_grad_workspace.data(),
      cudnn_objects.workspace.size(),
      cudnn_objects.workspace.data(),
      cudnn_objects.reserve_space.size(),
      cudnn_objects.reserve_space.data()));

#if !defined(LBANN_DEBUG) // Disable CUDA graphs

    // Finish capturing graph and update cache
    auto graph = cuda::Graph::end_capture(stream);
    auto& cache_pair =
      cudnn_objects.backward_prop_graph_cache[workspace_mini_batch_size];
    cache_pair.first = hash;
    cache_pair.second.update(graph);
  }

  // Launch CUDA graph
  cudnn_objects.backward_prop_graph_cache[workspace_mini_batch_size]
    .second.launch(stream);

#endif // !defined(LBANN_DEBUG)

  // Send gradients to optimizers
  unpack_cudnn_rnn_weights<TensorDataType>(
    handle,
    cudnn_objects.rnn_desc,
    sync_info,
    input_size,
    hidden_size,
    num_layers,
    cudnn_objects.weights_grad_workspace.data(),
    cudnn_objects.weights_grad_workspace.size(),
    weights_grad_list);
  send_weight_grads_to_optimizers();

  // Gradients w.r.t. input tensors
  // Note: We can't output directly to layer's input grad tensors
  // since they are allocated every step from the memory pool,
  // preventing us from reusing a CUDA graph.
  constexpr size_t one{1};
  El::LockedView(input_sequence_grad,
                 cudnn_objects.input_sequence_grad_workspace,
                 El::ALL,
                 El::IR(0, mini_batch_size));
  cuda::copy_tensor(
    stream,
    {mini_batch_size, num_layers, hidden_size},
    cudnn_objects.init_hidden_grad_workspace.LockedBuffer(),
    {hidden_size, workspace_mini_batch_size * hidden_size, one},
    init_hidden_grad.Buffer(),
    {static_cast<size_t>(init_hidden_grad.LDim()), hidden_size, one});
}

#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED

// =========================================================
// Builder
// =========================================================

namespace {

template <typename TensorDataType, data_layout Layout, El::Device Device>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR("Attempted to construct gru_layer with invalid parameters ",
                "(TensorDataType=",
                TypeName<TensorDataType>(),
                ", ",
                "Layout=",
                to_string(Layout),
                ", ",
                "Device=",
                to_string(Device),
                ")");
    return nullptr;
  }
};

template <typename TensorDataType>
struct Builder<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    constexpr auto Layout = data_layout::DATA_PARALLEL;
    constexpr auto Device = El::Device::CPU;
#ifdef LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
    using LayerType = gru_layer<TensorDataType, Layout, Device>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
#else
    LBANN_ERROR("CPU gru_layer requires oneDNN "
                "(TensorDataType=",
                TypeName<TensorDataType>(),
                ", ",
                "Layout=",
                to_string(Layout),
                ", ",
                "Device=",
                to_string(Device),
                ")");
    return nullptr;
#endif // LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
  }
};

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
struct Builder<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    constexpr auto Layout = data_layout::DATA_PARALLEL;
    constexpr auto Device = El::Device::GPU;
#ifdef LBANN_GRU_LAYER_CUDNN_SUPPORTED
    using LayerType = gru_layer<TensorDataType, Layout, Device>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
#else
    LBANN_ERROR("GPU gru_layer requires at least CUDA 11.0 and cuDNN 8.0.4 "
                "(TensorDataType=",
                TypeName<TensorDataType>(),
                ", ",
                "Layout=",
                to_string(Layout),
                ", ",
                "Device=",
                to_string(Device),
                ")");
    return nullptr;
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED
  }
};
#endif // LBANN_HAS_GPU

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_gru_layer_from_pbuf(lbann_comm* comm,
                          lbann_data::Layer const& proto_layer)
{
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, gru);
  const auto& params = proto_layer.gru();
  const size_t num_layers =
    (params.has_num_layers() ? params.num_layers().value() : 1);
  return BuilderType::Build(params.hidden_size(), num_layers);
}

// =========================================================
// Explicit template instantiation
// =========================================================

#ifdef LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
#define PROTO(T)                                                               \
  template class gru_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>;
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
#ifdef LBANN_GRU_LAYER_CUDNN_SUPPORTED
#define PROTO(T)                                                               \
  template class gru_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>;
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED

#define PROTO_DEVICE(T, Device) LBANN_LAYER_BUILDER_ETI(gru, T, Device)
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

} // namespace lbann
