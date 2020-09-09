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

#define LBANN_GRU_LAYER_INSTANTIATE
#include "lbann/layers/learning/gru.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/proto/proto_common.hpp"
#include <layers.pb.h>

namespace lbann {

// ---------------------------------------------
// Life cycle
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>::gru_layer(lbann_comm* comm, size_t hidden_size)
  : data_type_layer<TensorDataType>(comm),
    m_hidden_size{hidden_size} {
  this->m_expected_num_parent_layers = 2;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>::gru_layer(const gru_layer& other)
  : data_type_layer<TensorDataType>(other),
    m_hidden_size{other.m_hidden_size}
#ifdef LBANN_HAS_CUDNN
  , m_rnn_cudnn_desc{other.m_rnn_cudnn_desc},
    m_input_cudnn_desc{other.m_input_cudnn_desc},
    m_output_cudnn_desc{other.m_output_cudnn_desc},
    m_hidden_cudnn_desc{other.m_hidden_cudnn_desc},
    m_weights_cudnn_desc{other.m_weights_cudnn_desc}
#endif // LBANN_HAS_CUDNN
{
#ifdef LBANN_HAS_CUDNN
  /// @todo Copy m_cudnn_reserve_space?
#endif // LBANN_HAS_CUDNN
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>& gru_layer<TensorDataType, Layout, Device>
::operator=(const gru_layer& other) {
  data_type_layer<TensorDataType>::operator=(other);
  m_hidden_size = other.m_hidden_size;
#ifdef LBANN_HAS_CUDNN
  m_rnn_cudnn_desc = other.m_rnn_cudnn_desc;
  m_input_cudnn_desc = other.m_input_cudnn_desc;
  m_output_cudnn_desc = other.m_output_cudnn_desc;
  m_hidden_cudnn_desc = other.m_hidden_cudnn_desc;
  m_weights_cudnn_desc = other.m_weights_cudnn_desc;
  /// @todo Copy m_cudnn_reserve_space?
#endif // LBANN_HAS_CUDNN
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType,Layout,Device>*
gru_layer<TensorDataType,Layout,Device>
::copy() const
{
  return new gru_layer(*this);
}

// ---------------------------------------------
// Query functions
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string
gru_layer<TensorDataType,Layout,Device>
::get_type() const
{
  return "GRU";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
gru_layer<TensorDataType,Layout,Device>
::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
gru_layer<TensorDataType,Layout,Device>
::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
gru_layer<TensorDataType,Layout,Device>
::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Hidden size", m_hidden_size);
  return desc;
}

// ---------------------------------------------
// Setup
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  const int sequence_length = this->get_input_dims()[0];
  if (static_cast<size_t>(this->get_input_size(1)) != m_hidden_size) {
    LBANN_ERROR(
      this->get_type()," layer \"",this->get_name(),"\" ",
      "has an invalid input tensor for the initial hidden state");
  }
  const std::vector<int> output_dims = {sequence_length, static_cast<int>(m_hidden_size)};
  this->set_output_dims(output_dims);
}

namespace {

std::string get_gru_weight_name(size_t index) {
  switch (index) {
  case 0:
    return "ih_matrix";
  case 1:
    return "hh_matrix";
  case 2:
    return "ih_bias";
  case 3:
    return "hh_bias";
  default:
    LBANN_ERROR("unknown index for GRU weight (",index,")");
  }
}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>
::setup_data(size_t max_mini_batch_size) {
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  const size_t sequence_length = this->get_input_dims()[0];
  const size_t input_size = this->get_input_size(0) / sequence_length;

  // Construct default weights if needed
  if (!this->has_weights()) {
    const auto scale = El::To<TensorDataType>(1./std::sqrt(m_hidden_size));
    for (size_t i=0; i<4; ++i) {
      auto w = make_unique<data_type_weights<TensorDataType>>(this->get_comm());
      auto init = make_unique<uniform_initializer<TensorDataType>>(-scale, scale);
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_" + get_gru_weight_name(i));
      w->set_initializer(std::move(init));
      w->set_optimizer(std::move(opt));
      this->set_weights(i, w.get());
      this->m_model->add_weights(std::move(w));
    }
  }
  if (this->num_weights() != 4) {
    LBANN_ERROR(
      "attempted to setup ",
      this->get_type()," layer \"",this->get_name(),"\" ",
      "with an invalid number of weights ",
      "(expected 4, found ",this->num_weights(),")");
  }

  // Setup weight dimensions and distribution
  auto& ih_matrix = this->get_weights(0);
  auto& hh_matrix = this->get_weights(1);
  auto& ih_bias = this->get_weights(2);
  auto& hh_bias = this->get_weights(3);
  ih_matrix.set_dims({static_cast<int>(3*m_hidden_size)}, {static_cast<int>(input_size)});
  hh_matrix.set_dims({static_cast<int>(3*m_hidden_size)}, {static_cast<int>(m_hidden_size)});
  ih_bias.set_dims({static_cast<int>(3*m_hidden_size)});
  hh_bias.set_dims({static_cast<int>(3*m_hidden_size)});
  auto dist = this->get_prev_activations().DistData();
  dist.colDist = El::STAR;
  dist.rowDist = El::STAR;
  ih_matrix.set_matrix_distribution(dist);
  hh_matrix.set_matrix_distribution(dist);
  ih_bias.set_matrix_distribution(dist);
  hh_bias.set_matrix_distribution(dist);

}

#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::setup_gpu() {

  // Dimensions
  const size_t sequence_length = this->get_input_dims(0)[0];
  const size_t input_size = this->get_input_size(0) / sequence_length;

  // GPU objects
  auto&& handle = cudnn::get_handle();
  auto data_type = cudnn::get_data_type<TensorDataType>();

  // RNN descriptor
  size_t dropout_state_size;
  CHECK_CUDNN(cudnnDropoutGetStatesSize(handle, &dropout_state_size));
  cudnn::DropoutDescriptor dropout_desc(0.f, nullptr, dropout_state_size, 0);
  m_rnn_cudnn_desc.set(
    m_hidden_size,
    1,  // num_layers
    dropout_desc,
    CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL,
    CUDNN_GRU,
    CUDNN_RNN_ALGO_STANDARD,
    data_type);

  // Input and output tensor descriptors
  m_input_cudnn_desc.set(data_type, 1, input_size, 1);
  m_output_cudnn_desc.set(data_type, 1, m_hidden_size, 1);
  m_hidden_cudnn_desc.set(data_type, 1, 1, m_hidden_size);

  // Packed weights descriptor
  size_t weights_size;
  CHECK_CUDNN(
    cudnnGetRNNParamsSize(
      handle,
      m_rnn_cudnn_desc,
      m_input_cudnn_desc,
      &weights_size,
      data_type));
  m_weights_cudnn_desc.set(
    data_type,
    CUDNN_TENSOR_NCHW,
    weights_size / sizeof(TensorDataType),
    1,
    1);

}
#endif // LBANN_HAS_CUDNN

// ---------------------------------------------
// Forward prop
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::fp_compute() {
  fp_compute_impl(*this);
}

#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType>
hydrogen::simple_buffer<El::byte, El::Device::GPU> pack_cudnn_rnn_weights(
  const cudnnHandle_t& handle,
  const cudnn::RNNDescriptor& rnn_desc,
  const cudnn::TensorDescriptor& input_desc,
  const cudnn::FilterDescriptor& weights_desc,
  const El::SyncInfo<El::Device::GPU>& sync_info,
  size_t input_size,
  size_t hidden_size,
  const El::Matrix<TensorDataType,El::Device::GPU>& ih_matrix,
  const El::Matrix<TensorDataType,El::Device::GPU>& hh_matrix,
  const El::Matrix<TensorDataType,El::Device::GPU>& ih_bias,
  const El::Matrix<TensorDataType,El::Device::GPU>& hh_bias) {

  // Allocate buffer for packed weights
  size_t packed_weights_size;
  CHECK_CUDNN(
    cudnnGetRNNParamsSize(
      handle,
      rnn_desc,
      input_desc,
      &packed_weights_size,
      cudnn::get_data_type<TensorDataType>()));
  hydrogen::simple_buffer<El::byte, El::Device::GPU> packed_weights(packed_weights_size, sync_info);

  // Construct objects
  static cudnn::FilterDescriptor result_weights_desc;
  result_weights_desc.create();
  TensorDataType* packed_weights_ptr;
  El::Matrix<TensorDataType,El::Device::GPU> packed_weights_view;
  packed_weights_view.SetSyncInfo(sync_info);

  // Copy values from ih_matrix
  for (auto i : {0, 1, 2}) {
    CHECK_CUDNN(
      cudnnGetRNNLinLayerMatrixParams(
        handle,
        rnn_desc,
        0, // pseudoLayer
        input_desc,
        weights_desc,
        packed_weights.data(),
        i, // linLayerID
        result_weights_desc,
        reinterpret_cast<void**>(&packed_weights_ptr)));
    packed_weights_view.Attach(input_size, hidden_size, packed_weights_ptr, input_size);
    El::Transpose(
      ih_matrix(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL),
      packed_weights_view,
      false);
  }

  // Copy values from hh_matrix
  for (auto i : {0, 1, 2}) {
    CHECK_CUDNN(
      cudnnGetRNNLinLayerMatrixParams(
        handle,
        rnn_desc,
        0, // pseudoLayer
        input_desc,
        weights_desc,
        packed_weights.data(),
        3+i, // linLayerID
        result_weights_desc,
        reinterpret_cast<void**>(&packed_weights_ptr)));
    packed_weights_view.Attach(hidden_size, hidden_size, packed_weights_ptr, hidden_size);
    El::Transpose(
      hh_matrix(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL),
      packed_weights_view,
      false);
  }

  // Copy values from ih_bias
  for (auto i : {0, 1, 2}) {
    CHECK_CUDNN(
      cudnnGetRNNLinLayerBiasParams(
        handle,
        rnn_desc,
        0, // pseudoLayer
        input_desc,
        weights_desc,
        packed_weights.data(),
        i, // linLayerID
        result_weights_desc,
        reinterpret_cast<void**>(&packed_weights_ptr)));
    packed_weights_view.Attach(hidden_size, 1, packed_weights_ptr, hidden_size);
    El::Copy(
      ih_bias(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL),
      packed_weights_view);
  }

  // Copy values from hh_bias
  for (auto i : {0, 1, 2}) {
    CHECK_CUDNN(
      cudnnGetRNNLinLayerBiasParams(
        handle,
        rnn_desc,
        0, // pseudoLayer
        input_desc,
        weights_desc,
        packed_weights.data(),
        3+i, // linLayerID
        result_weights_desc,
        reinterpret_cast<void**>(&packed_weights_ptr)));
    packed_weights_view.Attach(hidden_size, 1, packed_weights_ptr, hidden_size);
    El::Copy(
      ih_bias(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL),
      packed_weights_view);
  }

  return packed_weights;
}
#endif // LBANN_HAS_CUDNN

#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType>
void fp_compute_impl(
  gru_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::GPU>& l) {

  // Matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input_sequence
    = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& local_init_hidden
    = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  auto& local_output_sequence
    = dynamic_cast<LocalMat&>(l.get_local_activations());
  const auto& ih_matrix
    = dynamic_cast<const LocalMat&>(l.weights_values(0).LockedMatrix());
  const auto& hh_matrix
    = dynamic_cast<const LocalMat&>(l.weights_values(1).LockedMatrix());
  const auto& ih_bias
    = dynamic_cast<const LocalMat&>(l.weights_values(2).LockedMatrix());
  const auto& hh_bias
    = dynamic_cast<const LocalMat&>(l.weights_values(3).LockedMatrix());

  // Dimensions
  const size_t sequence_length = l.get_input_dims(0)[0];
  const size_t mini_batch_size = local_input_sequence.Width();
  const size_t input_size = l.get_input_size(0) / sequence_length;
  const size_t hidden_size = l.m_hidden_size;

  // Return immediately if there is no work to be done
  if (mini_batch_size <= 0) {
    return;
  }

  // GPU objects
  auto&& sync_info = local_input_sequence.GetSyncInfo();
  auto&& handle = cudnn::get_handle();
  const auto data_type = cudnn::get_data_type<TensorDataType>();

  // Configure input and output tensor descriptors
  auto& input_desc = l.m_input_cudnn_desc;
  auto& output_desc = l.m_output_cudnn_desc;
  auto& hidden_desc = l.m_hidden_cudnn_desc;
  input_desc.set(data_type, mini_batch_size, input_size, 1);
  output_desc.set(data_type, mini_batch_size, hidden_size, 1);
  hidden_desc.set(data_type, 1, mini_batch_size, hidden_size);
  std::vector<cudnnTensorDescriptor_t> input_desc_list(sequence_length, input_desc);
  std::vector<cudnnTensorDescriptor_t> output_desc_list(sequence_length, output_desc);

  // Reorder input tensor dims
  // Note: cuDNN uses sequence_length x mini_batch_size x hidden_size
  /// @todo Consider custom kernel
  LocalMat input_sequence_workspace, output_sequence_workspace;
  input_sequence_workspace.SetSyncInfo(sync_info);
  output_sequence_workspace.SetSyncInfo(sync_info);
  input_sequence_workspace.Resize(mini_batch_size*input_size, sequence_length);
  output_sequence_workspace.Resize(mini_batch_size*hidden_size, sequence_length);
  for (size_t i=0; i<sequence_length; ++i) {
    const auto input_sequence_view
      = local_input_sequence(El::IR(i*input_size, (i+1)*input_size), El::ALL);
    LocalMat input_sequence_workspace_view(
      input_size,
      mini_batch_size,
      input_sequence_workspace.Buffer(0, i),
      input_size);
    input_sequence_workspace_view.SetSyncInfo(sync_info);
    El::Copy(input_sequence_view, input_sequence_workspace_view);
  }

  // Pack weights into workspace buffer
  auto weights_workspace = pack_cudnn_rnn_weights(
    handle,
    l.m_rnn_cudnn_desc,
    input_desc,
    l.m_weights_cudnn_desc,
    sync_info,
    input_size,
    hidden_size,
    ih_matrix,
    hh_matrix,
    ih_bias,
    hh_bias);

  // Allocate cuDNN workspace buffers
  /// @todo Handle synchronization for m_cudnn_reserve_space
  size_t cudnn_workspace_size, cudnn_reserve_space_size;
  CHECK_CUDNN(
    cudnnGetRNNWorkspaceSize(
      handle,
      l.m_rnn_cudnn_desc,
      sequence_length,
      input_desc_list.data(),
      &cudnn_workspace_size));
  CHECK_CUDNN(
    cudnnGetRNNTrainingReserveSize(
      handle,
      l.m_rnn_cudnn_desc,
      sequence_length,
      input_desc_list.data(),
      &cudnn_reserve_space_size));
  using ByteBuffer = hydrogen::simple_buffer<El::byte, El::Device::GPU>;
  ByteBuffer cudnn_workspace(cudnn_workspace_size, sync_info);
  l.m_cudnn_reserve_space.allocate(cudnn_reserve_space_size);

  // Launch cuDNN GRU
  CHECK_CUDNN(
    cudnnRNNForwardTraining(
      handle,
      l.m_rnn_cudnn_desc,
      sequence_length,
      input_desc_list.data(),
      input_sequence_workspace.LockedBuffer(),
      hidden_desc,
      local_init_hidden.LockedBuffer(),
      hidden_desc,  // cxDesc
      nullptr,      // cx
      l.m_weights_cudnn_desc,
      weights_workspace.data(),
      output_desc_list.data(),
      output_sequence_workspace.Buffer(),
      hidden_desc,  // hyDesc
      nullptr,      // hy
      hidden_desc,  // cyDesc
      nullptr,      // cy
      cudnn_workspace.data(),
      cudnn_workspace.size(),
      l.m_cudnn_reserve_space.data(),
      l.m_cudnn_reserve_space.size()));

  // Reorder output tensor dims
  // Note: cuDNN uses sequence_length x mini_batch_size x hidden_size
  /// @todo Consider custom kernel
  for (size_t i=0; i<sequence_length; ++i) {
    LocalMat output_sequence_workspace_view(
      hidden_size,
      mini_batch_size,
      output_sequence_workspace.LockedBuffer(0, i),
      hidden_size);
    output_sequence_workspace_view.SetSyncInfo(sync_info);
    auto output_sequence_view
      = local_output_sequence(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL);
    El::Copy(output_sequence_workspace_view, output_sequence_view);
  }

}
#endif // LBANN_HAS_CUDNN

// ---------------------------------------------
// Back prop
// ---------------------------------------------

/// @todo Implement

// ---------------------------------------------
// Builder
// ---------------------------------------------

namespace
{

template <typename TensorDataType, data_layout Layout, El::Device Device>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR(
      "Attempted to construct gru_layer with invalid parameters ",
      "(TensorDataType=",TypeName<TensorDataType>(),", ",
      "Layout=",to_string(Layout),", ",
      "Device=",to_string(Device),")");
    return nullptr;
  }
};

#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType>
struct Builder<TensorDataType,data_layout::DATA_PARALLEL,El::Device::GPU>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    constexpr auto Layout = data_layout::DATA_PARALLEL;
    constexpr auto Device = El::Device::GPU;
    using LayerType = gru_layer<TensorDataType,Layout,Device>;
    return make_unique<LayerType>(std::forward<Args>(args)...);
  }
};
#endif // LBANN_HAS_CUDNN

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_gru_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const& proto_layer)
{
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, gru);
  const auto& params = proto_layer.gru();
  return BuilderType::Build(comm, params.hidden_size());
}

// ---------------------------------------------
// Explicit template instantiation
// ---------------------------------------------

/// @todo CPU implementation
#ifdef LBANN_HAS_CUDNN
#define PROTO(T)                                                        \
  template class gru_layer<                                             \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_HAS_CUDNN

#define PROTO_DEVICE(T, Device)                 \
  LBANN_LAYER_BUILDER_ETI(gru, T, Device)
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

} // namespace lbann
