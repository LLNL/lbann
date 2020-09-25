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
  , m_hidden_cudnn_desc{other.m_hidden_cudnn_desc}
#endif // LBANN_HAS_CUDNN
{
#ifdef LBANN_HAS_CUDNN
  /// @todo Copy cuDNN objects?
#endif // LBANN_HAS_CUDNN
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>& gru_layer<TensorDataType, Layout, Device>
::operator=(const gru_layer& other) {
  data_type_layer<TensorDataType>::operator=(other);
  m_hidden_size = other.m_hidden_size;
#ifdef LBANN_HAS_CUDNN
  m_hidden_cudnn_desc = other.m_hidden_cudnn_desc;
  /// @todo Copy cuDNN objects?
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
  const int sequence_length = this->get_input_dims(0)[0];
  if (static_cast<size_t>(this->get_input_size(1)) != m_hidden_size) {
    LBANN_ERROR(
      this->get_type()," layer \"",this->get_name(),"\" ",
      "has an invalid input tensor for the initial hidden state");
  }
  const std::vector<int> output_dims = {sequence_length, static_cast<int>(m_hidden_size)};
  this->set_output_dims(output_dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>
::setup_data(size_t max_mini_batch_size) {
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  const size_t sequence_length = this->get_input_dims()[0];
  const size_t input_size = this->get_input_size(0) / sequence_length;

  // Construct default weights if needed
  if (!this->has_weights()) {
    const std::vector<std::string> weight_names
      = {"ih_matrix", "hh_matrix", "ih_bias", "hh_bias"};
    this->set_num_weights(4);
    const auto scale = El::To<TensorDataType>(1./std::sqrt(m_hidden_size));
    for (size_t i=0; i<4; ++i) {
      auto w = make_unique<data_type_weights<TensorDataType>>(this->get_comm());
      auto init = make_unique<uniform_initializer<TensorDataType>>(-scale, scale);
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_" + weight_names[i]);
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
  // static hydrogen::simple_buffer<El::byte, El::Device::GPU> dummy_buffer(dropout_state_size);
  // m_dropout_cudnn_desc.set(0, dummy_buffer.data(), dropout_state_size, 0);
  m_dropout_cudnn_desc.set(0, nullptr, 0, 0);
  m_rnn_cudnn_desc.set(
    CUDNN_RNN_ALGO_STANDARD,
    CUDNN_GRU,
    CUDNN_RNN_DOUBLE_BIAS,
    CUDNN_UNIDIRECTIONAL,
    CUDNN_LINEAR_INPUT,
    data_type,
    data_type,
    cudnn::get_default_convolution_math_type(),
    input_size,
    m_hidden_size,
    m_hidden_size,  // proj_size
    1,              // num_layers
    m_dropout_cudnn_desc,
    CUDNN_RNN_PADDED_IO_ENABLED);

}
#endif // LBANN_HAS_CUDNN

// ---------------------------------------------
// Forward prop
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::fp_compute() {
  fp_compute_impl(*this);
}

namespace {
#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType>
hydrogen::simple_buffer<El::byte, El::Device::GPU> pack_cudnn_rnn_weights(
  const cudnnHandle_t& handle,
  const cudnn::RNNDescriptor& rnn_desc,
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
    cudnnGetRNNWeightSpaceSize(
      handle,
      rnn_desc,
      &packed_weights_size));
  hydrogen::simple_buffer<El::byte, El::Device::GPU> packed_weights(packed_weights_size, sync_info);

  // Construct objects
  static cudnn::TensorDescriptor matrix_desc, bias_desc;
  El::Matrix<TensorDataType,El::Device::GPU> packed_weights_view;
  packed_weights_view.SetSyncInfo(sync_info);

  // Function to get pointers in packed weights buffer
  using PtrPair = std::pair<TensorDataType*,TensorDataType*>;
  auto get_ptrs = [&] (size_t id) -> PtrPair {
    PtrPair ptrs;
    matrix_desc.create();
    bias_desc.create();
    CHECK_CUDNN(
      cudnnGetRNNWeightParams(
        handle,
        rnn_desc,
        0,  // pseudoLayer
        packed_weights.size(),
        packed_weights.data(),
        id,
        matrix_desc,
        reinterpret_cast<void**>(&ptrs.first),
        bias_desc,
        reinterpret_cast<void**>(&ptrs.second)));
    return ptrs;
  };

  // Copy from ih_matrix
  for (auto i : {0, 1, 2}) {
    packed_weights_view.Attach(
      input_size,
      hidden_size,
      get_ptrs(i).first,
      input_size);
    El::Transpose(
      ih_matrix(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL),
      packed_weights_view,
      false);
  }

  // Copy from hh_matrix
  for (auto i : {0, 1, 2}) {
    packed_weights_view.Attach(
      hidden_size,
      hidden_size,
      get_ptrs(3+i).first,
      hidden_size);
    El::Transpose(
      hh_matrix(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL),
      packed_weights_view,
      false);
  }

  // Copy from ih_bias
  for (auto i : {0, 1, 2}) {
    packed_weights_view.Attach(
      hidden_size,
      1,
      get_ptrs(i).second,
      hidden_size);
    El::Copy(
      ih_bias(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL),
      packed_weights_view);
  }

  // Copy from hh_bias
  for (auto i : {0, 1, 2}) {
    packed_weights_view.Attach(
      hidden_size,
      1,
      get_ptrs(3+i).second,
      hidden_size);
    El::Copy(
      hh_bias(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL),
      packed_weights_view);
  }

  return packed_weights;
}
#endif // LBANN_HAS_CUDNN
} // namespace <anon>

#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType>
void fp_compute_impl(
  gru_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::GPU>& l) {
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  using ByteBuffer = hydrogen::simple_buffer<El::byte, El::Device::GPU>;

  // Matrices
  const auto& input_sequence
    = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& init_hidden
    = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  auto& output_sequence
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
  const size_t mini_batch_size = input_sequence.Width();
  const size_t sequence_length = l.get_input_dims(0)[0];
  const size_t input_size = l.get_input_size(0) / sequence_length;
  const size_t hidden_size = l.m_hidden_size;

  // Return immediately if there is no local data
  if (mini_batch_size <= 0) {
    return;
  }

  // GPU objects
  auto&& sync_info = input_sequence.GetSyncInfo();
  auto&& stream = sync_info.Stream();
  auto&& handle = cudnn::get_handle();
  auto&& rnn_desc = l.m_rnn_cudnn_desc;
  const auto data_type = cudnn::get_data_type<TensorDataType>();

  // Configure input and output tensor descriptors
  auto& input_desc = l.m_input_cudnn_desc;
  auto& output_desc = l.m_output_cudnn_desc;
  auto& hidden_desc = l.m_hidden_cudnn_desc;
  std::vector<int> sequence_lengths(mini_batch_size, sequence_length);
  static const TensorDataType zero{El::TypeTraits<TensorDataType>::Zero()};
  input_desc.set(
    data_type,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
    sequence_length,
    mini_batch_size,
    input_size,
    sequence_lengths.data(),
    const_cast<void*>(reinterpret_cast<const void*>(&zero)));
  output_desc.set(
    data_type,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
    sequence_length,
    mini_batch_size,
    hidden_size,
    sequence_lengths.data(),
    const_cast<void*>(reinterpret_cast<const void*>(&zero)));
  hidden_desc.set(data_type, 1, mini_batch_size, hidden_size);

  // Pack weights into workspace buffer
  auto packed_weights = pack_cudnn_rnn_weights(
    handle,
    rnn_desc,
    sync_info,
    input_size,
    hidden_size,
    ih_matrix,
    hh_matrix,
    ih_bias,
    hh_bias);

  // Allocate cuDNN workspace buffers
  size_t cudnn_workspace_size, cudnn_reserve_space_size;
  CHECK_CUDNN(
    cudnnGetRNNTempSpaceSizes(
      handle,
      rnn_desc,
      CUDNN_FWD_MODE_TRAINING,
      input_desc,
      &cudnn_workspace_size,
      &cudnn_reserve_space_size));
  ByteBuffer cudnn_workspace(cudnn_workspace_size, sync_info);
  if (l.m_cudnn_reserve_space.size() < cudnn_reserve_space_size) {
    /// @todo Handle synchronization
    l.m_cudnn_reserve_space.allocate(cudnn_reserve_space_size);
  }
  if (l.m_gpu_sequence_lengths.size() < mini_batch_size) {
    /// @todo Handle synchronization
    l.m_gpu_sequence_lengths.allocate(mini_batch_size);
    std::vector<int32_t> cpu_sequence_lengths(mini_batch_size, sequence_length);
    CHECK_CUDA(
      cudaMemcpyAsync(
        l.m_gpu_sequence_lengths.data(),
        cpu_sequence_lengths.data(),
        cpu_sequence_lengths.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // Make sure tensors are formatted correctly
  // Note (tym 9/24/20): cuDNNDataDescriptor has an option for
  // CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED, but I haven't been
  // able to get it to work with CUDA 10.2.89 and cuDNN 8.0.2.
  LocalMat input_sequence_workspace, output_sequence_workspace;
  LocalMat init_hidden_workspace;
  input_sequence_workspace.SetSyncInfo(sync_info);
  output_sequence_workspace.SetSyncInfo(sync_info);
  init_hidden_workspace.SetSyncInfo(sync_info);
  input_sequence_workspace.Resize(mini_batch_size*input_size, sequence_length);
  output_sequence_workspace.Resize(mini_batch_size*hidden_size, sequence_length);
  constexpr size_t one{1};
  cuda::copy_tensor(
    stream,
    {mini_batch_size, sequence_length, input_size},
    input_sequence.LockedBuffer(),
    {static_cast<size_t>(input_sequence.LDim()), input_size, one},
    input_sequence_workspace.Buffer(),
    {input_size, mini_batch_size*input_size, one});
  if (init_hidden.Contiguous()) {
    El::LockedView(init_hidden_workspace, init_hidden);
  }
  else {
    El::Copy(init_hidden, init_hidden_workspace);
  }

  // Launch cuDNN GRU
  cuda::Graph::begin_capture(stream);
  CHECK_CUDNN(
    cudnnRNNForward(
      handle,
      rnn_desc,
      CUDNN_FWD_MODE_TRAINING,
      l.m_gpu_sequence_lengths.data(),
      input_desc,
      input_sequence_workspace.LockedBuffer(),
      output_desc,
      output_sequence_workspace.Buffer(),
      hidden_desc,
      init_hidden_workspace.LockedBuffer(),
      nullptr,      // hy
      hidden_desc,  // cDesc
      nullptr,      // cx
      nullptr,      // cy
      packed_weights.size(),
      packed_weights.data(),
      cudnn_workspace.size(),
      cudnn_workspace.data(),
      l.m_cudnn_reserve_space.size(),
      l.m_cudnn_reserve_space.data()));
  auto graph = cuda::Graph::end_capture(stream);
  l.m_graph_forward_prop.update(graph);
  l.m_graph_forward_prop.launch(stream);

  // Reorder output tensor dims
  // Note (tym 9/24/20): cuDNNDataDescriptor has an option for
  // CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED, but I haven't been
  // able to get it to work with CUDA 10.2.89 and cuDNN 8.0.2.
  cuda::copy_tensor(
    stream,
    {mini_batch_size, sequence_length, hidden_size},
    output_sequence_workspace.LockedBuffer(),
    {hidden_size, mini_batch_size*hidden_size, one},
    output_sequence.Buffer(),
    {static_cast<size_t>(output_sequence.LDim()), hidden_size, one});

}
#endif // LBANN_HAS_CUDNN

// ---------------------------------------------
// Back prop
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::bp_compute() {
  bp_compute_impl(*this);
}

namespace {
#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType>
void unpack_cudnn_rnn_weights(
  const cudnnHandle_t& handle,
  const cudnn::RNNDescriptor& rnn_desc,
  const El::SyncInfo<El::Device::GPU>& sync_info,
  size_t input_size,
  size_t hidden_size,
  const TensorDataType* packed_weights_buffer,
  size_t packed_weights_size,
  El::Matrix<TensorDataType,El::Device::GPU>& ih_matrix,
  El::Matrix<TensorDataType,El::Device::GPU>& hh_matrix,
  El::Matrix<TensorDataType,El::Device::GPU>& ih_bias,
  El::Matrix<TensorDataType,El::Device::GPU>& hh_bias) {

  // Construct objects
  static cudnn::TensorDescriptor matrix_desc, bias_desc;
  El::Matrix<TensorDataType,El::Device::GPU> packed_weights_view;
  packed_weights_view.SetSyncInfo(sync_info);

  // Function to get pointers in packed weights buffer
  using PtrPair = std::pair<TensorDataType*,TensorDataType*>;
  auto get_ptrs = [&] (size_t id) -> PtrPair {
    PtrPair ptrs;
    matrix_desc.create();
    bias_desc.create();
    CHECK_CUDNN(
      cudnnGetRNNWeightParams(
        handle,
        rnn_desc,
        0,  // pseudoLayer
        packed_weights_size,
        packed_weights_buffer,
        id,
        matrix_desc,
        reinterpret_cast<void**>(&ptrs.first),
        bias_desc,
        reinterpret_cast<void**>(&ptrs.second)));
    return ptrs;
  };

  // Copy from ih_matrix
  for (auto i : {0, 1, 2}) {
    packed_weights_view.LockedAttach(
      input_size,
      hidden_size,
      get_ptrs(i).first,
      input_size);
    auto ih_matrix_view = ih_matrix(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL);
    El::Transpose(packed_weights_view, ih_matrix_view, false);
  }

  // Copy from hh_matrix
  for (auto i : {0, 1, 2}) {
    packed_weights_view.LockedAttach(
      hidden_size,
      hidden_size,
      get_ptrs(3+i).first,
      hidden_size);
    auto hh_matrix_view = hh_matrix(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL);
    El::Transpose(packed_weights_view, hh_matrix_view, false);
  }

  // Copy from ih_bias
  for (auto i : {0, 1, 2}) {
    packed_weights_view.LockedAttach(
      hidden_size,
      1,
      get_ptrs(i).second,
      hidden_size);
    auto ih_bias_view = ih_bias(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL);
    El::Copy(packed_weights_view, ih_bias_view);
  }

  // Copy from hh_bias
  for (auto i : {0, 1, 2}) {
    packed_weights_view.LockedAttach(
      hidden_size,
      1,
      get_ptrs(3+i).second,
      hidden_size);
    auto hh_bias_view = hh_bias(El::IR(i*hidden_size, (i+1)*hidden_size), El::ALL);
    El::Copy(packed_weights_view, hh_bias_view);
  }

}
#endif // LBANN_HAS_CUDNN
} // namespace <anon>

#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType>
void bp_compute_impl(
  gru_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::GPU>& l) {
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  using ByteBuffer = hydrogen::simple_buffer<El::byte, El::Device::GPU>;

#if 0

  // Matrices
  const auto& input_sequence
    = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& init_hidden
    = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  const auto& output_sequence
    = dynamic_cast<const LocalMat&>(l.get_local_activations());
  const auto& output_sequence_grad
    = dynamic_cast<const LocalMat&>(l.get_local_prev_error_signals());
  auto& input_sequence_grad
    = dynamic_cast<LocalMat&>(l.get_local_error_signals(0));
  auto& init_hidden_grad
    = dynamic_cast<LocalMat&>(l.get_local_error_signals(1));
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
  const size_t mini_batch_size = input_sequence.Width();
  const size_t input_size = l.get_input_size(0) / sequence_length;
  const size_t hidden_size = l.m_hidden_size;

  // GPU objects
  auto&& sync_info = input_sequence.GetSyncInfo();
  auto&& stream = sync_info.Stream();
  auto&& handle = cudnn::get_handle();

  // Define closure to send weight gradients to optimizers
  LocalMat ih_matrix_grad, hh_matrix_grad, ih_bias_grad, hh_bias_grad;
  ih_matrix_grad.SetSyncInfo(sync_info);
  hh_matrix_grad.SetSyncInfo(sync_info);
  ih_bias_grad.SetSyncInfo(sync_info);
  hh_bias_grad.SetSyncInfo(sync_info);
  ih_matrix_grad.Resize(3*hidden_size, input_size);
  hh_matrix_grad.Resize(3*hidden_size, hidden_size);
  ih_bias_grad.Resize(3*hidden_size, 1);
  hh_bias_grad.Resize(3*hidden_size, 1);
  auto send_weight_grads_to_optimizers = [&] () {
    TensorDataType buf_scale, in_scale;
    auto&& ih_matrix_opt = l.get_weights(0).get_optimizer();
    auto&& hh_matrix_opt = l.get_weights(1).get_optimizer();
    auto&& ih_bias_opt = l.get_weights(2).get_optimizer();
    auto&& hh_bias_opt = l.get_weights(3).get_optimizer();
    if (ih_matrix_opt != nullptr) {
      auto& buf = ih_matrix_opt->get_gradient_buffer(buf_scale, in_scale, true);
      El::Scale(buf_scale, buf);
      El::Axpy(in_scale, ih_matrix_grad, buf.Matrix());
    }
    if (hh_matrix_opt != nullptr) {
      auto& buf = hh_matrix_opt->get_gradient_buffer(buf_scale, in_scale, true);
      El::Scale(buf_scale, buf);
      El::Axpy(in_scale, hh_matrix_grad, buf.Matrix());
    }
    if (ih_bias_opt != nullptr) {
      auto& buf = ih_bias_opt->get_gradient_buffer(buf_scale, in_scale, true);
      El::Scale(buf_scale, buf);
      El::Axpy(in_scale, ih_bias_grad, buf.Matrix());
    }
    if (hh_bias_opt != nullptr) {
      auto& buf = hh_bias_opt->get_gradient_buffer(buf_scale, in_scale, true);
      El::Scale(buf_scale, buf);
      El::Axpy(in_scale, hh_bias_grad, buf.Matrix());
    }
  };

  // Return immediately if there is no local data
  if (mini_batch_size <= 0) {
    El::Zero(ih_matrix_grad);
    El::Zero(hh_matrix_grad);
    El::Zero(ih_bias_grad);
    El::Zero(hh_bias_grad);
    send_weight_grads_to_optimizers();
  }

  // Configure input and output tensor descriptors
  // Note: Descriptor dims have already been set in forward prop
  auto& input_desc = l.m_input_cudnn_desc;
  auto& output_desc = l.m_output_cudnn_desc;
  auto& hidden_desc = l.m_hidden_cudnn_desc;
  std::vector<cudnnTensorDescriptor_t>
    input_desc_list(sequence_length, input_desc),
    output_desc_list(sequence_length, output_desc);

  // Reorder tensor dims
  // Note: cuDNN uses sequence_length x mini_batch_size x size
  LocalMat input_sequence_workspace, output_sequence_workspace;
  LocalMat input_sequence_grad_workspace, output_sequence_grad_workspace;
  input_sequence_workspace.SetSyncInfo(sync_info);
  output_sequence_workspace.SetSyncInfo(sync_info);
  input_sequence_grad_workspace.SetSyncInfo(sync_info);
  output_sequence_grad_workspace.SetSyncInfo(sync_info);
  input_sequence_workspace.Resize(mini_batch_size*input_size, sequence_length);
  output_sequence_workspace.Resize(mini_batch_size*hidden_size, sequence_length);
  input_sequence_grad_workspace.Resize(mini_batch_size*input_size, sequence_length);
  output_sequence_grad_workspace.Resize(mini_batch_size*hidden_size, sequence_length);
  constexpr size_t one{1};
  cuda::copy_tensor(
    stream,
    {mini_batch_size, sequence_length, input_size},
    input_sequence.LockedBuffer(),
    {sequence_length*input_size, input_size, one},
    input_sequence_workspace.Buffer(),
    {input_size, mini_batch_size*input_size, one});
  cuda::copy_tensor(
    stream,
    {mini_batch_size, sequence_length, hidden_size},
    output_sequence.LockedBuffer(),
    {sequence_length*hidden_size, hidden_size, one},
    output_sequence_workspace.Buffer(),
    {hidden_size, mini_batch_size*hidden_size, one});
  cuda::copy_tensor(
    stream,
    {mini_batch_size, sequence_length, hidden_size},
    output_sequence_grad.LockedBuffer(),
    {sequence_length*hidden_size, hidden_size, one},
    output_sequence_grad_workspace.Buffer(),
    {hidden_size, mini_batch_size*hidden_size, one});

  // Pack weights into workspace buffer
  auto packed_weights = pack_cudnn_rnn_weights(
    handle,
    l.m_rnn_cudnn_desc,
    sync_info,
    input_size,
    hidden_size,
    ih_matrix,
    hh_matrix,
    ih_bias,
    hh_bias);
  LocalMat weights_grad_workspace;
  weights_grad_workspace.SetSyncInfo(sync_info);
  El::Zeros(
    weights_grad_workspace,
    packed_weights.size() / sizeof(TensorDataType),
    1);

  // Allocate cuDNN workspace buffers
  size_t cudnn_workspace_size;
  CHECK_CUDNN(
    cudnnGetRNNWorkspaceSize(
      handle,
      l.m_rnn_cudnn_desc,
      sequence_length,
      input_desc_list.data(),
      &cudnn_workspace_size));
  ByteBuffer cudnn_workspace(cudnn_workspace_size, sync_info);

  // Launch cuDNN GRU backprop
  // cuda::Graph::begin_capture(stream);
  CHECK_CUDNN(
    cudnnRNNBackwardData(
      handle,
      l.m_rnn_cudnn_desc,
      sequence_length,
      output_desc_list.data(),
      output_sequence_workspace.LockedBuffer(),
      output_desc_list.data(),
      output_sequence_grad_workspace.LockedBuffer(),
      hidden_desc,  // dhyDesc
      nullptr,
      hidden_desc,  // dcyDesc
      nullptr,
      l.m_packed_weights_cudnn_desc,
      packed_weights.data(),
      hidden_desc,
      init_hidden.LockedBuffer(),
      hidden_desc,  // cxDesc
      nullptr,
      input_desc_list.data(),
      input_sequence_grad_workspace.Buffer(),
      hidden_desc,
      init_hidden_grad.Buffer(),
      hidden_desc,  // dcxDesc
      nullptr,
      cudnn_workspace.data(),
      cudnn_workspace.size(),
      l.m_cudnn_reserve_space.data(),
      l.m_cudnn_reserve_space.size()));
  CHECK_CUDNN(
    cudnnRNNBackwardWeights(
      handle,
      l.m_rnn_cudnn_desc,
      sequence_length,
      input_desc_list.data(),
      input_sequence_workspace.LockedBuffer(),
      hidden_desc,
      init_hidden.LockedBuffer(),
      output_desc_list.data(),
      output_sequence_workspace.LockedBuffer(),
      cudnn_workspace.data(),
      cudnn_workspace.size(),
      l.m_packed_weights_cudnn_desc,
      weights_grad_workspace.Buffer(),
      l.m_cudnn_reserve_space.data(),
      l.m_cudnn_reserve_space.size()));
  // auto graph = cuda::Graph::end_capture(stream);
  // l.m_graph_backward_prop.update(graph);
  // l.m_graph_backward_prop.launch(stream);

  // Send gradients to optimizers
  unpack_cudnn_rnn_weights(
    handle,
    l.m_rnn_cudnn_desc,
    sync_info,
    input_size,
    hidden_size,
    weights_grad_workspace.LockedBuffer(),
    ih_matrix_grad,
    hh_matrix_grad,
    ih_bias_grad,
    hh_bias_grad);
  send_weight_grads_to_optimizers();

  // Reorder input grad tensor dims
  // Note: cuDNN uses sequence_length x mini_batch_size x input_size
  cuda::copy_tensor(
    stream,
    {mini_batch_size, sequence_length, input_size},
    input_sequence_grad_workspace.LockedBuffer(),
    {input_size, mini_batch_size*input_size, one},
    input_sequence_grad.Buffer(),
    {sequence_length*input_size, input_size, one});

#endif // 0

}
#endif // LBANN_HAS_CUDNN

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
