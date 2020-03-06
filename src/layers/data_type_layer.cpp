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

#define LBANN_DATA_TYPE_LAYER_INSTANTIATE
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/io/input/generic_input_layer.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

template <typename TensorDataType>
data_type_layer<TensorDataType>::data_type_layer(const data_type_layer<TensorDataType>& other) :
  Layer(other),
  m_weights(other.m_weights) {

  // Deep matrix copies
  m_inputs.reserve(other.m_inputs.size());
  m_outputs.reserve(other.m_outputs.size());
  m_gradient_wrt_outputs.reserve(other.m_gradient_wrt_outputs.size());
  m_gradient_wrt_inputs.reserve(other.m_gradient_wrt_inputs.size());
  for (const auto& ptr : other.m_inputs) {
    m_inputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }

}

template <typename TensorDataType>
data_type_layer<TensorDataType>& data_type_layer<TensorDataType>::operator=(const data_type_layer<TensorDataType>& other) {
  Layer::operator=(other);

  // Shallow copies
  m_weights = other.m_weights;

  // Deep matrix copies
  m_inputs.clear();
  m_outputs.clear();
  m_gradient_wrt_outputs.clear();
  m_gradient_wrt_inputs.clear();
  m_inputs.reserve(other.m_inputs.size());
  m_outputs.reserve(other.m_outputs.size());
  m_gradient_wrt_outputs.reserve(other.m_gradient_wrt_outputs.size());
  m_gradient_wrt_inputs.reserve(other.m_gradient_wrt_inputs.size());
  for (const auto& ptr : other.m_inputs) {
    m_inputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }

  return *this;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::forward_prop() {
  const auto fp_start = get_time();

  // Setup tensors
  const auto& c = static_cast<sgd_execution_context&>(m_model->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  fp_setup_inputs(mini_batch_size);
  fp_setup_outputs(mini_batch_size);

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

#ifdef LBANN_HAS_DISTCONV
  fp_setup_distconv(mini_batch_size);
#endif // LBANN_HAS_DISTCONV

  // Apply layer's compute function
  const auto fp_compute_start = get_time();
  fp_compute();
  m_fp_compute_time += get_time() - fp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (early_terminate_last_iteration()) {
    dump_activations();
  }
#endif // LBANN_HAS_DISTCONV

  // Add this layer as a gradient source for weight optimizers
  for (auto&& w : get_data_type_weights()) {
    optimizer* opt = w->get_optimizer();
    if (opt != nullptr) { opt->add_gradient_source(this); }
  }

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  m_fp_time += get_time() - fp_start;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::back_prop() {
  const auto bp_start = get_time();

  // Setup tensors
  const auto& c = static_cast<sgd_execution_context&>(m_model->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  bp_setup_gradient_wrt_outputs(mini_batch_size);
  bp_setup_gradient_wrt_inputs(mini_batch_size);

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

#ifdef LBANN_HAS_DISTCONV
  bp_setup_distconv(mini_batch_size);
#endif // LBANN_HAS_DISTCONV

  // Backprop the compute function.
  const auto bp_compute_start = get_time();
  bp_compute();
  m_bp_compute_time += get_time() - bp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (early_terminate_last_iteration()) {
    dump_error_signals();
  }
#endif // LBANN_HAS_DISTCONV

  // Remove this layer as a gradient source for weight optimizers
  for (auto&& w : get_data_type_weights()) {
    auto&& opt = w->get_optimizer();
    if (opt != nullptr) { opt->remove_gradient_source(this); }
  }

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  m_bp_time += get_time() - bp_start;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::summarize_matrices(lbann_summary& summarizer, int step) {

  // Summarize activation matrices
  const int num_children = get_num_children();
  for (int i = 0; i < num_children; ++i) {
    AbsDistMatReadProxyType<El::Device::CPU> acts(*m_outputs[i]);
    std::string prefix = m_name + "/activations";
    if (num_children > 1) { prefix += std::to_string(i); }
    summarizer.reduce_mean(prefix + "/mean", acts.GetLocked(), step);
    summarizer.reduce_min(prefix + "/min", acts.GetLocked(), step);
    summarizer.reduce_max(prefix + "/max", acts.GetLocked(), step);
    summarizer.reduce_stdev(prefix + "/stdev", acts.GetLocked(), step);
    summarizer.reduce_2norm(prefix + "/2norm2", acts.GetLocked(), step);
  }

  // Summarize error signal matrices
  const int num_parents = get_num_parents();
  for (int i = 0; i < num_parents; ++i) {
    AbsDistMatReadProxyType<El::Device::CPU> error_signals(*m_gradient_wrt_inputs[i]);
    std::string prefix = m_name + "/error_signals";
    if (num_parents > 1) { prefix += std::to_string(i); }
    summarizer.reduce_mean(prefix + "/mean", error_signals.GetLocked(), step);
    summarizer.reduce_min(prefix + "/min", error_signals.GetLocked(), step);
    summarizer.reduce_max(prefix + "/max", error_signals.GetLocked(), step);
    summarizer.reduce_stdev(prefix + "/stdev", error_signals.GetLocked(), step);
    summarizer.reduce_2norm(prefix + "/2norm2", error_signals.GetLocked(), step);
  }

}

// ===================================================================
// Tensor access functions
// ===================================================================

// Accessing distributed matrices
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_prev_activations(int parent_index) const -> const AbsDistMatrixType& {
  if (parent_index < 0 || parent_index >= (int) m_inputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous activation matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_inputs.size() << " previous activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_inputs[parent_index];
}

template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_activations(int child_index) const -> const AbsDistMatrixType& {
  if (child_index < 0 || child_index >= (int) m_outputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid activation matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_outputs.size() << " activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_outputs[child_index];
}

template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_prev_error_signals(int child_index) const -> const AbsDistMatrixType& {
  if (child_index < 0 || child_index >= (int) m_gradient_wrt_outputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_gradient_wrt_outputs.size() << " previous error signal matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_gradient_wrt_outputs[child_index];
}

template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_error_signals(int parent_index) const -> const AbsDistMatrixType& {
  if (parent_index < 0 || parent_index >= (int) m_gradient_wrt_inputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_gradient_wrt_inputs.size() << " error signal matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_gradient_wrt_inputs[parent_index];
}

// Accessing non-const distributed matrices
// Note: Using idiom from Item 3, p. 23 in "Effective C++", 3rd ed.,
// by Scott Meyers.
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_activations(int child_index) -> AbsDistMatrixType& {
  return const_cast<AbsDistMatrixType&>(static_cast<const data_type_layer<TensorDataType>&>(*this).get_activations(child_index));
}

template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_error_signals(int parent_index) -> AbsDistMatrixType& {
  return const_cast<AbsDistMatrixType&>(static_cast<const data_type_layer<TensorDataType>&>(*this).get_error_signals(parent_index));
}

// Accessing local matrices
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_activations(int child_index) -> AbsMatrixType& {
  return get_activations(child_index).Matrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_error_signals(int parent_index) -> AbsMatrixType& {
  return get_error_signals(parent_index).Matrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_prev_activations(int parent_index) const -> const AbsMatrixType&{
  return get_prev_activations(parent_index).LockedMatrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_activations(int child_index) const -> const AbsMatrixType& {
  return get_activations(child_index).LockedMatrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_prev_error_signals(int child_index) const -> const AbsMatrixType& {
  return get_prev_error_signals(child_index).LockedMatrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_error_signals(int parent_index) const -> const AbsMatrixType& {
  return get_error_signals(parent_index).LockedMatrix();
}

// Accessing matrices corresponding to parent/child layer
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_activations(const Layer& child) const -> const BaseDistMat& {
  const int child_index = (std::find(m_child_layers.begin(),
                                     m_child_layers.end(),
                                     &child)
                           - m_child_layers.begin());
  if (child_index >= get_num_children()) {
    std::stringstream err;
    err << "attempted to get activation tensor of "
        << "layer \"" << get_name() << "\" "
        << "corresponding to layer\"" << child.get_name() << "\", "
        << "which is not a child layer";
    LBANN_ERROR(err.str());
  }
  return get_activations(child_index);
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_error_signals(const Layer& parent) const -> const BaseDistMat& {
  const int parent_index = (std::find(m_parent_layers.begin(),
                                      m_parent_layers.end(),
                                      &parent)
                           - m_parent_layers.begin());
  if (parent_index >= get_num_parents()) {
    std::stringstream err;
    err << "attempted to get error signal tensor of "
        << "layer \"" << get_name() << "\" "
        << "corresponding to layer\"" << parent.get_name() << "\", "
        << "which is not a parent layer";
    LBANN_ERROR(err.str());
  }
  return get_error_signals(parent_index);
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_matrices(const El::Grid& grid) {

  // Destroy previously setup matrices
  m_inputs.clear();
  m_outputs.clear();
  m_gradient_wrt_outputs.clear();
  m_gradient_wrt_inputs.clear();

  // Construct matrices
  m_inputs.resize(get_num_parents());
  m_outputs.resize(get_num_children());
  m_gradient_wrt_outputs.resize(get_num_children());
  m_gradient_wrt_inputs.resize(get_num_parents());
  for (int i = 0; i < get_num_parents(); ++i) {
    m_inputs[i] = construct_matrix(grid, "input", i);
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_outputs[i] = construct_matrix(grid, "output", i);
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_gradient_wrt_outputs[i]
      = construct_matrix(grid, "gradient_wrt_output", i);
  }
  for (int i = 0; i < get_num_parents(); ++i) {
    m_gradient_wrt_inputs[i]
      = construct_matrix(grid, "gradient_wrt_input", i);
  }
}

template <typename TensorDataType>
auto data_type_layer<TensorDataType>::construct_matrix(const El::Grid& grid,
                                                       std::string type,
                                                       El::Int index) -> std::unique_ptr<AbsDistMatrixType> {

  // Choose matrix distribution
  El::Distribution col_dist, row_dist;
  El::DistWrap wrap;
  El::Device device = this->get_device_allocation();
  switch (get_data_layout()) {
  case data_layout::DATA_PARALLEL:
    col_dist = El::STAR;
    row_dist = El::VC;
    wrap     = El::ELEMENT;
    break;
  case data_layout::MODEL_PARALLEL:
    col_dist = El::MC;
    row_dist = El::MR;
    wrap     = El::ELEMENT;
    break;
  default: LBANN_ERROR("invalid data layout");
  }

  // Construct matrix
  std::unique_ptr<AbsDistMatrixType> mat;
  mat.reset(AbsDistMatrixType::Instantiate(grid, 0,
                                    col_dist, row_dist, wrap, device));

#ifdef LBANN_HAS_GPU
  // Allocate GPU memory with the CUDA API
  if (device == El::Device::GPU) { mat->Matrix().SetMemoryMode(0); }
  // Use pinned memory for data on the host.
  if (device == El::Device::CPU) { mat->Matrix().SetMemoryMode(1); }
#endif // LBANN_HAS_GPU

  return mat;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_data() {
  Layer::setup_data();

  // Get mini-batch size
  const auto& mini_batch_size = m_model->get_max_mini_batch_size();

  // Initialize input and output tensors
  fp_setup_inputs(mini_batch_size);
  fp_setup_outputs(mini_batch_size);

  // Initialize gradient w.r.t. output tensors
  // Note: We guess whether the tensor is a view or needs to allocate
  // memory, but there are some edge cases that are not handled.
  for (int i = 0; i < get_num_children(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled() && !keep_original_output(i)) {
      // Avoids allocating unused matrices
      continue;
    }
#endif // LBANN_HAS_DISTCONV
    const auto& child = *m_child_layers[i];
    const auto& output = get_activations(i);
    auto& gradient_wrt_output = *m_gradient_wrt_outputs[i];
    gradient_wrt_output.Empty(false);
    gradient_wrt_output.AlignWith(output);
    if (child.get_data_layout() == get_data_layout()
        && child.get_device_allocation() == get_device_allocation()
        && gradient_wrt_output.DistData() == output.DistData()) {
      El::LockedView(gradient_wrt_output, output);
    } else {
      El::Copy(output, gradient_wrt_output);
    }
  }

  // Initialize gradient w.r.t. input tensors
  bp_setup_gradient_wrt_inputs(mini_batch_size);

}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::bp_compute() {
  for (int i = 0; i < get_num_parents(); ++i) {
    El::Zero(get_error_signals(i));
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::check_setup() {
  Layer::check_setup();
  std::stringstream err;

  // Check number of tensors
  const int num_parents = get_num_parents();
  const int num_children = get_num_children();
  if ((int) m_inputs.size() != num_parents
      || (int) m_outputs.size() != num_children
      || (int) m_gradient_wrt_outputs.size() != num_children
      || (int) m_gradient_wrt_inputs.size() != num_parents) {
    err << "layer \"" << get_name() << "\" has an "
        << "invalid number of input and output tensors "
        << "(found " << num_parents << " parent layers, "
        << num_children << " child layers, "
        << m_inputs.size() << " input tensors, "
        << m_outputs.size() << " output tensors, "
        << m_gradient_wrt_outputs.size() << " gradient w.r.t. output tensors, "
        << m_gradient_wrt_inputs.size() << " gradient w.r.t. input tensors)";
    LBANN_ERROR(err.str());
  }

  // Check that tensors are initialized
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_inputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized input tensor (index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_outputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized output tensor (index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_gradient_wrt_outputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized gradient w.r.t. output tensor "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_gradient_wrt_inputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized gradient w.r.t. input tensor "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
}

// ===========================================================
// Weights access functions
// ===========================================================

template <typename TensorDataType>
void data_type_layer<TensorDataType>::replace_weights(Layer* other_layer) {
  if (other_layer == nullptr) {
    LBANN_ERROR("attempted to add null pointer as a replacement layer");
  }

  const std::vector<WeightsType*>& other_layer_weights =
    dynamic_cast<data_type_layer<TensorDataType>*>(other_layer)->get_data_type_weights();
  for (size_t i = 0; i < m_weights.size(); ++i) {
    m_weights[i]->set_values(other_layer_weights[i]->get_values());
  }

}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::fp_setup_inputs(El::Int mini_batch_size) {
  if (get_num_parents() < 1) { return; }

  // Determine distributed matrix alignment
  const auto& alignment_dist = m_parent_layers.front()->get_activations(*this).DistData();

  // Iterate through input tensors
  for (int i = 0; i < get_num_parents(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled() && !keep_original_input(i)) continue;
#endif // LBANN_HAS_DISTCONV
    // Initialize input tensor
    const auto& parent = *m_parent_layers[i];
    const auto& parent_output = parent.get_activations(*this);
    auto& input = *m_inputs[i];
    input.Empty(false);
    input.AlignWith(alignment_dist);
    if (parent_output.DistData() == input.DistData()) {
      El::LockedView(input, dynamic_cast<const AbsDistMatrixType&>(parent_output));
    } else {
      bool async_copy = false;
#if defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
      // Asynchronously copy CPU data to GPU data if they are otherwise aligned
      if (parent_output.GetLocalDevice() == El::Device::CPU
          && input.GetLocalDevice() == El::Device::GPU) {
        auto parent_dist_data = parent_output.DistData();
        parent_dist_data.device = El::Device::GPU;
        async_copy = parent_dist_data == input.DistData();
      }
#endif // defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
      if (async_copy) {
        El::CopyAsync(parent_output, input);
      } else {
        El::Copy(parent_output, input);
      }
    }

    // Check input matrix dimensions
    const auto& height = get_input_size(i);
    const auto& width = mini_batch_size;
    if (input.Height() != height || input.Width() != width) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" "
          << "expected an input tensor stored in a "
          << height << " x " << width << " matrix "
          << "from layer \"" << parent.get_name() << "\", but got a "
          << input.Height() << " x " << input.Width() << " matrix";
      LBANN_ERROR(err.str());
    }

  }

}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::fp_setup_outputs(El::Int mini_batch_size) {
  if (get_num_children() < 1) { return; }

  // Determine distributed matrix alignment
  const bool align_outputs = get_num_parents() > 0;
  const auto& alignment_dist = (align_outputs ?
                                get_prev_activations().DistData() :
                                get_activations().DistData());

  // Initialize output tensors
  for (int i = 0; i < get_num_children(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled() && !keep_original_output(i)) continue;
#endif // LBANN_HAS_DISTCONV
    auto& output = get_activations(i);
    output.Empty(false);
    if (align_outputs) { output.AlignWith(alignment_dist); }
    output.Resize(get_output_size(i), mini_batch_size);
  }

}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::bp_setup_gradient_wrt_outputs(El::Int mini_batch_size) {
  for (int i = 0; i < get_num_children(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled() && !keep_original_output(i)) continue;
#endif // LBANN_HAS_DISTCONV
    // Initialize gradient w.r.t. output tensor
    const auto& child = *m_child_layers[i];
    const auto& child_gradient_wrt_input = child.get_error_signals(*this);
    auto& gradient_wrt_output = *m_gradient_wrt_outputs[i];
    gradient_wrt_output.Empty(false);
    gradient_wrt_output.AlignWith(get_activations(i));
    if (child_gradient_wrt_input.DistData()
        == gradient_wrt_output.DistData()) {
      El::LockedView(gradient_wrt_output, dynamic_cast<const AbsDistMatrixType&>(child_gradient_wrt_input));
    } else {
      bool async_copy = false;
#if defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
      // Asynchronously copy CPU data to GPU data if they are otherwise aligned
      if (child_gradient_wrt_input.GetLocalDevice() == El::Device::CPU
          && gradient_wrt_output.GetLocalDevice() == El::Device::GPU) {
        auto child_dist_data = child_gradient_wrt_input.DistData();
        child_dist_data.device = El::Device::GPU;
        async_copy = child_dist_data == gradient_wrt_output.DistData();
      }
#endif // defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
      if (async_copy) {
        El::CopyAsync(child_gradient_wrt_input, gradient_wrt_output);
      } else {
        El::Copy(child_gradient_wrt_input, gradient_wrt_output);
      }
    }

    // Check gradient w.r.t. output matrix dimensions
    const auto& height = get_output_size(i);
    const auto& width = mini_batch_size;
    if (gradient_wrt_output.Height() != height
        || gradient_wrt_output.Width() != width) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" "
          << "expected a gradient w.r.t. output tensor stored in a "
          << height << " x " << width << " matrix "
          << "from layer \"" << child.get_name() << "\", but got a "
          << gradient_wrt_output.Height() << " x "
          << gradient_wrt_output.Width() << " matrix";
      LBANN_ERROR(err.str());
    }

  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) {
#ifdef LBANN_HAS_DISTCONV
  if (skip_first_layer_bp()) {
    return;
  }
#endif

  for (int i = 0; i < get_num_parents(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled() && !keep_original_input(i)) continue;
#endif // LBANN_HAS_DISTCONV
    auto& gradient_wrt_input = get_error_signals(i);
    gradient_wrt_input.Empty(false);
    gradient_wrt_input.AlignWith(get_prev_activations(i));
    gradient_wrt_input.Resize(get_input_size(i), mini_batch_size);
  }
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType>
bool data_type_layer<TensorDataType>::using_distconv() const {
  // Distconv is disabled if no parallel strategy is defined. When no
  // strategy is defined, the layer has the default strategy of all
  // zeros, which is invalid, thus should not be used when distconv is
  // used.
  const auto &ps = get_parallel_strategy();
  ParallelStrategy default_zero_ps;
  if (ps == default_zero_ps) {
    dc::MPIRootPrintStreamDebug()
        << "Disable " << get_name()
        << " as it does not have a parallel strategy.";
    return false;
  }

  // When DISTCONV_ENABLE is defined, all layers included in the
  // variable are enabled.
  char *env = getenv("DISTCONV_ENABLE");
  if (env) {
    std::string s(env);
    auto layer_names = dc::util::split(s, ',');
    for (const auto &name: layer_names) {
      if (get_name() == name) {
        return true;
      }
    }
    dc::MPIRootPrintStreamInfo()
        << "Disable " << get_name()
        << " as its name is not found in DISTCONV_ENABLE";
    return false;
  }

  // It is also disabled when the layer name is included in
  // an environment variable
  env = std::getenv("DISTCONV_DISABLE");
  if (env) {
    std::string s(env);
    auto layer_names = dc::util::split(s, ',');
    for (const auto &name: layer_names) {
      if (get_name() == name) {
        dc::MPIRootPrintStreamInfo()
            << "Disable " << get_name()
            << " as its name found in DISTCONV_DISABLE";
        return false;
      }
    }
  }
  return true;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_distconv() {
  setup_early_termination();
  setup_inter_layer_adaptation();
  setup_keep_original_tensors();
  setup_data();
  if (using_gpus()) { setup_gpu(); }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::init_distribution(
    std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
    std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
    std::set<dc::Dist*> &updated,
    std::set<dc::Dist*> &fixed) {
  if (!distconv_enabled()) return;
  const int num_dims = get_num_dims();
  auto &ps = get_parallel_strategy();
  dc::MPIRootPrintStreamDebug() << "Parallel Strategy for layer "
                                << get_name() << ": " << ps;
  int n = ps.sample_groups;
  int c = ps.channel_groups;
  int f = ps.filter_groups;
  int d = get_num_spatial_dims() == 3 ? ps.depth_groups : 1;
  int h = ps.height_groups;
  int w = ps.width_groups;
  int np = m_comm->get_procs_per_trainer();

  const int spatial_prod = d * h * w;

  // if only one process is used, do not parallelize
  if (np == 1) {
    n = c = f = h = w = d = 1;
  }

  if (c != f) {
    LBANN_ERROR("The numbers of channel and filter decomposition should be the same.");
  }
  if (c != 1 || f != 1) {
    LBANN_ERROR("Distconv does not support channel/filter parallelization yet. Layer: ",
                get_name(), ", ps: ", ps);
  }
  if (n * c * spatial_prod > np) {
    LBANN_ERROR("The number of MPI ranks must be at least as large as the number of processes implied by parallel strategy: ", ps);
  }
  // Put the remaining factor into the outer-most process dimension
  float rem = np / (float) (n * c * spatial_prod);
  n *= rem;
  ps.sample_splits *= rem;
  if (n * c * spatial_prod != np) {
    LBANN_ERROR("Can't determine factorization of the number of MPI ranks for parallel strategy: ",
                ps);
  }
  std::string xd_array, xd_array_names;
  if (num_dims == 5) {
    xd_array = dc::util::join_xd_array(std::vector<int>({n, c, d, h, w}));
    xd_array_names = "NxCxDxHxW";
  } else {
    assert_eq(num_dims, 4);
    xd_array = dc::util::join_xd_array(std::vector<int>({n, c, h, w}));
    xd_array_names = "NxCxHxW";
  }
  dc::MPIRootPrintStreamDebug() << "Process grid of " << xd_array_names << ": "
                                << xd_array;

  assert_always(spatial_prod * n * c == np && spatial_prod * n * f == np);

  ps.sample_groups = n;
  ps.channel_groups = c;
  ps.filter_groups = f;
  ps.depth_groups = d;
  ps.height_groups = h;
  ps.width_groups = w;
  // If splits are not set, set them to be equal to the group numbers
  if (ps.sample_splits == 0) ps.sample_splits = n;
  if (ps.channel_splits == 0) ps.channel_splits = c;
  if (ps.filter_splits == 0) ps.filter_splits = f;
  if (ps.depth_splits == 0) ps.depth_splits = d;
  if (ps.height_splits == 0) ps.height_splits = h;
  if (ps.width_splits == 0) ps.width_splits = w;

  dc::Shape input_locale_shape(num_dims);
  dc::Shape input_split_shape(num_dims);
  dc::Shape output_locale_shape(num_dims);
  dc::Shape output_split_shape(num_dims);

  input_locale_shape[dc::get_sample_dim()] = n;
  input_locale_shape[dc::get_channel_dim()] = c;
  input_locale_shape[0] = w;
  input_locale_shape[1] = h;
  if (num_dims == 5)  input_locale_shape[2] = d;

  input_split_shape[dc::get_sample_dim()] = ps.sample_splits;
  input_split_shape[dc::get_channel_dim()] = ps.channel_splits;
  input_split_shape[0] = ps.width_splits;
  input_split_shape[1] = ps.height_splits;
  if (num_dims == 5)  input_split_shape[2] = ps.depth_splits;

  output_locale_shape[dc::get_sample_dim()] = n;
  output_locale_shape[dc::get_channel_dim()] = f;
  output_locale_shape[0] = w;
  output_locale_shape[1] = h;
  if (num_dims == 5)  output_locale_shape[2] = d;

  output_split_shape[dc::get_sample_dim()] = ps.sample_splits;
  output_split_shape[dc::get_channel_dim()] = ps.filter_splits;
  output_split_shape[0] = ps.width_splits;
  output_split_shape[1] = ps.height_splits;
  if (num_dims == 5)  output_split_shape[2] = ps.depth_splits;

  auto prev_activations_dist =  dc::Dist::make_shared_distribution(
      input_locale_shape, input_split_shape);
  auto activations_dist = dc::Dist::make_shared_distribution(
      output_locale_shape, output_split_shape);
  auto prev_error_signals_dist = activations_dist;
  auto error_signals_dist = prev_activations_dist;
  std::array<dc::Dist, dc::num_dists> layer_dists = {prev_activations_dist,
                                                     activations_dist,
                                                     error_signals_dist,
                                                     prev_error_signals_dist};
  dists.insert(std::make_pair(this, layer_dists));
  invariants.insert(std::make_pair(&dists[this][0], std::set<dc::Dist*>()));
  invariants.insert(std::make_pair(&dists[this][1], std::set<dc::Dist*>()));
  invariants.insert(std::make_pair(&dists[this][2], std::set<dc::Dist*>()));
  invariants.insert(std::make_pair(&dists[this][3], std::set<dc::Dist*>()));
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_tensor_distribution_add_adjacent_invariants(
    std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
    std::map<dc::Dist*, std::set<dc::Dist*>> &invariants) {
  if (!distconv_enabled()) return;
  auto &layer_dists = dists[this];
  const auto &ps = get_parallel_strategy();

  // TEMPORARY HACK. Each tensor should be able to have its own
  // distribution, however, the current design only allows for a
  // single distribution for all output tensors in each layer,
  // meaning the data and label tensors need to have the same
  // distribution. The data tensor is likely to have halo as the
  // next layer will be convolution, whereas the label won't need to
  // have halo. For now, ignore the child layer for the label data.

  if (dynamic_cast<const generic_input_layer<TensorDataType>*>(this)) {
    const auto &child =  get_child_layers()[0];
    if (child->distconv_enabled() &&
        child->get_parallel_strategy() == ps) {
      invariants[&layer_dists[1]].insert(
          &dists[child][0]);
      invariants[&layer_dists[3]].insert(
          &dists[child][2]);
    }
  } else {
    for (auto &child: get_child_layers()) {
      if (child->distconv_enabled() &&
          child->get_parallel_strategy() == ps) {
        invariants[&layer_dists[1]].insert(
            &dists[child][0]);
        invariants[&layer_dists[3]].insert(
            &dists[child][2]);
      }
    }
  }
  for (auto &parent: get_parent_layers()) {
    if (dynamic_cast<const generic_input_layer<TensorDataType>*>(parent)) {
      const int child_index = std::find(
          parent->get_child_layers().begin(),
          parent->get_child_layers().end(),
          this) - parent->get_child_layers().begin();
      if (child_index == 1) continue;
      assert_eq(child_index, 0);
    }
    if (parent->distconv_enabled() &&
        parent->get_parallel_strategy() == ps) {
      invariants[&layer_dists[0]].insert(
          &dists[parent][1]);
      invariants[&layer_dists[2]].insert(
          &dists[parent][3]);
    }
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_inter_layer_adaptation() {
  if (!distconv_enabled()) return;

  const auto &ps = get_parallel_strategy();
  for (const auto &p: get_parent_layers()) {
    m_parent_copy_in_required.push_back(!p->distconv_enabled());
    m_parent_shuffle_required.push_back(
        (!p->distconv_enabled()) ||
        (ps != p->get_parallel_strategy()));
  }

  for (const auto &c: get_child_layers()) {
    m_child_copy_out_required.push_back(!c->distconv_enabled());
    m_child_shuffle_required.push_back(
        (!c->distconv_enabled()) ||
        (ps != c->get_parallel_strategy()));
  }

  std::stringstream ss;
  std::stringstream parent_copyin_ss;
  std::stringstream parent_shuffle_ss;
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_parent_copy_in_required[i]) {
      parent_copyin_ss << " " << i;
    }
    if (m_parent_shuffle_required[i]) {
      parent_shuffle_ss << " " << i;
    }
  }
  std::stringstream child_copyout_ss;
  std::stringstream child_shuffle_ss;
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_child_copy_out_required[i]) {
      child_copyout_ss << " " << i;
    }
    if (m_child_shuffle_required[i]) {
      child_shuffle_ss << " " << i;
    }
  }
  if (!parent_copyin_ss.str().empty()) {
    ss << " parent copyin:" << parent_copyin_ss.str() << ",";
  }
  if (!parent_shuffle_ss.str().empty()) {
    ss << " parent shuffle:" << parent_shuffle_ss.str() << ",";
  }
  if (!child_copyout_ss.str().empty()) {
    ss << " child copyout:" << child_copyout_ss.str() << ",";
  }
  if (!child_shuffle_ss.str().empty()) {
    ss << " child shuffle:" << child_shuffle_ss.str();
  }
  if (ss.str().size() > 0) {
    dc::MPIRootPrintStreamInfo() << get_name() << ":" << ss.str();
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_keep_original_tensors() {
  if (!distconv_enabled()) return;
  bool env_set = std::getenv("DISTCONV_KEEP_ORIGINAL_TENSORS") != nullptr;
  for (auto b: m_parent_copy_in_required) {
    m_keep_original_input.push_back(env_set || b);
  }
  for (auto b: m_child_copy_out_required) {
    m_keep_original_output.push_back(env_set || b);
  }
  return;
}

template <typename TensorDataType>
int data_type_layer<TensorDataType>::get_num_dims() const {
  auto nd = get_input_dims().size() + 1;
  if (!(nd == 4 || nd == 5)) {
    LBANN_ERROR(get_name(), ": Invalid number of dimensions: ", nd);
  }
  return nd;
}

template <typename TensorDataType>
int data_type_layer<TensorDataType>::get_num_spatial_dims() const {
  return get_num_dims() - 2;
}

template <typename TensorDataType>
const typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::
get_prev_activations_t() const {
  return m_prev_activations_t;
}

template <typename TensorDataType>
typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_prev_activations_t() {
  return m_prev_activations_t;
}

template <typename TensorDataType>
const typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_prev_activations_const_view() const {
  return m_prev_activations_const_view;
}

template <typename TensorDataType>
const typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_activations_t() const {
  return m_activations_t;
}

template <typename TensorDataType>
const typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_activations_t(
    const Layer &child) const {
  assert_always(get_num_children() == 1);
  return m_activations_t;
}

template <typename TensorDataType>
typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_activations_t() {
  return m_activations_t;
}

template <typename TensorDataType>
typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_activations_copyout() {
  return m_activations_copyout;
}

template <typename TensorDataType>
const typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::
get_prev_error_signals_t() const {
  return m_prev_error_signals_t;
}

template <typename TensorDataType>
typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_prev_error_signals_t() {
  return m_prev_error_signals_t;
}

template <typename TensorDataType>
const typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::
get_prev_error_signals_const_view() const {
  return m_prev_error_signals_const_view;
}

template <typename TensorDataType>
const typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_error_signals_t() const {
  return m_error_signals_t;
}

template <typename TensorDataType>
const typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_error_signals_t(
    const Layer &parent) const {
  assert_always(get_num_parents() == 1);
  return m_error_signals_t;
}

template <typename TensorDataType>
typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_error_signals_t() {
  return m_error_signals_t;
}

template <typename TensorDataType>
typename data_type_layer<TensorDataType>::TensorDevType &data_type_layer<TensorDataType>::get_error_signals_copyout() {
  return m_error_signals_copyout;
}

template <typename TensorDataType>
const dc::Shape data_type_layer<TensorDataType>::get_input_tensor_shape() const {
  const auto input_dims = get_input_dims();
  std::vector<int> input_tensor_shape_v(input_dims.rbegin(), input_dims.rend());
  input_tensor_shape_v.push_back(this->m_model->get_max_mini_batch_size());
  return dc::Shape(input_tensor_shape_v);
}

template <typename TensorDataType>
const dc::Shape data_type_layer<TensorDataType>::get_output_tensor_shape(
    int output_index) const {
  const auto output_dims = get_output_dims(output_index);
  std::vector<int> output_tensor_shape_v(output_dims.rbegin(), output_dims.rend());
  output_tensor_shape_v.push_back(this->m_model->get_max_mini_batch_size());
  return dc::Shape(output_tensor_shape_v);
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_prev_activations_tensor(
    const std::array<dc::Dist, dc::num_dists> &dists) {
  const auto input_tensor_shape = get_input_tensor_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(get_num_dims());
  auto input_local_shape = input_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  input_local_shape[-1] = 0;

  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_parent_copy_in_required[i] || m_parent_shuffle_required[i]) {
      if (i != 0) LBANN_ERROR("Copyin of non-first tensor not supported yet");
      if (m_parent_copy_in_required[i]) {
        m_prev_activations_const_view = TensorDevType(input_tensor_shape, loc,
                                                      sample_dist,
                                                      input_local_shape);
      } else {
        m_prev_activations_const_view = dynamic_cast<
          const data_type_layer<TensorDataType>*>(get_parent_layers()[i])->get_activations_t(*this);
      }
      m_prev_activations_t = TensorDevType(input_tensor_shape, loc, dists[0]);
      assert0(m_prev_activations_t.allocate());
      m_prev_activations_t.zero(dc::get_stream());
      m_prev_activations_shuffler = dc::get_tensor_shuffler(
          m_prev_activations_const_view, m_prev_activations_t);
      for (int mode = 0; mode < 3; ++mode) {
        m_prev_activations_shuffler_last_mb[mode] = nullptr;
      }
    } else {
      // TODO: Only setup the first input. Needs to make
      // m_activations_t a vector as well.
      if (i != 0) continue;
      m_prev_activations_t = dynamic_cast<const data_type_layer<TensorDataType>*>(get_parent_layers()[i])->get_activations_t(*this);
      assert_always(m_prev_activations_t.get_distribution() == dists[0]);
    }
  }
  dc::MPIPrintStreamDebug() << get_name() << "; "
                            << "prev activations: " << m_prev_activations_t;
}

template <typename TensorDataType>
dc::Shape data_type_layer<TensorDataType>::get_activations_tensor_local_shape() const {
  return m_prev_activations_t.get_local_shape();
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_activations_tensor(
    const std::array<dc::Dist, dc::num_dists> &dists, bool allocate) {
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const dc::Shape output_tensor_shape = get_output_tensor_shape();
  const auto activations_local_shape =
      get_activations_tensor_local_shape();
  m_activations_t = TensorDevType(output_tensor_shape,
                                  loc, dists[1], activations_local_shape);
  if (allocate) {
    assert0(m_activations_t.allocate());
    m_activations_t.zero(dc::get_stream());
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_activations_copyout_tensor(
    const std::array<dc::Dist, dc::num_dists> &dists) {
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(get_num_dims());
  const auto output_tensor_shape = m_activations_t.get_shape();
  assert_always(!output_tensor_shape.is_empty());
  auto output_local_shape = output_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  output_local_shape[-1] = 0;
  m_activations_copyout = TensorDevType(output_tensor_shape, loc, sample_dist,
                                        output_local_shape);
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_child_copy_out_required[i]) {
      if (i != 0) {
        LBANN_ERROR("Copyout of non-first tensor not supported yet");
      }
      m_activations_shuffler = dc::get_tensor_shuffler(
          m_activations_t, m_activations_copyout);
      for (int mode = 0; mode < 3; ++mode) {
        m_activations_shuffler_last_mb[mode] = nullptr;
      }
    }
  }
  dc::MPIPrintStreamDebug() << get_name() << "; "
                            << "activations: " << m_activations_t;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_prev_error_signals_tensor(
    const std::array<dc::Dist, dc::num_dists> &dists) {
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(get_num_dims());
  const auto output_tensor_shape = m_activations_t.get_shape();
  assert_always(!output_tensor_shape.is_empty());
  auto output_local_shape = output_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  output_local_shape[-1] = 0;

  for (int i = 0; i < get_num_children(); ++i) {
    if (m_child_copy_out_required[i] || m_child_shuffle_required[i]) {
      if (i != 0) LBANN_ERROR("Copyout of non-first tensor not supported yet");
      if (m_child_copy_out_required[i]) {
        m_prev_error_signals_const_view = TensorDevType(output_tensor_shape, loc,
                                                        sample_dist,
                                                        output_local_shape);
      } else {
        m_prev_error_signals_const_view =
            dynamic_cast<const data_type_layer<TensorDataType>*>(
                get_child_layers()[i])->get_error_signals_t(*this);
      }
      m_prev_error_signals_t = TensorDevType(output_tensor_shape, loc,
                                             dists[3],
                                             m_activations_t.get_local_shape());
      assert0(m_prev_error_signals_t.allocate());
      m_prev_error_signals_t.zero(dc::get_stream());
      m_prev_error_signals_shuffler = dc::get_tensor_shuffler(
          m_prev_error_signals_const_view, m_prev_error_signals_t);
      for (int mode = 0; mode < 3; ++mode) {
        m_prev_error_signals_shuffler_last_mb[mode] = nullptr;
      }
    } else {
      // TODO: Only setup the first input. Needs to make
      // m_prev_error_signals_t a vector as well.
      if (i != 0) continue;
      m_prev_error_signals_t = dynamic_cast<
        const data_type_layer<TensorDataType>*>(
            get_child_layers()[i])->get_error_signals_t(*this);
      assert_always(m_prev_error_signals_t.get_distribution() ==
                    dists[3]);
    }
  }
  dc::MPIPrintStreamDebug() << get_name() << "; "
                            << "prev error signals: " << m_prev_error_signals_t;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_error_signals_tensor(
    const std::array<dc::Dist, dc::num_dists> &dists) {
  const auto input_tensor_shape = get_input_tensor_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  m_error_signals_t = TensorDevType(input_tensor_shape, loc,
                                    dists[2],
                                    m_prev_activations_t.get_local_shape());
  if (skip_first_layer_bp()) {
    dc::MPIPrintStreamDebug()
        << get_name() << ": skipping allocation of error signals";
  } else {
    assert0(m_error_signals_t.allocate());
    m_error_signals_t.zero(dc::get_stream());
  }
  dc::MPIPrintStreamDebug() << get_name() << "; "
                            << "error signals: " << m_error_signals_t;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_error_signals_copyout_tensor(
    const std::array<dc::Dist, dc::num_dists> &dists) {
  const auto input_tensor_shape = get_input_tensor_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(get_num_dims());
  auto input_local_shape = input_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  input_local_shape[-1] = 0;

  // TODO: Only the first error signal tensor is handled
  m_error_signals_copyout = TensorDevType(input_tensor_shape, loc, sample_dist,
                                          input_local_shape);
  if (m_parent_copy_in_required[0] && !skip_first_layer_bp()) {
    m_error_signals_shuffler = dc::get_tensor_shuffler(
        m_error_signals_t, m_error_signals_copyout);
    for (int i = 0; i < 3; ++i) {
      m_error_signals_shuffler_last_mb[i] = nullptr;
    }
  }
}

namespace {
template <typename TensorDataType>
dc::TensorShuffler<TensorDataType> *get_shuffler(Layer *layer,
                                                 dc::TensorShuffler<TensorDataType> *main_shuffler,
                                                 dc::TensorShuffler<TensorDataType> **last_mb_shufflers,
                                                 const dc::TensorDev<TensorDataType> &src,
                                                 const dc::TensorDev<TensorDataType> &dst) {
  dc::TensorShuffler<TensorDataType> *shuffler = nullptr;
  const auto& c = static_cast<sgd_execution_context&>(
      layer->get_model()->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  if (layer->get_model()->get_max_mini_batch_size() == mini_batch_size) {
    shuffler = main_shuffler;
  } else {
    // The last remaining mini-batches for the train, validation, and
    // testing modes
    auto mode = layer->get_model()->get_execution_context().get_execution_mode();
    auto shfl_idx = static_cast<int>(mode);
    assert_always(shfl_idx >= 0 && shfl_idx < 3);
    if (last_mb_shufflers[shfl_idx] == nullptr) {
      last_mb_shufflers[shfl_idx] = dc::get_tensor_shuffler(src, dst);
    }
    shuffler = last_mb_shufflers[shfl_idx];
  }
  return shuffler;
}
} // namespace

template <typename TensorDataType>
void data_type_layer<TensorDataType>::ensure_prev_activations() {
  for (int i = 0; i < get_num_parents(); ++i) {
    if (!(m_parent_copy_in_required[i] || m_parent_shuffle_required[i])) {
      continue;
    }
    if (i != 0) {
      LBANN_ERROR("Distconv assumes non-first tensors are available as distconv tensors");
    }
    if (m_parent_copy_in_required[i]) {
      dc::MPIPrintStreamDebug()
          << "Copying previous activations from sample decomposition";
      assert0(dc::tensor::View(
          m_prev_activations_const_view,
          get_prev_activations().LockedBuffer()));
    }
    auto *shuffler = get_shuffler(this, m_prev_activations_shuffler,
                                  m_prev_activations_shuffler_last_mb,
                                  m_prev_activations_const_view,
                                  m_prev_activations_t);
    assert_always(shuffler != nullptr);
    shuffler->shuffle_forward(
        m_prev_activations_const_view.get_const_base_ptr(),
        m_prev_activations_t.get_base_ptr(),
        El::GPUManager::Stream());
  }
  this->m_model->clock_start();
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::copy_out_activations() {
  for (int i = 0; i < get_num_children(); ++i) {
    if (!m_child_copy_out_required[i]) continue;

    if (i != 0) LBANN_ERROR("Copyout of non-first tensor not supported");

    this->m_model->clock_end();

    dc::MPIPrintStreamDebug()
        << "Copying activations back to sample decomposition";
    assert0(dc::tensor::View(
        m_activations_copyout, get_activations().Buffer()));
    auto *shuffler = get_shuffler(this, m_activations_shuffler,
                                  m_activations_shuffler_last_mb,
                                  m_activations_t, m_activations_copyout);
    assert_always(shuffler != nullptr);
    shuffler->shuffle_forward(
        m_activations_t.get_const_base_ptr(),
        m_activations_copyout.get_base_ptr(),
        El::GPUManager::Stream());
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::ensure_prev_error_signals() {
  for (int i = 0; i < get_num_children(); ++i) {
    if (!(m_child_copy_out_required[i] || m_child_shuffle_required[i])) {
      continue;
    }

    if (i != 0) LBANN_ERROR("Copyin non-first tensor not supported");

    if (m_child_copy_out_required[i]) {
      dc::MPIPrintStreamDebug()
          << "Copying previous error signals from sample decomposition";
      assert0(dc::tensor::View(
          m_prev_error_signals_const_view,
          get_prev_error_signals().LockedBuffer()));
    }
    auto *shuffler = get_shuffler(this, m_prev_error_signals_shuffler,
                                  m_prev_error_signals_shuffler_last_mb,
                                  m_prev_error_signals_const_view,
                                  m_prev_error_signals_t);
    assert_always(shuffler != nullptr);
    shuffler->shuffle_forward(
        m_prev_error_signals_const_view.get_const_base_ptr(),
        m_prev_error_signals_t.get_base_ptr(),
        El::GPUManager::Stream());
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::copy_out_error_signals() {
  if (skip_first_layer_bp()) {
    // No need to copy back when the parent is an input layer
    dc::MPIPrintStreamDebug()
        << "Skipping copy back as this layer is the first layer";
    return;
  }

  // No need to copy back as the original layer compute function
  // will be called
  if (get_exit_count() == 0) return;

  for (int i = 0; i < get_num_parents(); ++i) {
    if (!m_parent_copy_in_required[i]) continue;

    if (i != 0) LBANN_ERROR("Copyout non-first tensor not supported");

    dc::MPIPrintStreamDebug()
        << "Copying error signals back to sample decomposition";
    assert0(dc::tensor::View(
        m_error_signals_copyout, get_error_signals().Buffer()));
    auto *shuffler = get_shuffler(this, m_error_signals_shuffler,
                                  m_error_signals_shuffler_last_mb,
                                  m_error_signals_t, m_error_signals_copyout);
    assert_always(shuffler != nullptr);
    shuffler->shuffle_forward(
        m_error_signals_t.get_const_base_ptr(),
        m_error_signals_copyout.get_base_ptr(),
        El::GPUManager::Stream());
  }
}

template <typename TensorDataType>
bool data_type_layer<TensorDataType>::parent_copy_in_required(size_t input_index) const {
  if (input_index < m_parent_copy_in_required.size()) {
    return m_parent_copy_in_required.at(input_index);
  } else {
    LBANN_ERROR("Out of range error! parent_copy_in_required size: ",
                m_parent_copy_in_required.size(),
                ", index: ", input_index);
  }
}

template <typename TensorDataType>
bool data_type_layer<TensorDataType>::parent_shuffle_required(size_t input_index) const {
  if (input_index < m_parent_shuffle_required.size()) {
    return m_parent_shuffle_required.at(input_index);
  } else {
    LBANN_ERROR("Out of range error! parent_shuffle_required size: ",
                m_parent_shuffle_required.size(),
                ", index: ", input_index);
  }
}

template <typename TensorDataType>
bool data_type_layer<TensorDataType>::child_copy_out_required(size_t output_index) const {
  if (output_index < m_child_copy_out_required.size()) {
    return m_child_copy_out_required.at(output_index);
  } else {
    LBANN_ERROR("Out of range error! child_copy_out_required size: ",
                m_child_copy_out_required.size(),
                ", index: ", output_index);
  }
}

template <typename TensorDataType>
bool data_type_layer<TensorDataType>::child_shuffle_required(size_t output_index) const {
  if (output_index < m_child_shuffle_required.size()) {
    return m_child_shuffle_required.at(output_index);
  } else {
    LBANN_ERROR("Out of range error! child_shuffle_required size: ",
                m_child_shuffle_required.size(),
                ", index: ", output_index);
  }
}

template <typename TensorDataType>
bool data_type_layer<TensorDataType>::keep_original_input(size_t input_index) const {
  if (input_index < m_keep_original_input.size()) {
    return m_keep_original_input.at(input_index);
  } else {
    LBANN_ERROR("Out of range error! m_keep_original_input size: ",
                m_keep_original_input.size(),
                ", index: ", input_index);
  }
}

template <typename TensorDataType>
bool data_type_layer<TensorDataType>::keep_original_output(size_t output_index) const {
  if (output_index < m_keep_original_output.size()) {
    return m_keep_original_output.at(output_index);
  } else {
    LBANN_ERROR("Out of range error! m_keep_original_output size: ",
                m_keep_original_output.size(),
                ", index: ", output_index);
  }
}

template <typename TensorDataType>
bool data_type_layer<TensorDataType>::keep_original() const {
  for (int i = 0; i < get_num_parents(); ++i) {
    if (!keep_original_input(i)) return false;
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (!keep_original_output(i)) return false;
  }
  return true;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::fp_setup_distconv(El::Int mini_batch_size) {
  if (!distconv_enabled()) return;

  early_terminate();

  // Reconfigure the sample dimension as the mini batch size may vary
  // at the end of epoch
  m_prev_activations_t.set_outermost_dimension(mini_batch_size);
  assert_eq((int)m_prev_activations_t.get_shape()[-1],
            mini_batch_size);
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_parent_copy_in_required[i] || m_parent_shuffle_required[i]) {
      if (i != 0) {
        LBANN_ERROR("Copyin non-first tensor not supported");
      }
      m_prev_activations_const_view.set_outermost_dimension(
          mini_batch_size);
      assert_eq((int)m_prev_activations_const_view.get_shape()[-1],
                mini_batch_size);
      if (m_parent_copy_in_required[i]) {
        // then, parent is assumed to be data parallel, so the local
        // size of the sample dimension should be equal to
        // the local width of previous activations. The check only
        // matters for split root processes as the rest just hold
        // invalid copy of the root data.
        if (m_prev_activations_const_view.is_split_root()) {
          assert_eq(
              (int)m_prev_activations_const_view.get_local_shape()[-1],
              get_prev_activations().LocalWidth());
        }
      }
    }
  }
  m_activations_t.set_outermost_dimension(mini_batch_size);
  assert_eq((int)m_activations_t.get_shape()[-1],
            mini_batch_size);
  m_activations_copyout.set_outermost_dimension(mini_batch_size);
  assert_eq((int)m_activations_copyout.get_shape()[-1],
            mini_batch_size);
  // TODO: Needs to check other output tensors
  if (keep_original_output(0) && m_activations_copyout.is_split_root()) {
    assert_eq((int)m_activations_copyout.get_local_shape()[-1],
              get_activations().LocalWidth());
  }

  ensure_prev_activations();
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::bp_setup_distconv(El::Int mini_batch_size) {

  if (!distconv_enabled()) return;

  // Reconfigure the sample dimension as the mini batch size may vary
  // at the end of epoch
  m_prev_error_signals_t.set_outermost_dimension(mini_batch_size);
  assert_always((int)m_prev_error_signals_t.get_shape()[-1] ==
                mini_batch_size);
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_child_copy_out_required[i] || m_child_shuffle_required[i]) {
      if (i != 0) {
        LBANN_ERROR("Copyout non-first tensor not supported");
      }
      m_prev_error_signals_const_view.set_outermost_dimension(mini_batch_size);
      assert_eq((int)m_prev_error_signals_const_view.get_shape()[-1],
                mini_batch_size);
      if (m_child_copy_out_required[i] &&
          m_prev_error_signals_const_view.is_split_root()) {
        assert_eq(
            (int)m_prev_error_signals_const_view.get_local_shape()[-1],
            get_prev_error_signals().LocalWidth());
      }
    }
    m_error_signals_t.set_outermost_dimension(mini_batch_size);
    assert_eq((int)m_error_signals_t.get_shape()[-1],
              mini_batch_size);
    m_error_signals_copyout.set_outermost_dimension(mini_batch_size);
    assert_eq((int)m_error_signals_copyout.get_shape()[-1],
              mini_batch_size);
    // TODO: Check other input tensors
    if (i == 0) {
      if (keep_original_input(i) && !skip_first_layer_bp()
          && m_error_signals_copyout.is_split_root()) {
        assert_eq((int)m_error_signals_copyout.get_local_shape()[-1],
                  get_error_signals().LocalWidth());
      }
    }
  }
  ensure_prev_error_signals();
}

template <typename TensorDataType>
size_t data_type_layer<TensorDataType>::estimate_memory_usage(
    const std::array<dc::Dist, dc::num_dists> &dists) {
  if (!distconv_enabled()) {
    return 0;
  }
  auto max_mb = this->m_model->get_max_mini_batch_size();
  size_t usage = 0;
  // fp
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_parent_copy_in_required[i] || m_parent_shuffle_required[i]) {
      usage += get_input_size(i) * max_mb / dists[0].get_split_shape().size();
    }
  }
  usage += get_output_size() * max_mb / dists[1].get_split_shape().size();
  // bp
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_child_copy_out_required[i] || m_child_shuffle_required[i]) {
      usage += get_output_size(i) * max_mb / dists[3].get_split_shape().size();
    }
  }
  usage += get_input_size() * max_mb / dists[2].get_split_shape().size();
  return usage * sizeof(TensorDataType);
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::dump_activations() const {
  dc::dump_tensor(early_terminate_last_iteration(),
                  m_activations_t, get_name() + "_activations");
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::dump_reference_activations() {
  assert0(dc::tensor::View(
      m_activations_copyout, get_activations().LockedBuffer()));
  dc::dump_tensor(early_terminate_last_iteration(), m_activations_copyout,
                  get_name() + "_activations_original");
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::dump_error_signals() const {
  dc::dump_tensor(early_terminate_last_iteration(), m_error_signals_t,
                  get_name() + "_error_signals");
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::dump_reference_error_signals() {
  assert0(dc::tensor::View(
      m_error_signals_copyout, get_error_signals().LockedBuffer()));
  dc::dump_tensor(early_terminate_last_iteration(), m_error_signals_copyout,
                  get_name() + "_error_signals_original");
}

#endif // LBANN_HAS_DISTCONV

#define PROTO(T)                     \
  template class data_type_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
