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

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"

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
    m_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
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
    m_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
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

  // Apply layer's compute function
  const auto fp_compute_start = get_time();
  fp_compute();
  m_fp_compute_time += get_time() - fp_compute_start;

  // Add this layer as a gradient source for weight optimizers
  for (auto&& w : get_weights()) {
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

  // Backprop the compute function.
  const auto bp_compute_start = get_time();
  bp_compute();
  m_bp_compute_time += get_time() - bp_compute_start;

  // Remove this layer as a gradient source for weight optimizers
  for (auto&& w : get_weights()) {
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
    AbsDistMatReadProxy<El::Device::CPU> acts(*m_outputs[i]);
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
    AbsDistMatReadProxy<El::Device::CPU> error_signals(*m_gradient_wrt_inputs[i]);
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
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_prev_activations(int parent_index) const {
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
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_activations(int child_index) const {
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
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_prev_error_signals(int child_index) const {
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
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_error_signals(int parent_index) const {
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
El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_activations(int child_index) {
  return const_cast<El::AbstractDistMatrix<TensorDataType>&>(static_cast<const data_type_layer<TensorDataType>&>(*this).get_activations(child_index));
}

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_error_signals(int parent_index) {
  return const_cast<El::AbstractDistMatrix<TensorDataType>&>(static_cast<const data_type_layer<TensorDataType>&>(*this).get_error_signals(parent_index));
}

// Accessing local matrices
template <typename TensorDataType>
El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_activations(int child_index) {
  return get_activations(child_index).Matrix();
}
template <typename TensorDataType>
El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_error_signals(int parent_index) {
  return get_error_signals(parent_index).Matrix();
}
template <typename TensorDataType>
const El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_prev_activations(int parent_index) const {
  return get_prev_activations(parent_index).LockedMatrix();
}
template <typename TensorDataType>
const El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_activations(int child_index) const {
  return get_activations(child_index).LockedMatrix();
}
template <typename TensorDataType>
const El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_prev_error_signals(int child_index) const {
  return get_prev_error_signals(child_index).LockedMatrix();
}
template <typename TensorDataType>
const El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_error_signals(int parent_index) const {
  return get_error_signals(parent_index).LockedMatrix();
}

// Accessing matrices corresponding to parent/child layer
template <typename TensorDataType>
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_activations(const data_type_layer<TensorDataType>& child) const {
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
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_error_signals(const data_type_layer<TensorDataType>& parent) const {
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
std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> data_type_layer<TensorDataType>::construct_matrix(const El::Grid& grid,
                                                                              std::string type,
                                                                              El::Int index) {

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
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> mat;
  mat.reset(El::AbstractDistMatrix<TensorDataType>::Instantiate(grid, 0,
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

  const std::vector<data_type_weights<TensorDataType>*>& other_layer_weights =
    dynamic_cast<data_type_layer<TensorDataType>*>(other_layer)->get_data_type_weights();
  for (size_t i = 0; i < m_weights.size(); ++i) {
    m_weights[i]->set_values(other_layer_weights[i]->get_values());
  }

}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::fp_setup_inputs(El::Int mini_batch_size) {
  if (get_num_parents() < 1) { return; }

  // Determine distributed matrix alignment
  const auto& alignment_dist
    = dynamic_cast<const data_type_layer<TensorDataType>*>(m_parent_layers.front())->get_activations(*this).DistData();

  // Iterate through input tensors
  for (int i = 0; i < get_num_parents(); ++i) {

    // Initialize input tensor
    const auto& parent = dynamic_cast<const data_type_layer<TensorDataType>&>(*m_parent_layers[i]);
    const auto& parent_output = parent.get_activations(*this);
    auto& input = *m_inputs[i];
    input.Empty(false);
    input.AlignWith(alignment_dist);
    if (parent_output.DistData() == input.DistData()) {
      El::LockedView(input, parent_output);
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
    auto& output = get_activations(i);
    output.Empty(false);
    if (align_outputs) { output.AlignWith(alignment_dist); }
    output.Resize(get_output_size(i), mini_batch_size);
  }

}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::bp_setup_gradient_wrt_outputs(El::Int mini_batch_size) {
  for (int i = 0; i < get_num_children(); ++i) {

    // Initialize gradient w.r.t. output tensor
    const auto& child = dynamic_cast<const data_type_layer<TensorDataType>&>(*m_child_layers[i]);
    const auto& child_gradient_wrt_input = child.get_error_signals(*this);
    auto& gradient_wrt_output = *m_gradient_wrt_outputs[i];
    gradient_wrt_output.Empty(false);
    gradient_wrt_output.AlignWith(get_activations(i));
    if (child_gradient_wrt_input.DistData()
        == gradient_wrt_output.DistData()) {
      El::LockedView(gradient_wrt_output, child_gradient_wrt_input);
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
  for (int i = 0; i < get_num_parents(); ++i) {
    auto& gradient_wrt_input = get_error_signals(i);
    gradient_wrt_input.Empty(false);
    gradient_wrt_input.AlignWith(get_prev_activations(i));
    gradient_wrt_input.Resize(get_input_size(i), mini_batch_size);
  }
}

template class data_type_layer<float>;

} // namespace lbann
