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

#include "lbann/layers/layer.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"

#include <layers.pb.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

// Asynchronous memory transfers for input data
// Note: This introduces a race condition. It is possible for the
// input data to be modified by another layer before it is used by
// this layer.
// #define ASYNC_INPUT_MEMORY_TRANSFER
#include "lbann/utils/cuda.hpp"

namespace lbann {

Layer::Layer(lbann_comm *comm)
  : m_comm(comm),
    m_frozen(false) {

  // Initialize layer name
  static int num_layers = 0;
  m_name = "layer" + std::to_string(num_layers);
  num_layers++;

  // Reset timing counters
  reset_counters();

}

Layer::Layer(const Layer& other) :
  m_comm(other.m_comm),
  m_weights(other.m_weights),
  m_parent_layers(other.m_parent_layers),
  m_child_layers(other.m_child_layers),
  m_expected_num_parent_layers(other.m_expected_num_parent_layers),
  m_expected_num_child_layers(other.m_expected_num_child_layers),
  m_model(other.m_model),
  m_frozen(other.m_frozen),
  m_fp_time(other.m_fp_time),
  m_fp_compute_time(other.m_fp_compute_time),
  m_bp_time(other.m_bp_time),
  m_bp_compute_time(other.m_bp_compute_time),
  m_update_time(other.m_update_time),
  m_name(other.m_name),
  m_output_dims_list(other.m_output_dims_list),
  m_hint_layer(other.m_hint_layer) {

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

Layer& Layer::operator=(const Layer& other) {

  // Shallow copies
  m_comm = other.m_comm;
  m_weights = other.m_weights;
  m_parent_layers = other.m_parent_layers;
  m_child_layers = other.m_child_layers;
  m_expected_num_parent_layers = other.m_expected_num_parent_layers;
  m_expected_num_child_layers = other.m_expected_num_child_layers;
  m_model = other.m_model;
  m_frozen = other.m_frozen;
  m_fp_time = other.m_fp_time;
  m_fp_compute_time = other.m_fp_compute_time;
  m_bp_time = other.m_bp_time;
  m_bp_compute_time = other.m_bp_compute_time;
  m_update_time = other.m_update_time;
  m_name = other.m_name;
  m_output_dims_list = other.m_output_dims_list;
  m_hint_layer = other.m_hint_layer;

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

description Layer::get_description() const {

  // Construct description object
  std::stringstream ss;
  ss << get_name() << " (" << get_type() << ")";
  description desc(ss.str());

  // Input dimensions
  const auto& parents = get_parent_layers();
  if (!parents.empty()) {
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < parents.size(); ++i) {
      ss << (i > 0 ? ", " : "");
      const auto& dims = get_input_dims(i);
      for (size_t j = 0; j < dims.size(); ++j) {
        ss << (j == 0 ? "" : "x") << dims[j];
      }
      ss << " (from ";
      if (parents[i] == nullptr) {
        ss << "unknown layer";
      } else {
        ss << parents[i]->get_type() << " layer "
           << "\"" << parents[i]->get_name() << "\"";
      }
      ss << ")";
    }
    desc.add("Input dimensions", ss.str());
  }

  // Output dimensions
  const auto& children = get_child_layers();
  if (!children.empty()) {
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < children.size(); ++i) {
      ss << (i > 0 ? ", " : "");
      const auto& dims = get_output_dims(i);
      for (size_t j = 0; j < dims.size(); ++j) {
        ss << (j == 0 ? "" : "x") << dims[j];
      }
      ss << " (to ";
      if (children[i] == nullptr) {
        ss << "unknown layer";
      } else {
        ss << children[i]->get_type() << " layer "
           << "\"" << children[i]->get_name() << "\"";
      }
      ss << ")";
    }
    desc.add("Output dimensions", ss.str());
  }

  // Weights
  const auto& weights_list = get_weights();
  if (!weights_list.empty()) {
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < weights_list.size(); ++i) {
      ss << (i > 0 ? ", " : "");
      if (weights_list[i] == nullptr) {
        ss << "unknown weights";
      } else {
        const auto& dims = weights_list[i]->get_dims();
        ss << weights_list[i]->get_name() << " (";
        for (size_t j = 0; j < dims.size(); ++j) {
          ss << (j > 0 ? "x" : "") << dims[j];
        }
        ss << ")";
      }
    }
    desc.add("Weights", ss.str());
  }

  // Data layout
  ss.str(std::string{});
  ss.clear();
  switch (get_data_layout()) {
  case data_layout::DATA_PARALLEL:  ss << "data-parallel";  break;
  case data_layout::MODEL_PARALLEL: ss << "model-parallel"; break;
  case data_layout::invalid:
  default:
    ss << "invalid";
  }
  desc.add("Data layout", ss.str());

  // Device
  ss.str(std::string{});
  ss.clear();
  switch (get_device_allocation()) {
  case El::Device::CPU: ss << "CPU";     break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU: ss << "GPU";     break;
#endif // LBANN_HAS_GPU
  default:              ss << "unknown";
  }
  desc.add("Device", ss.str());

  // Freeze state
  if (is_frozen()) {
    desc.add("Frozen");
  }

  return desc;
}

void Layer::forward_prop() {
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
#endif

  // Apply layer's compute function
  const auto fp_compute_start = get_time();
  fp_compute();
  m_fp_compute_time += get_time() - fp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (early_terminate_last_iteration()) {
    dump_activations();
  }
#endif

  // Add this layer as a gradient source for weight optimizers
  for (auto&& w : m_weights) {
    optimizer* opt = w->get_optimizer();
    if (opt != nullptr) { opt->add_gradient_source(this); }
  }

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  m_fp_time += get_time() - fp_start;
}

void Layer::back_prop() {
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
#endif

  // Backprop the compute function.
  const auto bp_compute_start = get_time();
  bp_compute();
  m_bp_compute_time += get_time() - bp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (early_terminate_last_iteration()) {
    dump_error_signals();
  }
#endif

  // Remove this layer as a gradient source for weight optimizers
  for (auto&& w : m_weights) {
    auto&& opt = w->get_optimizer();
    if (opt != nullptr) { opt->remove_gradient_source(this); }
  }

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  m_bp_time += get_time() - bp_start;
}

bool Layer::update() {
  if (m_frozen) { return true; }
  // Apply any updates.
  const auto update_compute_start = get_time();
  const auto layer_done = update_compute();
  m_update_time += get_time() - update_compute_start;
  return layer_done;
}

void Layer::reset_counters() {
  m_fp_time         = EvalType(0);
  m_fp_compute_time = EvalType(0);
  m_bp_time         = EvalType(0);
  m_bp_compute_time = EvalType(0);
  m_update_time     = EvalType(0);
}

void Layer::summarize_stats(lbann_summary& summarizer, int step) {
  std::string prefix = m_name + "/";
  summarizer.reduce_scalar(prefix + "fp_time", m_fp_time, step);
  summarizer.reduce_scalar(prefix + "bp_time", m_bp_time, step);
  summarizer.reduce_scalar(prefix + "update_time", m_update_time, step);
  summarizer.reduce_scalar_all(prefix + "fp_time", m_fp_time, step);
  summarizer.reduce_scalar_all(prefix + "bp_time", m_bp_time, step);
  summarizer.reduce_scalar_all(prefix + "update_time", m_update_time, step);
  reset_counters();
  // Combine the optimizer step time from all the weights.
  double step_time = 0.0;
  for (weights *w : get_weights()) {
    optimizer *opt = w->get_optimizer();
    if (opt) {
      step_time += opt->get_step_time();
      opt->reset_counters();
    }
  }
  summarizer.reduce_scalar(prefix + "opt_time", step_time, step);
  summarizer.reduce_scalar_all(prefix + "opt_time", step_time, step);
}

void Layer::summarize_matrices(lbann_summary& summarizer, int step) {

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
// Tensor dimension access functions
// ===================================================================

std::vector<int> Layer::get_input_dims(int input_index) const {

  // Get parent layer
  const auto& num_inputs = get_num_parents();
  if (input_index < 0 || input_index >= num_inputs) {
    std::stringstream err;
    err << "attempted to access dimensions of invalid input tensor "
        << "in layer \"" << get_name() << "\" "
        << "(requested index " << input_index << ", but there are "
        << num_inputs << " input tensors)";
    LBANN_ERROR(err.str());
  } else if (m_parent_layers[input_index] == nullptr) {
    std::stringstream err;
    err << "layer \"" << get_name() << "\" "
        << "has a null pointer to parent layer "
        << "(index " << input_index << ")";
    LBANN_ERROR(err.str());
  }
  const auto& parent = *m_parent_layers[input_index];

  // Get dimensions of corresponding output tensor in parent layer
  const auto num_parent_outputs = parent.get_num_children();
  const int parent_output_index = (std::find(parent.m_child_layers.begin(),
                                             parent.m_child_layers.end(),
                                             this)
                                   - parent.m_child_layers.begin());
  if (parent_output_index >= num_parent_outputs) {
    std::stringstream err;
    err << "layer \"" << parent.get_name() << "\" is a parent of "
        << "layer \"" << get_name() << "\", but "
        << "\"" << get_name() << "\" is not a child of "
        << "\"" << parent.get_name() << "\"";
    LBANN_ERROR(err.str());
  }
  return parent.get_output_dims(parent_output_index);

}

El::Int Layer::get_input_size(int input_index) const {
  const auto& dims = get_input_dims(input_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<El::Int>());
  }
}

std::vector<int> Layer::get_output_dims(int output_index) const {
  const auto num_outputs = get_num_children();
  if ((int) m_output_dims_list.size() != num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of output tensor "
        << "in layer \"" << get_name() << "\" "
        << "before they are initialized";
    LBANN_ERROR(err.str());
  } else if (output_index < 0 || output_index >= num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of invalid output tensor "
        << "in layer \"" << get_name() << "\" "
        << "(requested index " << output_index << ", but there are "
        << num_outputs << " output tensors)";
    LBANN_ERROR(err.str());
  }
  return m_output_dims_list[output_index];
}

El::Int Layer::get_output_size(int output_index) const {
  const auto& dims = get_output_dims(output_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<El::Int>());
  }
}

void Layer::set_output_dims(std::vector<int> dims, int output_index) {
  if ((int) m_output_dims_list.size() != get_num_children()
      || (int) m_output_dims_list.size() <= output_index) {
    // Handles case where dims are set before child layers are set
    m_output_dims_list.resize(std::max(get_num_children(),
                                       output_index + 1));
  }
  m_output_dims_list[output_index] = dims;
}

// ===================================================================
// Tensor access functions
// ===================================================================

// Accessing distributed matrices
const AbsDistMat& Layer::get_prev_activations(int parent_index) const {
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
const AbsDistMat& Layer::get_activations(int child_index) const {
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
const AbsDistMat& Layer::get_prev_error_signals(int child_index) const {
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
const AbsDistMat& Layer::get_error_signals(int parent_index) const {
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
AbsDistMat& Layer::get_activations(int child_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_activations(child_index));
}
AbsDistMat& Layer::get_error_signals(int parent_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_error_signals(parent_index));
}

// Accessing local matrices
AbsMat& Layer::get_local_activations(int child_index) {
  return get_activations(child_index).Matrix();
}
AbsMat& Layer::get_local_error_signals(int parent_index) {
  return get_error_signals(parent_index).Matrix();
}
const AbsMat& Layer::get_local_prev_activations(int parent_index) const {
  return get_prev_activations(parent_index).LockedMatrix();
}
const AbsMat& Layer::get_local_activations(int child_index) const {
  return get_activations(child_index).LockedMatrix();
}
const AbsMat& Layer::get_local_prev_error_signals(int child_index) const {
  return get_prev_error_signals(child_index).LockedMatrix();
}
const AbsMat& Layer::get_local_error_signals(int parent_index) const {
  return get_error_signals(parent_index).LockedMatrix();
}

// Accessing matrices corresponding to parent/child layer
const AbsDistMat& Layer::get_activations(const Layer& child) const {
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
const AbsDistMat& Layer::get_error_signals(const Layer& parent) const {
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

void Layer::freeze() {
  m_frozen = true;
  for(auto& w : m_weights) {
    w->freeze();
  }
}

void Layer::unfreeze() {
  m_frozen = false;
  for(auto& w : m_weights) {
    w->unfreeze();
  }
}

bool Layer::is_frozen() const {
  for(auto& w : m_weights) {
    if (w->is_frozen() != m_frozen) {
      LBANN_ERROR("layer and weights of them are inconsistently frozen");
    }
  }
  return m_frozen;
}

void Layer::setup() {
  setup_pointers();
  setup_dims();
  setup_matrices(m_comm->get_trainer_grid());
  // setup_data and setup_gpu are delayed to setup_distconv when
  // distconv is used
#ifndef LBANN_HAS_DISTCONV
  setup_data();
  if (using_gpus()) { setup_gpu(); }
#endif
}

#ifdef LBANN_HAS_DISTCONV
void Layer::setup_distconv() {
  setup_early_termination();
  setup_inter_layer_adaptation();
  setup_keep_original_tensors();
  setup_data();
  if (using_gpus()) { setup_gpu(); }
}
#endif

void Layer::setup_pointers() {
  std::stringstream err;

  // Check that the parent pointers are valid
  for (size_t i = 0; i < m_parent_layers.size(); ++i) {
    const auto* parent = m_parent_layers[i];
    if (parent == nullptr) {
      err << "layer \"" << get_name() << "\" "
          << "has a null pointer to parent layer " << i;
      LBANN_ERROR(err.str());
    }
    const auto& parent_children = parent->m_child_layers;
    if (std::find(parent_children.begin(), parent_children.end(), this)
        == parent_children.end()) {
      err << "layer \"" << parent->get_name() << "\" is a parent of "
          << "layer \"" << get_name() << "\", but "
          << "\"" << get_name() << "\" is not a child of "
          << "\"" << parent->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
  }

  // Check that the child pointers are valid
  for (size_t i = 0; i < m_child_layers.size(); ++i) {
    const auto* child = m_child_layers[i];
    if (child == nullptr) {
      err << "layer \"" << get_name() << "\" "
          << "has a null pointer to child layer " << i;
      LBANN_ERROR(err.str());
    }
    const auto& child_parents = child->m_parent_layers;
    if (std::find(child_parents.begin(), child_parents.end(), this)
        == child_parents.end()) {
      err << "layer \"" << child->get_name() << "\" is a child of "
          << "layer \"" << get_name() << "\", but "
          << "\"" << get_name() << "\" is not a parent of "
          << "\"" << child->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
  }

  // Check that the number of parents/children are valid
  if(m_expected_num_parent_layers >= 0
     && get_num_parents() != m_expected_num_parent_layers) {
    err << get_type() << " layer \"" << get_name() << "\" "
        << "expects " << m_expected_num_parent_layers << " "
        << "parent layer" << (m_expected_num_parent_layers != 1 ? "s" : "")
        << ", but found " << get_num_parents();
    if (get_num_parents() > 0) {
      err << " (";
      for (int i = 0; i < get_num_parents(); ++i) {
        err << (i > 0 ? ", " : "")
            << "\"" << m_parent_layers[i]->get_name() << "\"";
      }
      err << ")";
    }
    LBANN_ERROR(err.str());
  }
  if(m_expected_num_child_layers >= 0
     && get_num_children() != m_expected_num_child_layers) {
    err << get_type() << " layer \"" << get_name() << "\" "
        << "expects " << m_expected_num_child_layers << " "
        << "child layer" << (m_expected_num_child_layers != 1 ? "s" : "")
        << ", but found " << get_num_children();
    if (get_num_children() > 0) {
      err << " (";
      for (int i = 0; i < get_num_children(); ++i) {
        err << (i > 0 ? ", " : "")
            << "\"" << m_child_layers[i]->get_name() << "\"";
      }
      err << ")";
    }
    LBANN_ERROR(err.str());
  }

}

void Layer::setup_dims() {
  m_output_dims_list.resize(get_num_children());
  if (m_hint_layer != nullptr) {
    const auto& hint_dims = m_hint_layer->get_output_dims();
    for (auto& output_dims : m_output_dims_list) {
      output_dims = hint_dims;
    }
  } else if (get_num_parents() > 0) {
    const auto& input_dims = get_input_dims();
    for (auto& output_dims : m_output_dims_list) {
      if (output_dims.empty()) {
        output_dims = input_dims;
      }
    }
  }
}

void Layer::setup_matrices(const El::Grid& grid) {

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

std::unique_ptr<AbsDistMat> Layer::construct_matrix(const El::Grid& grid,
                                                    std::string type,
                                                    El::Int index) {

  // Choose matrix distribution
  El::Distribution col_dist, row_dist;
  El::DistWrap wrap;
  El::Device device = get_device_allocation();
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
  std::unique_ptr<AbsDistMat> mat;
  mat.reset(AbsDistMat::Instantiate(grid, 0,
                                    col_dist, row_dist, wrap, device));

#ifdef LBANN_HAS_GPU
  // Allocate GPU memory with the CUDA API
  if (device == El::Device::GPU) { mat->Matrix().SetMemoryMode(0); }
  // Use pinned memory for data on the host.
  if (device == El::Device::CPU) { mat->Matrix().SetMemoryMode(1); }
#endif // LBANN_HAS_GPU

  return mat;
}

void Layer::setup_data() {
  // Get mini-batch size
  const auto& mini_batch_size = m_model->get_max_mini_batch_size();

  // Initialize input and output tensors
  fp_setup_inputs(mini_batch_size);
  fp_setup_outputs(mini_batch_size);

  // Initialize gradient w.r.t. output tensors
  // Note: We guess whether the tensor is a view or needs to allocate
  // memory, but there are some edge cases that are not handled.
#ifndef LBANN_HAS_DISTCONV
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
#else
  if (keep_original_output()) {
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
  }
#endif

  // Initialize gradient w.r.t. input tensors
  bp_setup_gradient_wrt_inputs(mini_batch_size);

}

void Layer::bp_compute() {
  for (int i = 0; i < get_num_parents(); ++i) {
    El::Zero(get_error_signals(i));
  }
}

void Layer::check_setup() {
  std::stringstream err;

  // Check tensor dimensions
  for (int i = 0; i < get_num_parents(); ++i) {
    const auto& dims = get_input_dims(i);
    if (dims.empty()) {
      err << "layer \"" << get_name() << "\" has "
          << "uninitialized input tensor dimensions "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
    if (std::any_of(dims.begin(), dims.end(),
                    [](int d) { return d <= 0; })) {
      err << "layer \"" << get_name() << "\" has invalid "
          << "input tensor dimensions (";
      for (size_t j = 0; j < dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << dims[j];
      }
      err << " at index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    const auto& dims = get_output_dims(i);
    if (dims.empty()) {
      err << "layer \"" << get_name() << "\" has "
          << "uninitialized output tensor dimensions "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
    if (std::any_of(dims.begin(), dims.end(),
                    [](int d) { return d <= 0; })) {
      err << "layer \"" << get_name() << "\" has invalid "
          << "output tensor dimensions (";
      for (size_t j = 0; j < dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << dims[j];
      }
      err << " at index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }

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

void Layer::replace_weights(Layer* other_layer) {
  if (other_layer == nullptr) {
    LBANN_ERROR("attempted to add null pointer as a replacement layer");
  }

  const std::vector<weights *> other_layer_weights = other_layer->get_weights();
  for (size_t i = 0; i < m_weights.size(); ++i) {
    m_weights[i]->set_values(other_layer_weights[i]->get_values());
  }

}

bool Layer::save_to_checkpoint_shared(persist& p) const {
  return true;
}

bool Layer::load_from_checkpoint_shared(persist& p) {
  return true;
}

bool Layer::save_to_checkpoint_distributed(persist& p) const {
  return true;
}

bool Layer::load_from_checkpoint_distributed(persist& p) {
  return true;
}

void Layer::write_proto(lbann_data::Layer* proto) const {
  proto->Clear();
  proto->set_name(get_name());
  proto->set_type(get_type());
  if(!m_parent_layers.empty()) proto->set_bottom(m_parent_layers.front()->get_name());
  proto->set_top(get_name());
  //Add weights
  for (weights *w : m_weights) {
    auto weight_proto = proto->add_weights_data();
    w->write_proto(weight_proto);
  }
}

void Layer::fp_setup_inputs(El::Int mini_batch_size) {
  if (get_num_parents() < 1) { return; }

#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_input()) {
    return;
  }
#endif

  // Determine distributed matrix alignment
  const auto& alignment_dist
    = m_parent_layers.front()->get_activations(*this).DistData();

  // Iterate through input tensors
  for (int i = 0; i < get_num_parents(); ++i) {

    // Initialize input tensor
    const auto& parent = *m_parent_layers[i];
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

void Layer::fp_setup_outputs(El::Int mini_batch_size) {
  if (get_num_children() < 1) { return; }

#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_output()) {
    return;
  }
#endif

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

void Layer::bp_setup_gradient_wrt_outputs(El::Int mini_batch_size) {
#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_output()) {
    return;
  }
#endif

  for (int i = 0; i < get_num_children(); ++i) {

    // Initialize gradient w.r.t. output tensor
    const auto& child = *m_child_layers[i];
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

void Layer::bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) {
#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_input() || skip_first_layer_bp()) {
    return;
  }
#endif

  for (int i = 0; i < get_num_parents(); ++i) {
    auto& gradient_wrt_input = get_error_signals(i);
    gradient_wrt_input.Empty(false);
    gradient_wrt_input.AlignWith(get_prev_activations(i));
    gradient_wrt_input.Resize(get_input_size(i), mini_batch_size);
  }
}

std::string Layer::get_data_layout_string(data_layout d) const {
  switch(d) {
  case data_layout::DATA_PARALLEL:
    return "data_parallel";
  case data_layout::MODEL_PARALLEL:
    return "model_parallel";
  default:
    LBANN_ERROR("invalid data layout");
  }
}

std::string Layer::get_device_allocation_string(El::Device dev) const {
  switch(dev) {
  case El::Device::CPU:
    return "cpu";
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return "gpu";
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("invalid device allocation");
  }
}

std::string Layer::get_device_allocation_string_short(El::Device dev) const {
  switch(dev) {
  case El::Device::CPU:
    return "C";
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return "G";
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("invalid device allocation");
  }
}

std::string Layer::get_layer_names(const std::vector<const Layer*>& list) {
  std::string layer_names = ((list.size()==0u || !list[0])? "" : list[0]->get_name());

  for (size_t i=1u; i < list.size(); ++i) {
    if (list[i]) layer_names += ", " + list[i]->get_name();
  }
  return layer_names;
}

void Layer::add_parent_layer(const Layer* parent) {
  const auto parent_pos = std::find(m_parent_layers.begin(),
                                    m_parent_layers.end(),
                                    parent);
  if (parent != nullptr
      && parent != this
      && parent_pos == m_parent_layers.end()) {
    m_parent_layers.push_back(parent);
  }
}

void Layer::add_child_layer(const Layer* child) {
  const auto child_pos = std::find(m_child_layers.begin(),
                                   m_child_layers.end(),
                                   child);
  if (child != nullptr
      && child != this
      && child_pos == m_child_layers.end()) {
    m_child_layers.push_back(child);
  }
}

std::vector<Layer*> Layer::get_layer_pointers() {
  std::vector<Layer*> layers;
  for (const auto* parent: m_parent_layers) {
    layers.push_back(const_cast<Layer*>(parent));
  }
  for (const auto* child: m_child_layers) {
    layers.push_back(const_cast<Layer*>(child));
  }
  layers.push_back(const_cast<Layer*>(m_hint_layer));
  return layers;
}

void Layer::set_layer_pointers(std::vector<Layer*> layers) {
  const size_t expected_size = (m_parent_layers.size()
                                + m_child_layers.size()
                                + 1);
  if (layers.size() != expected_size) {
    LBANN_ERROR("attempted to set layer pointers with an invalid number of pointers");
  }
  size_t pos = 0;
  for (auto& parent: m_parent_layers) {
    parent = static_cast<const Layer*>(layers[pos]);
    pos++;
  }
  for (auto& child: m_child_layers) {
    child = static_cast<const Layer*>(layers[pos]);
    pos++;
  }
  m_hint_layer = layers[pos];
  pos++;
}

#ifdef LBANN_HAS_DISTCONV
using namespace dc;

bool Layer::using_distconv() const {
  // Distconv is disabled if no parallel strategy is defined. When no
  // strategy is defined, the layer has the default strategy of all
  // zeros, which is invalid, thus should not be used when distconv is
  // used.
  const auto &ps = get_parallel_strategy();
  ParallelStrategy default_zero_ps;
  if (ps == default_zero_ps) {
    MPIRootPrintStreamInfo()
        << "Disable " << get_name()
        << " as it does not have a parallel strategy.";
    return false;
  }

  // When DISTCONV_ENABLE is defined, all layers included in the
  // variable are enabled.
  char *env = getenv("DISTCONV_ENABLE");
  if (env) {
    std::string s(env);
    auto layer_names = util::split(s, ',');
    for (const auto &name: layer_names) {
      if (get_name() == name) {
        return true;
      }
    }
    MPIRootPrintStreamInfo()
        << "Disable " << get_name()
        << " as its name is not found in DISTCONV_ENABLE";
    return false;
  }

  // It is also disabled when the layer name is included in
  // an environment variable
  env = getenv("DISTCONV_DISABLE");
  if (env) {
    std::string s(env);
    auto layer_names = util::split(s, ',');
    for (const auto &name: layer_names) {
      if (get_name() == name) {
        MPIRootPrintStreamInfo()
            << "Disable " << get_name()
            << " as its name found in DISTCONV_DISABLE";
        return false;
      }
    }
  }

  return true;
}

void Layer::enable_distconv() {
  m_distconv_enabled = using_distconv();
}

void Layer::setup_early_termination() {
  char *count_str = std::getenv("DISTCONV_EARLY_TERMINATE");
  if (count_str) {
    m_exit_count = atoi(count_str);
    dc::MPIRootPrintStreamInfo()
        << "Exiting after " << m_exit_count
        << " iterations\n";
  }
}

void Layer::early_terminate() {
  if (m_exit_count == 0) {
    dc::MPIPrintStreamDebug() << "Early terminate\n";
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    cudaDeviceReset();
    exit(0);
  }
  if (m_exit_count > 0) --m_exit_count;
}

bool Layer::early_terminate_last_iteration() const {
  return m_exit_count == 0;
}

void Layer::setup_inter_layer_adaptation() {
  if (!distconv_enabled()) return;

  MPIRootPrintStreamInfo() << get_name() << ": setup_copyin_copyout";
  const auto &ps = get_parallel_strategy();
  m_parent_copy_in_required = false;
  m_parent_shuffle_required = false;
  for (const auto &p: get_parent_layers()) {
    if (!p->distconv_enabled()) {
      m_parent_copy_in_required = true;
      break;
    } else {
      m_parent_shuffle_required |= ps != p->get_parallel_strategy();
    }
  }
  m_parent_shuffle_required |= m_parent_copy_in_required;
  MPIRootPrintStreamInfo() << "m_parent_copy_in_required: "
                           << m_parent_copy_in_required
                           << ", m_parent_shuffle_required: "
                           << m_parent_shuffle_required;

  m_child_copy_out_required = false;
  m_child_shuffle_required = false;
  for (const auto &c: get_child_layers()) {
    if (!c->distconv_enabled()) {
      m_child_copy_out_required = true;
      break;
    } else {
      m_child_shuffle_required |= ps != c->get_parallel_strategy();
    }
  }
  // If this layer is the last layer, copy the distconv tensor back to
  // LBANN.
  if (get_num_children() == 0) {
    m_child_copy_out_required = true;
  }
  m_child_shuffle_required |= m_child_copy_out_required;
  MPIRootPrintStreamInfo() << "m_child_copy_out_required: "
                           << m_child_copy_out_required
                           << ", m_child_shuffle_required: "
                           << m_child_shuffle_required;
}

void Layer::setup_keep_original_tensors() {
  if (!using_distconv()) return;
  bool env_set = getenv("DISTCONV_KEEP_ORIGINAL_TENSORS");
  m_keep_original_input = env_set || m_parent_copy_in_required;
  m_keep_original_output = env_set || m_child_copy_out_required;
  return;
}

void Layer::setup_tensor_distribution_init(
    std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
    std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
    std::set<dc::Dist*> &updated,
    std::set<dc::Dist*> &fixed) {
  auto &ps = get_parallel_strategy();
  MPIRootPrintStreamInfo() << "Parallel Strategy for layer " << get_name()
                           << ": " << ps;
  int n = ps.sample_groups;
  int c = ps.channel_groups;
  int f = ps.filter_groups;
#ifdef LBANN_DISTCONV_HAS_DEPTH
  int d = ps.depth_groups;
#endif // LBANN_DISTCONV_HAS_DEPTH
  int h = ps.height_groups;
  int w = ps.width_groups;
  int np = m_comm->get_procs_per_trainer();

#ifdef LBANN_DISTCONV_HAS_DEPTH
  const int spatial_prod = d * h * w;
#else
  const int spatial_prod = h * w;
#endif // LBANN_DISTCONV_HAS_DEPTH

  // if only one process is used, do not parallelize
  if (np == 1) {
    n = c = f = h = w = 1;
#ifdef LBANN_DISTCONV_HAS_DEPTH
      d = 1;
#endif // LBANN_DISTCONV_HAS_DEPTH
  }
  if (distconv_enabled()) {
    if (c != f) {
      MPIRootPrintStreamError() << "The numbers of channel and filter decomposition should be the same.";
      throw lbann_exception();
    }
    if (c != 1 || f != 1) {
      MPIRootPrintStreamError() << "Distconv does not support channel/filter parallelization yet. Layer: " << get_name() << ", ps: " << ps;
      throw lbann_exception();
    }
    if (n * c * spatial_prod > np) {
      MPIRootPrintStreamError()
          << "The number of MPI ranks must be at least as large as the number of processes implied by parallel strategy: " << ps;
      throw lbann_exception();
    }
    // Put the remaining factor into the outer-most process dimension
    float rem = np / (float) (n * c * spatial_prod);
    n *= rem;
    ps.sample_splits *= rem;
    if (n * c * spatial_prod != np) {
      MPIRootPrintStreamError()
          << "Can't determine factorization of the number of MPI ranks for parallel strategy: " << ps;
      throw lbann_exception();
    }
    std::string xd_array, xd_array_names;
#ifdef LBANN_DISTCONV_HAS_DEPTH
    xd_array = dc::util::join_xd_array(std::vector<int>({n, c, d, h, w}));
    xd_array_names = "NxCxDxHxW";
#else
    xd_array = dc::util::join_xd_array(std::vector<int>({n, c, h, w}));
    xd_array_names = "NxCxHxW";
#endif // LBANN_DISTCONV_HAS_DEPTH
    MPIRootPrintStreamInfo() << "Process grid of " << xd_array_names << ": "
                             << xd_array;
  }

  assert_always(!distconv_enabled() || (
      spatial_prod * n * c == np && spatial_prod * n * f == np));

  ps.sample_groups = n;
  ps.channel_groups = c;
  ps.filter_groups = f;
#ifdef LBANN_DISTCONV_HAS_DEPTH
  ps.depth_groups = d;
#endif
  ps.height_groups = h;
  ps.width_groups = w;
  // If splits are not set, set them to be equal to the group numbers
  if (ps.sample_splits == 0) ps.sample_splits = n;
  if (ps.channel_splits == 0) ps.channel_splits = c;
  if (ps.filter_splits == 0) ps.filter_splits = f;
#ifdef LBANN_DISTCONV_HAS_DEPTH
    if (ps.depth_splits == 0) ps.depth_splits = d;
#endif
  if (ps.height_splits == 0) ps.height_splits = h;
  if (ps.width_splits == 0) ps.width_splits = w;

  Shape input_locale_shape;
  Shape input_split_shape;
  Shape output_locale_shape;
  Shape output_split_shape;

#ifdef LBANN_DISTCONV_HAS_DEPTH
  input_locale_shape = Shape({w, h, d, c, n});
  input_split_shape = Shape({ps.width_splits, ps.height_splits, ps.depth_splits,
                             ps.channel_splits, ps.sample_splits});
  output_locale_shape = Shape({w, h, d, f, n});
  output_split_shape = Shape({ps.width_splits, ps.height_splits, ps.depth_splits,
                              ps.filter_splits, ps.sample_splits});
#else
  input_locale_shape = Shape({w, h, c, n});
  input_split_shape = Shape({ps.width_splits, ps.height_splits,
                             ps.channel_splits, ps.sample_splits});
  output_locale_shape = Shape({w, h, f, n});
  output_split_shape = Shape({ps.width_splits, ps.height_splits,
                              ps.filter_splits, ps.sample_splits});
#endif

  auto prev_activations_dist =  Dist::make_shared_distribution(
      input_locale_shape, input_split_shape);
  auto activations_dist = Dist::make_shared_distribution(
      output_locale_shape, output_split_shape);
  auto prev_error_signals_dist = activations_dist;
  auto error_signals_dist = prev_activations_dist;
  std::array<Dist, dc::num_dists> layer_dists = {prev_activations_dist,
                                                activations_dist,
                                                error_signals_dist,
                                                prev_error_signals_dist};
  dists.insert(std::make_pair(this, layer_dists));
  invariants.insert(std::make_pair(&dists[this][0], std::set<Dist*>()));
  invariants.insert(std::make_pair(&dists[this][1], std::set<Dist*>()));
  invariants.insert(std::make_pair(&dists[this][2], std::set<Dist*>()));
  invariants.insert(std::make_pair(&dists[this][3], std::set<Dist*>()));
}

void Layer::setup_tensor_distribution_add_adjacent_invariants(
    std::map<const Layer*, std::array<Dist, dc::num_dists>> &dists,
    std::map<Dist*, std::set<Dist*>> &invariants) {
  if (!distconv_enabled()) return;
  auto &layer_dists = dists[this];
  const auto &ps = get_parallel_strategy();
  for (auto &child: get_child_layers()) {
    if (child->distconv_enabled() &&
        child->get_parallel_strategy() == ps) {
      invariants[&layer_dists[1]].insert(
          &dists[child][0]);
      invariants[&layer_dists[3]].insert(
          &dists[child][2]);
    }
  }
  for (auto &parent: get_parent_layers()) {
    if (parent->distconv_enabled() &&
        parent->get_parallel_strategy() == ps) {
      invariants[&layer_dists[0]].insert(
          &dists[parent][1]);
      invariants[&layer_dists[2]].insert(
          &dists[parent][3]);
    }
  }
}

Dist Layer::get_hydrogen_matrix_distribution() {
  using ::distconv::index_t;
  // When rank stride is 1, the distribution is just sample
  // distribution. When it's greater than 1, multiple consecutive
  // ranks of length rank stride share a split in the first
  // dimension. It is assumed that LBANN uses only the
  // NUM_RANKS/STRIDE ranks in a data-parallel input layer to read
  // training data.
  Shape sample_locale_shape(dc::num_dims, 1);
  sample_locale_shape[0] = static_cast<index_t>(dc::get_rank_stride());
  sample_locale_shape[-1] = static_cast<index_t>(dc::get_mpi_num_ranks() / dc::get_rank_stride());
  auto sample_split_shape = sample_locale_shape;
  sample_split_shape[0] = 1;
  auto sample_dist = Dist::make_shared_distribution
      (sample_locale_shape, sample_split_shape);
  return sample_dist;
}

size_t Layer::estimate_memory_usage(const std::array<Dist, dc::num_dists> &dists) {
  if (!distconv_enabled()) {
    return 0;
  }
  auto max_mb = this->m_model->get_max_mini_batch_size();
  size_t usage = 0;
  // fp
  if (m_parent_copy_in_required || m_parent_shuffle_required) {
    usage += get_input_size() * max_mb / dists[0].get_split_shape().size();
  }
  usage += get_output_size() * max_mb / dists[1].get_split_shape().size();
  // bp
  if (m_child_copy_out_required || m_child_shuffle_required) {
    usage += get_output_size() * max_mb / dists[3].get_split_shape().size();
  }
  usage += get_input_size() * max_mb / dists[2].get_split_shape().size();
  return usage * sizeof(DataType);
}

void Layer::setup_prev_activations_tensor(const std::array<Dist, dc::num_dists> &dists) {
  const auto input_tensor_shape = get_input_tensor_shape();
  const LocaleMPI loc(dc::get_mpi_comm(), false);
  const Dist sample_dist = get_hydrogen_matrix_distribution();
  auto input_local_shape = input_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  input_local_shape[-1] = 0;

  if (m_parent_copy_in_required || m_parent_shuffle_required) {
    if (m_parent_copy_in_required) {
      m_prev_activations_const_view = TensorDev(input_tensor_shape, loc,
                                                sample_dist,
                                                input_local_shape);
    } else {
      m_prev_activations_const_view = get_parent_layers()[0]->get_activations_t(*this);
    }
    m_prev_activations_t = TensorDev(input_tensor_shape, loc, dists[0]);
    assert0(m_prev_activations_t.allocate());
    m_prev_activations_t.zero(dc::get_stream());
    m_prev_activations_shuffler = get_tensor_shuffler(
        m_prev_activations_const_view, m_prev_activations_t);
    for (int i = 0; i < 3; ++i) {
      m_prev_activations_shuffler_last_mb[i] = nullptr;
    }
  } else {
    // TODO: Think about the parent has two output tensors (e.g., split).
    m_prev_activations_t = get_parent_layers()[0]->get_activations_t(*this);
    assert_always(m_prev_activations_t.get_distribution() == dists[0]);
  }

  MPIPrintStreamDebug() << get_name() << "; "
                        << "prev activations: " << m_prev_activations_t;
}

Shape Layer::get_activations_tensor_local_shape() const {
  return m_prev_activations_t.get_local_shape();
}

void Layer::setup_activations_tensor(const std::array<Dist, dc::num_dists> &dists,
                                     bool allocate) {
  const LocaleMPI loc(dc::get_mpi_comm(), false);
  const Shape output_tensor_shape = get_output_tensor_shape();
  const auto activations_local_shape =
      get_activations_tensor_local_shape();
  m_activations_t = TensorDev(output_tensor_shape,
                              loc, dists[1], activations_local_shape);
  if (allocate) {
    assert0(m_activations_t.allocate());
    m_activations_t.zero(dc::get_stream());
  }
}

void Layer::setup_activations_copyout_tensor(const std::array<Dist, dc::num_dists> &dists) {
  const LocaleMPI loc(dc::get_mpi_comm(), false);
  const Dist sample_dist = get_hydrogen_matrix_distribution();
  const Shape output_tensor_shape = get_output_tensor_shape();
  auto output_local_shape = output_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  output_local_shape[-1] = 0;
  m_activations_copyout = TensorDev(output_tensor_shape, loc, sample_dist,
                                    output_local_shape);
  if (m_child_copy_out_required) {
    m_activations_shuffler = get_tensor_shuffler(
        m_activations_t, m_activations_copyout);
    for (int i = 0; i < 3; ++i) {
      m_activations_shuffler_last_mb[i] = nullptr;
    }
  }
  MPIPrintStreamDebug() << get_name() << "; "
                        << "activations: " << m_activations_t;
}

void Layer::setup_tensors_bwd(const std::array<Dist, dc::num_dists> &dists) {}

void Layer::setup_distconv_post(size_t) {}

void Layer::setup_prev_error_signals_tensor(const std::array<Dist, dc::num_dists> &dists) {
  const LocaleMPI loc(dc::get_mpi_comm(), false);
  const Dist sample_dist = get_hydrogen_matrix_distribution();
  const Shape output_tensor_shape = get_output_tensor_shape();
  auto output_local_shape = output_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  output_local_shape[-1] = 0;

  if (m_child_copy_out_required || m_child_shuffle_required) {
    if (m_child_copy_out_required) {
      m_prev_error_signals_const_view = TensorDev(output_tensor_shape, loc,
                                                  sample_dist,
                                                  output_local_shape);
    } else {
      m_prev_error_signals_const_view =
          get_child_layers()[0]->get_error_signals_t(*this);
    }
    m_prev_error_signals_t = TensorDev(output_tensor_shape, loc,
                                       dists[3],
                                       m_activations_t.get_local_shape());
    assert0(m_prev_error_signals_t.allocate());
    m_prev_error_signals_t.zero(dc::get_stream());
    m_prev_error_signals_shuffler = get_tensor_shuffler(
        m_prev_error_signals_const_view, m_prev_error_signals_t);
    for (int i = 0; i < 3; ++i) {
      m_prev_error_signals_shuffler_last_mb[i] = nullptr;
    }
  } else {
    m_prev_error_signals_t = get_child_layers()[0]->get_error_signals_t(*this);
    assert_always(m_prev_error_signals_t.get_distribution() ==
                  dists[3]);
  }
  MPIPrintStreamDebug() << get_name() << "; "
                        << "prev error signals: " << m_prev_error_signals_t;
}

void Layer::setup_error_signals_tensor(const std::array<Dist, dc::num_dists> &dists) {
  const Shape input_tensor_shape = get_input_tensor_shape();
  const LocaleMPI loc(dc::get_mpi_comm(), false);
  m_error_signals_t = TensorDev(input_tensor_shape, loc,
                                dists[2],
                                m_prev_activations_t.get_local_shape());
  if (skip_first_layer_bp()) {
    MPIPrintStreamInfo()
        << get_name() << ": skipping allocation of error signals";
  } else {
    assert0(m_error_signals_t.allocate());
    m_error_signals_t.zero(dc::get_stream());
  }
  MPIPrintStreamDebug() << get_name() << "; "
                        << "error signals: " << m_error_signals_t;
}

void Layer::setup_error_signals_copyout_tensor(const std::array<Dist, dc::num_dists> &dists) {
  const Shape input_tensor_shape = get_input_tensor_shape();
  const LocaleMPI loc(dc::get_mpi_comm(), false);
  const Dist sample_dist = get_hydrogen_matrix_distribution();
  auto input_local_shape = input_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  input_local_shape[-1] = 0;

  m_error_signals_copyout = TensorDev(input_tensor_shape, loc, sample_dist,
                                      input_local_shape);
  if (m_parent_copy_in_required && !skip_first_layer_bp()) {
    m_error_signals_shuffler = get_tensor_shuffler(
        m_error_signals_t, m_error_signals_copyout);
    for (int i = 0; i < 3; ++i) {
      m_error_signals_shuffler_last_mb[i] = nullptr;
    }
  }
}

const TensorDev &Layer::get_activations_t(const Layer &child) const {
  assert_always(get_num_children() == 1);
  return m_activations_t;
}

const TensorDev &Layer::get_error_signals_t() const {
  return m_error_signals_t;
}

const TensorDev &Layer::get_error_signals_t(const Layer &parent) const {
  assert_always(get_num_parents() == 1);
  return m_error_signals_t;
}

void Layer::fp_setup_distconv(int mini_batch_size) {
  if (!distconv_enabled()) return;

  early_terminate();

  // Reconfigure the sample dimension as the mini batch size may vary
  // at the end of epoch
  m_prev_activations_t.set_outermost_dimension(mini_batch_size);
  assert_eq((int)m_prev_activations_t.get_shape()[-1],
            mini_batch_size);
  if (m_parent_copy_in_required || m_parent_shuffle_required) {
    m_prev_activations_const_view.set_outermost_dimension(
        mini_batch_size);
    assert_eq((int)m_prev_activations_const_view.get_shape()[-1],
              mini_batch_size);
    if (m_parent_copy_in_required) {
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
  m_activations_t.set_outermost_dimension(mini_batch_size);
  assert_eq((int)m_activations_t.get_shape()[-1],
            mini_batch_size);
  m_activations_copyout.set_outermost_dimension(mini_batch_size);
  assert_eq((int)m_activations_copyout.get_shape()[-1],
            mini_batch_size);
  if (keep_original_output() && m_activations_copyout.is_split_root()) {
    assert_eq((int)m_activations_copyout.get_local_shape()[-1],
              get_activations().LocalWidth());
  }

  ensure_prev_activations();
}

void Layer::bp_setup_distconv(int mini_batch_size) {

  if (!distconv_enabled()) return;

  // Reconfigure the sample dimension as the mini batch size may vary
  // at the end of epoch
  m_prev_error_signals_t.set_outermost_dimension(mini_batch_size);
  assert_always((int)m_prev_error_signals_t.get_shape()[-1] ==
                mini_batch_size);
  if (m_child_copy_out_required || m_child_shuffle_required) {
    m_prev_error_signals_const_view.set_outermost_dimension(mini_batch_size);
    assert_eq((int)m_prev_error_signals_const_view.get_shape()[-1],
              mini_batch_size);
    if (m_child_copy_out_required &&
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
  if (keep_original_input() && !skip_first_layer_bp()
      && m_error_signals_copyout.is_split_root()) {
    assert_eq((int)m_error_signals_copyout.get_local_shape()[-1],
              get_error_signals().LocalWidth());
  }

  ensure_prev_error_signals();
}

namespace {
TensorShuffler *get_shuffler(Layer *layer,
                             TensorShuffler *main_shuffler,
                             TensorShuffler **last_mb_shufflers,
                             const TensorDev &src,
                             const TensorDev &dst) {
  TensorShuffler *shuffler = nullptr;
  const auto& c = static_cast<sgd_execution_context&>(
      layer->get_model()->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  if (layer->get_model()->get_max_mini_batch_size() == mini_batch_size) {
    shuffler = main_shuffler;
  } else {
    // The last remaining mini-batches for the train, validation, and
    // testing modes
    execution_mode mode = layer->get_model()->get_execution_context().get_execution_mode();
    int shfl_idx = static_cast<int>(mode);
    assert_always(shfl_idx >= 0 && shfl_idx < 3);
    if (last_mb_shufflers[shfl_idx] == nullptr) {
      last_mb_shufflers[shfl_idx] = get_tensor_shuffler(src, dst);
    }
    shuffler = last_mb_shufflers[shfl_idx];
  }
  return shuffler;
}
}

void Layer::ensure_prev_activations() {
  if (!(m_parent_copy_in_required || m_parent_shuffle_required)) {
    return;
  }

  if (m_parent_copy_in_required) {
    MPIPrintStreamDebug()
        << "Copying previous activations from sample decomposition";
    assert0(dc::tensor::View(
        m_prev_activations_const_view,
        get_prev_activations().LockedBuffer()));
  } else {
    assert_always(m_parent_shuffle_required);
  }
  TensorShuffler *shuffler =
      get_shuffler(this, m_prev_activations_shuffler,
                   m_prev_activations_shuffler_last_mb,
                   m_prev_activations_const_view,
                   m_prev_activations_t);
  assert_always(shuffler != nullptr);
  shuffler->shuffle_forward(
      m_prev_activations_const_view.get_const_base_ptr(),
      m_prev_activations_t.get_base_ptr(),
      El::GPUManager::Stream());
  this->m_model->clock_start();
}

void Layer::copy_out_activations() {
  if (!m_child_copy_out_required) return;

  this->m_model->clock_end();

  MPIPrintStreamDebug()
      << "Copying activations back to sample decomposition";
  assert0(dc::tensor::View(
      m_activations_copyout, get_activations().Buffer()));
  TensorShuffler *shuffler =
      get_shuffler(this, m_activations_shuffler,
                   m_activations_shuffler_last_mb,
                   m_activations_t, m_activations_copyout);
  assert_always(shuffler != nullptr);
  shuffler->shuffle_forward(
      m_activations_t.get_const_base_ptr(),
      m_activations_copyout.get_base_ptr(),
      El::GPUManager::Stream());
}

void Layer::ensure_prev_error_signals() {
  if (!(m_child_copy_out_required || m_child_shuffle_required)) {
    return;
  }

  if (m_child_copy_out_required) {
    MPIPrintStreamDebug()
        << "Copying previous error signals from sample decomposition";
    assert0(dc::tensor::View(
        m_prev_error_signals_const_view,
        get_prev_error_signals().LockedBuffer()));
  } else {
    assert_always(m_child_shuffle_required);
  }
  TensorShuffler *shuffler =
      get_shuffler(this, m_prev_error_signals_shuffler,
                   m_prev_error_signals_shuffler_last_mb,
                   m_prev_error_signals_const_view,
                   m_prev_error_signals_t);
  assert_always(shuffler != nullptr);
  shuffler->shuffle_forward(
      m_prev_error_signals_const_view.get_const_base_ptr(),
      m_prev_error_signals_t.get_base_ptr(),
      El::GPUManager::Stream());
}

void Layer::copy_out_error_signals() {
  if (!m_parent_copy_in_required) return;

  if (skip_first_layer_bp()) {
    // No need to copy back when the parent is an input layer
    MPIPrintStreamDebug()
        << "Skipping copy back as this layer is the first layer";
    return;
  }

  // No need to copy back as the original layer compute function
  // will be called
  if (m_exit_count == 0) return;

  MPIPrintStreamDebug()
      << "Copying error signals back to sample decomposition";
  assert0(dc::tensor::View(
      m_error_signals_copyout, get_error_signals().Buffer()));
  TensorShuffler *shuffler =
      get_shuffler(this, m_error_signals_shuffler,
                   m_error_signals_shuffler_last_mb,
                   m_error_signals_t, m_error_signals_copyout);
  assert_always(shuffler != nullptr);
  shuffler->shuffle_forward(
      m_error_signals_t.get_const_base_ptr(),
      m_error_signals_copyout.get_base_ptr(),
      El::GPUManager::Stream());
}

const dc::Shape Layer::get_input_tensor_shape() const {
  const auto input_dims = get_input_dims();
  std::vector<int> input_tensor_shape_v(input_dims.rbegin(), input_dims.rend());
  input_tensor_shape_v.push_back(this->m_model->get_max_mini_batch_size());
  return dc::Shape(input_tensor_shape_v);
}
const dc::Shape Layer::get_output_tensor_shape(int output_index) const {
  const auto output_dims = get_output_dims(output_index);
  std::vector<int> output_tensor_shape_v(output_dims.rbegin(), output_dims.rend());
  output_tensor_shape_v.push_back(this->m_model->get_max_mini_batch_size());
  return dc::Shape(output_tensor_shape_v);
}

bool Layer::skip_first_layer_bp() const {
  if (!distconv_enabled()) return false;
  const auto &parents = get_parent_layers();
  if (parents.size() != 1) {
    return false;
  }
  const auto *parent = parents[0];
  // Traverse the graph while skipping split nodes
  while (parent->get_type() == "split") {
    parent = parent->get_parent_layers()[0];
  }
  if (parent->get_type().find("input") == 0) {
    // No need to copy back when the parent is an input layer
    return true;
  }
  return false;
}

#endif

}  // namespace lbann
