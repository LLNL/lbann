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
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// Asynchronous memory transfers for input data
// Note: This introduces a race condition. It is possible for the
// input data to be modified by another layer before it is used by
// this layer.
// #define ASYNC_INPUT_MEMORY_TRANSFER

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
  const auto& mini_batch_size = m_model->get_current_mini_batch_size();
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
  const auto& mini_batch_size = m_model->get_current_mini_batch_size();
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

int Layer::get_input_size(int input_index) const {
  const auto& dims = get_input_dims(input_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<int>());
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

int Layer::get_output_size(int output_index) const {
  const auto& dims = get_output_dims(output_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<int>());
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
  setup_data();
  if (using_gpus()) { setup_gpu(); }
}

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

}  // namespace lbann
