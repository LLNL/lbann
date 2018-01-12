////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
//
// lbann_layer .hpp .cpp - Parent class for all layer types
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

namespace lbann {

Layer::Layer(lbann_comm *comm)
  : m_comm(comm),
    m_cudnn(nullptr) {

  // Initialize layer name
  static int num_layers = 0;
  m_name = "layer" + std::to_string(num_layers);
  num_layers++;

  // Initialize neuron tensor dimensions
  m_num_neurons = 0;
  m_num_neuron_dims = 1;
  m_neuron_dims = std::vector<int>(1, 0);
  m_num_prev_neurons = 0;
  m_num_prev_neuron_dims = 1;
  m_neuron_dims = std::vector<int>(1, 0);

  // Default number of parent and child layers
  m_expected_num_parent_layers = 1;
  m_expected_num_child_layers = 1;

  // Initialize model
  m_model = nullptr;

  // Initialize GPU information
  m_using_gpus = false;
#ifdef __LIB_CUDNN
  m_mini_batch_size_per_gpu = 0;
  m_max_mini_batch_size_per_gpu = 0;
  m_prev_neurons_cudnn_desc = nullptr;
  m_neurons_cudnn_desc = nullptr;
#endif // __LIB_CUDNN

  // Reset timing counters
  reset_counters();

}

Layer::Layer(const Layer& other) :
  m_comm(other.m_comm),
  m_num_neurons(other.m_num_neurons),
  m_num_neuron_dims(other.m_num_neuron_dims),
  m_neuron_dims(other.m_neuron_dims),
  m_num_prev_neurons(other.m_num_prev_neurons),
  m_num_prev_neuron_dims(other.m_num_prev_neuron_dims),
  m_prev_neuron_dims(other.m_prev_neuron_dims),
  m_weights(other.m_weights),
  m_parent_layers(other.m_parent_layers),
  m_child_layers(other.m_child_layers),
  m_expected_num_parent_layers(other.m_expected_num_parent_layers),
  m_expected_num_child_layers(other.m_expected_num_child_layers),
  m_model(other.m_model),
  m_using_gpus(other.m_using_gpus),
  m_cudnn(other.m_cudnn),
#ifdef __LIB_CUDNN
  m_mini_batch_size_per_gpu(other.m_mini_batch_size_per_gpu),
  m_max_mini_batch_size_per_gpu(other.m_max_mini_batch_size_per_gpu),
#endif // __LIB_CUDNN
  m_fp_time(other.m_fp_time),
  m_fp_compute_time(other.m_fp_compute_time),
  m_bp_time(other.m_bp_time),
  m_bp_compute_time(other.m_bp_compute_time),
  m_update_time(other.m_update_time),
  m_name(other.m_name) {

  // Deep matrix copies
  m_prev_activations   = other.m_prev_activations;
  m_activations        = other.m_activations;
  m_prev_error_signals = other.m_prev_error_signals;
  m_error_signals      = other.m_error_signals;
  for (auto& m : m_prev_activations)   { m = m->Copy(); }
  for (auto& m : m_activations)        { m = m->Copy(); }
  for (auto& m : m_prev_error_signals) { m = m->Copy(); }
  for (auto& m : m_error_signals)      { m = m->Copy(); }

#ifdef __LIB_CUDNN
  // Copy GPU objects
  m_prev_activations_d = other.m_prev_activations_d;
  m_activations_d = other.m_activations_d;
  m_prev_error_signals_d = other.m_prev_error_signals_d;
  m_error_signals_d = other.m_error_signals_d;
  m_prev_neurons_cudnn_desc = nullptr;
  m_neurons_cudnn_desc = nullptr;
  cudnn::copy_tensor_cudnn_desc(other.m_prev_neurons_cudnn_desc,
                                m_prev_neurons_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_neurons_cudnn_desc,
                                m_neurons_cudnn_desc);
#endif // __LIB_CUDNN
}

Layer& Layer::operator=(const Layer& other) {
  
  // Shallow copies
  m_comm = other.m_comm;
  m_num_neurons = other.m_num_neurons;
  m_num_neuron_dims = other.m_num_neuron_dims;
  m_neuron_dims = other.m_neuron_dims;
  m_num_prev_neurons = other.m_num_prev_neurons;
  m_num_prev_neuron_dims = other.m_num_prev_neuron_dims;
  m_prev_neuron_dims = other.m_prev_neuron_dims;
  m_weights = other.m_weights;
  m_parent_layers = other.m_parent_layers;
  m_child_layers = other.m_child_layers;
  m_expected_num_parent_layers = other.m_expected_num_parent_layers;
  m_expected_num_child_layers = other.m_expected_num_child_layers;
  m_model = other.m_model;
  m_using_gpus = other.m_using_gpus;
  m_cudnn = other.m_cudnn;
#ifdef __LIB_CUDNN
  m_mini_batch_size_per_gpu = other.m_mini_batch_size_per_gpu;
  m_max_mini_batch_size_per_gpu = other.m_max_mini_batch_size_per_gpu;
#endif // __LIB_CUDNN
  m_fp_time = other.m_fp_time;
  m_fp_compute_time = other.m_fp_compute_time;
  m_bp_time = other.m_bp_time;
  m_bp_compute_time = other.m_bp_compute_time;
  m_update_time = other.m_update_time;
  m_name = other.m_name;

  // Deep matrix copies
  deallocate_matrices();
  m_prev_activations   = other.m_prev_activations;
  m_activations        = other.m_activations;
  m_prev_error_signals = other.m_prev_error_signals;
  m_error_signals      = other.m_error_signals;
  for (auto& m : m_prev_activations)   { m = m->Copy(); }
  for (auto& m : m_activations)        { m = m->Copy(); }
  for (auto& m : m_prev_error_signals) { m = m->Copy(); }
  for (auto& m : m_error_signals)      { m = m->Copy(); }

#ifdef __LIB_CUDNN
  m_prev_activations_d = other.m_prev_activations_d;
  m_activations_d = other.m_activations_d;
  m_prev_error_signals_d = other.m_prev_error_signals_d;
  m_error_signals_d = other.m_error_signals_d;
  cudnn::copy_tensor_cudnn_desc(other.m_prev_neurons_cudnn_desc,
                                m_prev_neurons_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_neurons_cudnn_desc,
                                m_neurons_cudnn_desc);
#endif // __LIB_CUDNN

  return *this;
}

Layer::~Layer() {
#ifdef __LIB_CUDNN
  if(m_prev_neurons_cudnn_desc != nullptr) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_prev_neurons_cudnn_desc));
  }
  if(m_neurons_cudnn_desc != nullptr) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_neurons_cudnn_desc));
  }
#endif // __LIB_CUDNN
  deallocate_matrices();
}

void Layer::forward_prop() {
  const auto fp_start = get_time();

  // Setup matrix data, e.g. input matrices
  fp_setup_data(m_model->get_current_mini_batch_size());

#ifdef __LIB_CUDNN
  if(m_using_gpus) {

    // Determine mini-batch size per GPU
    const int num_gpus = m_cudnn->get_num_gpus();
    const int local_mini_batch_size = (get_num_parents() > 0 ?
                                       get_prev_activations().LocalWidth() :
                                       get_activations().LocalWidth());
    m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

    // Set tensor descriptors
    cudnn::set_tensor_cudnn_desc(m_prev_neurons_cudnn_desc,
                                 m_mini_batch_size_per_gpu,
                                 m_prev_neuron_dims);
    cudnn::set_tensor_cudnn_desc(m_neurons_cudnn_desc,
                                 m_mini_batch_size_per_gpu,
                                 m_neuron_dims);

    // Transfer inputs from CPU to GPUs if needed
    for (int i = 0; i < get_num_parents(); ++i) {
      const auto& parent = *m_parent_layers[i];
      auto& gpu_prev_activations = m_prev_activations_d[i];
      if (parent.using_gpus()) {
        parent.get_gpu_fp_output(gpu_prev_activations, this);
      } else {
        gpu_prev_activations.resize(m_num_prev_neurons,
                                    m_max_mini_batch_size_per_gpu);
        m_cudnn->scatter_to_gpus(gpu_prev_activations.get_data(),
                                 get_local_prev_activations(i),
                                 m_mini_batch_size_per_gpu);
      }
    }

  }
#endif // __LIB_CUDNN

  // Apply layer's compute function
  const auto fp_compute_start = get_time();
  fp_compute();
  m_fp_compute_time += get_time() - fp_compute_start;

#ifdef __LIB_CUDNN
  // Transfer outputs from GPUs to CPU if needed
  if (m_using_gpus) {
    for (int i = 0; i < get_num_children(); ++i) {
      if (!m_child_layers[i]->using_gpus()) {
        m_cudnn->gather_from_gpus(get_local_activations(i),
                                  m_activations_d[i].get_data(),
                                  m_mini_batch_size_per_gpu);
        m_cudnn->synchronize();
      }
    }
  }
#endif // __LIB_CUDNN

  m_fp_time += get_time() - fp_start;
}

void Layer::back_prop() {
  const auto bp_start = get_time();

  // Setup matrix data, e.g. input matrices
  bp_setup_data(m_model->get_current_mini_batch_size());

#ifdef __LIB_CUDNN
  if(m_using_gpus) {
    // Transfer inputs from CPU to GPUs if needed
    for (int i = 0; i < get_num_children(); ++i) {
      const auto& child = *m_child_layers[i];
      auto& gpu_prev_error_signals = m_prev_error_signals_d[i];
      if (child.using_gpus()) {
        child.get_gpu_bp_output(gpu_prev_error_signals, this);
      } else {
        gpu_prev_error_signals.resize(m_num_neurons,
                                      m_max_mini_batch_size_per_gpu);
        m_cudnn->scatter_to_gpus(gpu_prev_error_signals.get_data(),
                                 get_local_prev_error_signals(i),
                                 m_mini_batch_size_per_gpu);
      }
    }
  }
#endif // __LIB_CUDNN

  // Backprop the compute function.
  const auto bp_compute_start = get_time();
  bp_compute();
  m_bp_compute_time += get_time() - bp_compute_start;

#ifdef __LIB_CUDNN
  // Transfer outputs from GPUs to CPU if needed
  if (m_using_gpus) {
    for (int i = 0; i < get_num_parents(); ++i) {
      if (!m_parent_layers[i]->using_gpus()) {
        m_cudnn->gather_from_gpus(get_local_error_signals(i),
                                  m_error_signals_d[i].get_data(),
                                  m_mini_batch_size_per_gpu);
        m_cudnn->synchronize();
      }
    }
  }
#endif // __LIB_CUDNN

  m_bp_time += get_time() - bp_start;
}

bool Layer::update() {
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
}

void Layer::summarize_matrices(lbann_summary& summarizer, int step) {

  // Summarize activation matrices
  const int num_children = get_num_children();
  for (int i = 0; i < num_children; ++i) {
    std::string prefix = m_name + "/activations";
    if (num_children > 1) { prefix += std::to_string(i); }
    summarizer.reduce_mean(prefix + "/mean", *m_activations[i], step);
    summarizer.reduce_min(prefix + "/min", *m_activations[i], step);
    summarizer.reduce_max(prefix + "/max", *m_activations[i], step);
    summarizer.reduce_stdev(prefix + "/stdev", *m_activations[i], step);
    summarizer.reduce_2norm(prefix + "/2norm2", *m_activations[i], step);
  }

  // Summarize error signal matrices
  const int num_parents = get_num_parents();
  for (int i = 0; i < num_parents; ++i) {
    std::string prefix = m_name + "/error_signals";
    if (num_parents > 1) { prefix += std::to_string(i); }
    summarizer.reduce_mean(prefix + "/mean", *m_error_signals[i], step);
    summarizer.reduce_min(prefix + "/min", *m_error_signals[i], step);
    summarizer.reduce_max(prefix + "/max", *m_error_signals[i], step);
    summarizer.reduce_stdev(prefix + "/stdev", *m_error_signals[i], step);
    summarizer.reduce_2norm(prefix + "/2norm2", *m_error_signals[i], step);
  }

}

AbsDistMat& Layer::get_prev_activations(int parent_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_prev_activations(parent_index));
}
const AbsDistMat& Layer::get_prev_activations(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_prev_activations.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access invalid previous activation matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_prev_activations.size() << " previous activation matrices)";
    throw lbann_exception(err.str());
  }
  return *m_prev_activations[parent_index];
}

AbsDistMat& Layer::get_activations(int child_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_activations(child_index));
}
const AbsDistMat& Layer::get_activations(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_activations.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access invalid activation matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_activations.size() << " activation matrices)";
    throw lbann_exception(err.str());
  }
  return *m_activations[child_index];
}

AbsDistMat& Layer::get_prev_error_signals(int child_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_prev_error_signals(child_index));
}
const AbsDistMat& Layer::get_prev_error_signals(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_prev_error_signals.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access invalid previous error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_prev_error_signals.size() << " previous error signal matrices)";
    throw lbann_exception(err.str());
  }
  return *m_prev_error_signals[child_index];
}

AbsDistMat& Layer::get_error_signals(int parent_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_error_signals(parent_index));
}
const AbsDistMat& Layer::get_error_signals(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_error_signals.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access invalid error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_error_signals.size() << " error signal matrices)";
    throw lbann_exception(err.str());
  }
  return *m_error_signals[parent_index];
}

Mat& Layer::get_local_prev_activations(int parent_index) {
  return get_prev_activations(parent_index).Matrix();
}
const Mat& Layer::get_local_prev_activations(int parent_index) const {
  return get_prev_activations(parent_index).LockedMatrix();
}
Mat& Layer::get_local_activations(int child_index) {
  return get_activations(child_index).Matrix();
}
const Mat& Layer::get_local_activations(int child_index) const {
  return get_activations(child_index).LockedMatrix();
}
Mat& Layer::get_local_prev_error_signals(int child_index) {
  return get_prev_error_signals(child_index).Matrix();
}
const Mat& Layer::get_local_prev_error_signals(int child_index) const {
  return get_prev_error_signals(child_index).LockedMatrix();
}
Mat& Layer::get_local_error_signals(int parent_index) {
  return get_error_signals(parent_index).Matrix();
}
const Mat& Layer::get_local_error_signals(int parent_index) const {
  return get_error_signals(parent_index).LockedMatrix();
}

void Layer::clear_error_signals(int mini_batch_size) {
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_using_gpus) {
#ifdef __LIB_CUDNN
      m_error_signals_d[i].zero();
#endif // __LIB_CUDNN
    } else {
      El::Zeros(get_error_signals(i),
                get_num_prev_neurons(i),
                mini_batch_size);
    }
  }
}

void Layer::setup() {
  setup_pointers();
  setup_dims();
  setup_matrices(m_comm->get_model_grid());
  setup_data();
  setup_views();
  if (m_using_gpus) {
    setup_gpu();
  }
}

void Layer::setup_pointers() {

  // Check if the number of parents/children are valid
  if(m_expected_num_parent_layers >= 0
     && get_num_parents() != m_expected_num_parent_layers) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "layer " << m_name << " has an invalid number of parent layers "
        << "(expected " << m_expected_num_parent_layers << ", "
        << "but found " << m_parent_layers.size() << ")";
    throw lbann_exception(err.str());
  }
  if(m_expected_num_child_layers >= 0
     && get_num_children() != m_expected_num_child_layers) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "layer " << m_name << " has an invalid number of child layers "
        << "(expected " << m_expected_num_child_layers << ", "
        << "but found " << m_child_layers.size() << ")";
    throw lbann_exception(err.str());
  }

}

void Layer::setup_dims() {

  // Get dimensions of previous neuron tensor
  if(m_parent_layers.empty()) {
    m_prev_neuron_dims.assign(1, 0);
  } else {
    m_prev_neuron_dims = m_parent_layers.front()->fp_output_dims(this);
  }
  m_num_prev_neuron_dims = m_prev_neuron_dims.size();
  m_num_prev_neurons = std::accumulate(m_prev_neuron_dims.begin(),
                                       m_prev_neuron_dims.end(),
                                       1,
                                       std::multiplies<int>());
  
  // Set neuron tensor dimensions equal to previous neuron tensor
  m_num_neurons = m_num_prev_neurons;
  m_num_neuron_dims = m_num_prev_neuron_dims;
  m_neuron_dims = m_prev_neuron_dims;
  
}

void Layer::setup_matrices(const El::Grid& grid) {

  // Delete any previously allocated matrices
  deallocate_matrices();

  // Allocate input and output matrices for forward an back prop
  switch (get_data_layout()) {
  case data_layout::MODEL_PARALLEL:
    for (int i = 0; i < get_num_parents(); ++i) {
      m_prev_activations.push_back(new MCMRMat(grid));
      m_error_signals.push_back(new MCMRMat(grid));
    }
    for (int i = 0; i < get_num_children(); ++i) {
      m_activations.push_back(new MCMRMat(grid));
      m_prev_error_signals.push_back(new MCMRMat(grid));
    }
    break;
  case data_layout::DATA_PARALLEL:
    for (int i = 0; i < get_num_parents(); ++i) {
      m_prev_activations.push_back(new StarVCMat(grid));
      m_error_signals.push_back(new StarVCMat(grid));
    }
    for (int i = 0; i < get_num_children(); ++i) {
      m_activations.push_back(new StarVCMat(grid));
      m_prev_error_signals.push_back(new StarVCMat(grid));
    }
    break;
  default:
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid distributed matrix layout";
    throw lbann_exception(err.str());
  }

}

void Layer::setup_data() {

  // Initialize matrices
  const int mini_batch_size = m_model->get_max_mini_batch_size();
  if(mini_batch_size <= 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid mini-batch size "
        << "(" << mini_batch_size << ")";
    throw lbann_exception(err.str());
  }

  // Initialize forward prop output
  for (int i = 0; i < get_num_children(); ++i) {
    El::Zeros(*m_activations[i],
              get_num_neurons(i),
              mini_batch_size);
  }

  // Initialize backward prop output
  for (int i = 0; i < get_num_parents(); ++i) {
    El::Zeros(*m_error_signals[i],
              get_num_prev_neurons(i),
              mini_batch_size);
  }

#ifdef __LIB_CUDNN
  // Pin host memory if needed for GPU memory transfers
  pin_data();
#endif // __LIB_CUDNN

}

void Layer::setup_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception(
      std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
      "Layer: cuDNN not detected");
#else

  // Throw error if data layout is not data parallel
  // TODO: Design a more general interface
  if(get_data_layout() != data_layout::DATA_PARALLEL) {
    throw lbann_exception(
      std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
      "Layer: GPUs are currently only supported for data parallel layers");
  }
  
  // Split mini-batch amongst GPUs
  const int num_gpus = m_cudnn->get_num_gpus();
  const int num_processes = m_comm->get_procs_per_model();
  const int mini_batch_size = m_model->get_max_mini_batch_size();
  const int local_mini_batch_size = (mini_batch_size + num_processes - 1) / num_processes;
  m_max_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;
  m_mini_batch_size_per_gpu = m_max_mini_batch_size_per_gpu;

  // Set tensor descriptors
  cudnn::set_tensor_cudnn_desc(m_prev_neurons_cudnn_desc,
                               m_mini_batch_size_per_gpu,
                               m_prev_neuron_dims);
  cudnn::set_tensor_cudnn_desc(m_neurons_cudnn_desc,
                               m_mini_batch_size_per_gpu,
                               m_neuron_dims);

  // Initialize GPU memory
  for (int i = 0; i < get_num_children(); ++i) {
    m_activations_d.emplace_back(m_cudnn,
                                 get_num_neurons(i),
                                 mini_batch_size);
    m_prev_error_signals_d.emplace_back(m_cudnn);
  }
  for (int i = 0; i < get_num_parents(); ++i) {
    m_prev_activations_d.emplace_back(m_cudnn);
    m_error_signals_d.emplace_back(m_cudnn,
                                   get_num_prev_neurons(i),
                                   mini_batch_size);
  }

#endif // __LIB_CUDNN
}

void Layer::check_setup() {
  std::stringstream err;

  // Check that matrices matches number of parent/child layers
  const int num_parents = get_num_parents();
  const int num_children = get_num_children();
  if ((int) m_prev_activations.size() != num_parents
      || (int) m_activations.size() != num_children) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "layer " << m_name << " has an invalid number of "
        << "forward prop matrices (expected "
        << num_parents << " input and " << num_children << " output, "
        << "but found " << m_prev_activations.size() << " and "
        << m_activations.size() << " respectively) ";
    throw lbann_exception(err.str());
  }
  if ((int) m_prev_error_signals.size() != num_children
      || (int) m_error_signals.size() != num_parents) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "layer " << m_name << " has an invalid number of "
        << "backward prop matrices (expected "
        << num_children << " input and " << num_parents << " output. "
        << "but found " << m_prev_error_signals.size() << " and "
        << m_error_signals.size() << " respectively) ";
    throw lbann_exception(err.str());
  }

  // Check that matrices are initialized
  for (const auto& m : m_prev_activations) {
    if (m == nullptr) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "layer " << m_name << " has an uninitialized previous activation matrix";
      throw lbann_exception(err.str());
    }
  }
  for (const auto& m : m_activations) {
    if (m == nullptr) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "layer " << m_name << " has an uninitialized activation matrix";
      throw lbann_exception(err.str());
    }
  }
  for (const auto& m : m_prev_error_signals) {
    if (m == nullptr) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "layer " << m_name << " has an uninitialized previous error signal matrix";
      throw lbann_exception(err.str());
    }
  }
  for (const auto& m : m_error_signals) {
    if (m == nullptr) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "layer " << m_name << " has an uninitialized error signal matrix";
      throw lbann_exception(err.str());
    }
  }

  // Check that number of neurons is greater than zero
  if (m_num_neurons <= 0) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "layer " << m_name << " has invalid output dimensions "
        << "(" << m_neuron_dims[0];
    for (size_t i = 1; i < m_neuron_dims.size(); ++i) {
      err << "x" << m_neuron_dims[i];
    }
    err << ")";
    throw lbann_exception(err.str());
  }

}

void Layer::replace_weights(Layer* other_layer) {
  if (other_layer == nullptr) {
    throw lbann_exception("Layer::replace_weights: Attempted to add null pointer as a replacement layer.");
  }
  
  const std::vector<weights *> other_layer_weights = other_layer->get_weights();
  for (size_t i = 0; i < m_weights.size(); ++i) {
    m_weights[i]->set_values(other_layer_weights[i]->get_values());
  }

}

void Layer::deallocate_matrices() {
#ifdef __LIB_CUDNN
  // Deallocate GPU memory
  m_prev_activations_d.clear();
  m_activations_d.clear();
  m_prev_error_signals_d.clear();
  m_error_signals_d.clear();

  // Unpin matrices
  if (m_cudnn != nullptr) {
    for (const auto& m : m_prev_activations) {
      if (m != nullptr) { m_cudnn->unpin_matrix(*m); }
    }
    for (const auto& m : m_activations) {
      if (m != nullptr) { m_cudnn->unpin_matrix(*m); }
    }
    for (const auto& m : m_prev_error_signals) {
      if (m != nullptr) { m_cudnn->unpin_matrix(*m); }
    }
    for (const auto& m : m_error_signals) {
      if (m != nullptr) { m_cudnn->unpin_matrix(*m); }
    }
  }
#endif // __LIB_CUDNN

  // Deallocate matrices
  for (const auto& m : m_prev_activations) {
    if (m != nullptr) delete m;
  }
  for (const auto& m : m_activations) {
    if (m != nullptr) delete m;
  }
  for (const auto& m : m_prev_error_signals) {
    if (m != nullptr) delete m;
  }
  for (const auto& m : m_error_signals) {
    if (m != nullptr) delete m;
  }
  m_prev_activations.clear();
  m_activations.clear();
  m_prev_error_signals.clear();
  m_error_signals.clear();

}

bool Layer::saveToCheckpoint(int fd, const char *filename, size_t *bytes) const {
  //writeDist(fd, filename, *m_weights, bytes);
  
  // Need to catch return value from function
  // m_optimizer->saveToCheckpoint(fd, filename, bytes);
  return true;
}

bool Layer::loadFromCheckpoint(int fd, const char *filename, size_t *bytes) {
  // TODO: implement reader for other matrix distributions
  //readDist(fd, filename, (DistMat&) *m_weights, bytes);

  // Need to catch return value from function
  // m_optimizer->loadFromCheckpoint(fd, filename, bytes);
  return true;
}

bool Layer::save_to_checkpoint_shared(persist& p) const {
  //for (weights *w : m_weights) {
  //  w->saveToCheckpointShared(p);  
  //}
  return true;
}

bool Layer::load_from_checkpoint_shared(persist& p) {
  //for (weights *w : m_weights) {
  //  w->loadFromCheckpointShared(p);
  //}

  return true;
}

void Layer::write_proto(lbann_data::Layer* proto) const {
  proto->set_name(get_name());
}

void Layer::fp_setup_data(int mini_batch_size) {

  // Get previous activations from parent layers
  for (int i = 0; i < get_num_parents(); ++i) {
    const auto& parent = m_parent_layers[i];
    auto& input = *m_prev_activations[i];
    parent->get_fp_output(input, this);
    const int expected_height = get_num_prev_neurons(i);
    if (input.Height() != expected_height
        || input.Width() != mini_batch_size) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << get_name() << " expected a "
          << expected_height << " x " << mini_batch_size
          << " input matrix from " << parent->get_name()
          << " during forward prop, but got a "
          << input.Height() << " x " << input.Width() << " matrix";
      throw lbann_exception(err.str());
    }
  }

  // Initialize activation matrices
  for (int i = 0; i < get_num_children(); ++i) {
    m_activations[i]->Resize(get_num_neurons(i), mini_batch_size);
  }
  
}

void Layer::bp_setup_data(int mini_batch_size) {

  // Get previous error signal matrices from child layers
  for (int i = 0; i < get_num_children(); ++i) {
    const auto& child = m_child_layers[i];
    auto& input = *m_prev_error_signals[i];
    child->get_bp_output(input, this);
    const int expected_height = get_num_neurons(i);
    if (input.Height() != expected_height
        || input.Width() != mini_batch_size) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << get_name() << " expected a "
          << expected_height << " x " << mini_batch_size
          << " input matrix from " << child->get_name()
          << " during backward prop, but got a "
          << input.Height() << " x " << input.Width() << " matrix";
      throw lbann_exception(err.str());
    }
  }

}


#ifdef __LIB_CUDNN
void Layer::pin_data() {
  for (int i = 0; i < get_num_parents(); ++i) {
    const auto& parent = *m_parent_layers[i];
    if (using_gpus() && !parent.using_gpus()) {
      m_cudnn->pin_matrix(get_error_signals(i));
      if (get_prev_activations().DistData()
          != parent.get_activations().DistData()) {
        m_cudnn->pin_matrix(get_prev_activations(i));
      }
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    const auto& child = *m_child_layers[i];
    if (using_gpus() && !child.using_gpus()) {
      m_cudnn->pin_matrix(get_activations(i));
      if (get_prev_error_signals().DistData()
          != child.get_error_signals().DistData()) {
        m_cudnn->pin_matrix(get_prev_error_signals(i));
      }
    }
  }
}

#endif // __LIB_CUDNN

void Layer::get_fp_output(AbsDistMat& output, const Layer* child) const {

  // Get activation matrix corresponding to child layer
  const size_t child_index = (std::find(m_child_layers.begin(),
                                        m_child_layers.end(),
                                        child)
                              - m_child_layers.begin());
  if (child_index >= m_child_layers.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << get_name() << " has no forward prop output corresponding to "
        << child->get_name();
    throw lbann_exception(err.str());
  }
  const auto& activation = *m_activations[child_index];

  // Put view or copy of activation matrix in output matrix
  if(activation.DistData() == output.DistData()) {
    El::LockedView(output, activation);
  }
  else {
    El::Copy(activation, output);
  }

}

void Layer::get_bp_output(AbsDistMat& output, const Layer* parent) const {

  // Get error signal matrix corresponding to parent layer
  const size_t parent_index = (std::find(m_parent_layers.begin(),
                                         m_parent_layers.end(),
                                         parent)
                               - m_parent_layers.begin());
  if (parent_index >= m_parent_layers.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << get_name() << " has no backward prop output corresponding to "
        << parent->get_name();
    throw lbann_exception(err.str());
  }
  const auto& error_signal = *m_error_signals[parent_index];

  // Put view or copy of error signal matrix in output matrix
  if(error_signal.DistData() == output.DistData()) {
    El::LockedView(output, error_signal);
  }
  else {
    El::Copy(error_signal, output);
  }

}

#ifdef __LIB_CUDNN

void Layer::get_gpu_fp_output(cudnn::matrix& output_d,
                              const Layer* child) const {
  const size_t child_index = (std::find(m_child_layers.begin(),
                                        m_child_layers.end(),
                                        child)
                              - m_child_layers.begin());
  if (child_index >= m_child_layers.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << get_name() << " has no forward prop output corresponding to "
        << child->get_name();
    throw lbann_exception(err.str());
  }
  output_d.view(m_activations_d[child_index]);
}

void Layer::get_gpu_bp_output(cudnn::matrix& output_d,
                              const Layer* parent) const {
  const size_t parent_index = (std::find(m_parent_layers.begin(),
                                         m_parent_layers.end(),
                                         parent)
                               - m_parent_layers.begin());
  if (parent_index >= m_parent_layers.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << get_name() << " has no backward prop output corresponding to "
        << parent->get_name();
    throw lbann_exception(err.str());
  }
  output_d.view(m_error_signals_d[parent_index]);
}

#endif // __LIB_CUDNN

std::string Layer::get_data_layout_string(data_layout d) const {
  switch(d) {
  case data_layout::DATA_PARALLEL:
    return "data_parallel";
  case data_layout::MODEL_PARALLEL:
    return "model_parallel";
  default:
    throw lbann_exception(
      std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
      "Layer: invalid data layout");
  }
}

const std::vector<int> Layer::fp_output_dims(const Layer* next_layer) const {
  return m_neuron_dims;
}

std::vector<const Layer*>& Layer::get_parent_layers() {
  return m_parent_layers;
}

const std::vector<const Layer*>& Layer::get_parent_layers() const {
  return m_parent_layers;
}

std::vector<const Layer*>& Layer::get_child_layers() {
  return m_child_layers;
}

const std::vector<const Layer*>& Layer::get_child_layers() const {
  return m_child_layers;
}

std::string Layer::get_layer_names(const std::vector<const Layer*>& list) {
  std::string layer_names = ((list.size()==0u || !list[0])? "" : list[0]->get_name());

  for (size_t i=1u; i < list.size(); ++i) {
    if (list[i]) layer_names += ", " + list[i]->get_name();
  }
  return layer_names;
}

void Layer::add_parent_layer(const Layer* parent) {
  auto parent_pos = std::find(m_parent_layers.begin(),
                              m_parent_layers.end(),
                              parent);
  if(parent != nullptr
     && parent != this
     && parent_pos == m_parent_layers.end()) {
    m_parent_layers.push_back(parent);
  }
}

void Layer::add_child_layer(const Layer* child) {
  auto child_pos = std::find(m_child_layers.begin(),
                             m_child_layers.end(),
                             child);
  if(child != nullptr
     && child != this
     && child_pos == m_child_layers.end()) {
    m_child_layers.push_back(child);
  }
}

void Layer::clear_parent_layers() {
  m_parent_layers.clear();
}

void Layer::clear_child_layers() {
  m_child_layers.clear();
}

std::vector<Layer*> Layer::get_layer_pointers() {
  std::vector<Layer*> layers;
  for(const Layer* parent: m_parent_layers) {
    layers.push_back(const_cast<Layer*>(parent));
  }
  for(const Layer* child: m_child_layers) {
    layers.push_back(const_cast<Layer*>(child));
  }
  return layers;
}

void Layer::set_layer_pointers(std::vector<Layer*> layers) {
  if(layers.size() != m_parent_layers.size() + m_child_layers.size()) {
    throw lbann_exception(
      std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
      "Layer: attempted to set layer pointers with an invalid number of pointers");
  }
  size_t pos = 0;
  for(const Layer*& parent: m_parent_layers) {
    parent = (const Layer*) layers[pos];
    pos++;
  }
  for(const Layer*& child: m_child_layers) {
    child = (const Layer*) layers[pos];
    pos++;
  }
}



}  // namespace lbann
