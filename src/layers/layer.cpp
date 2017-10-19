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

using namespace El;

namespace lbann {

template <>
void Layer::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>() {

  // Instantiate matrices with MC,MR distribution
  El::Grid& grid = m_comm->get_model_grid();
  m_prev_activations  = new DistMat(grid);
  m_activations       = new DistMat(grid);
  m_prev_error_signal = new DistMat(grid);
  m_error_signal      = new DistMat(grid);

  // Construct matrix views
  m_activations_v = m_activations->Construct(m_activations->Grid(),
                                             m_activations->Root());
  m_error_signal_v = m_error_signal->Construct(m_error_signal->Grid(),
                                               m_error_signal->Root());

}

template<>
void Layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>() {

  // Instantiate matrices with STAR,VC distribution
  El::Grid& grid = m_comm->get_model_grid();
  m_prev_activations  = new StarVCMat(grid);
  m_activations       = new StarVCMat(grid);
  m_prev_error_signal = new StarVCMat(grid);
  m_error_signal      = new StarVCMat(grid);

  // Construct matrix views
  m_activations_v = m_activations->Construct(m_activations->Grid(),
                                             m_activations->Root());
  m_error_signal_v = m_error_signal->Construct(m_error_signal->Grid(),
                                               m_error_signal->Root());

}

Layer::Layer(const int index, lbann_comm *comm)
  : m_index(index),
    m_comm(comm),
    m_execution_mode(execution_mode::training),
    m_cudnn(nullptr),
    m_name("none") {

  // Initialize neuron tensor dimensions
  m_num_neurons = 0;
  m_num_neuron_dims = 1;
  m_neuron_dims = std::vector<int>(1, 0);
  m_num_prev_neurons = 0;
  m_num_prev_neuron_dims = 1;
  m_neuron_dims = std::vector<int>(1, 0);

  // Default number of parent and child layers
  m_max_num_parent_layers = 1;
  m_max_num_child_layers = 1;

  // Initialize model
  m_neural_network_model = nullptr;

  // Initialize GPU information
  m_using_gpus = false;
#ifdef __LIB_CUDNN
  m_mini_batch_size_per_gpu = 0;
  m_max_mini_batch_size_per_gpu = 0;
  m_copy_fp_input_to_gpus = false;
  m_copy_fp_output_from_gpus = false;
  m_copy_bp_input_to_gpus = false;
  m_copy_bp_output_from_gpus = false;
  m_prev_neurons_cudnn_desc = nullptr;
  m_neurons_cudnn_desc = nullptr;
#endif // __LIB_CUDNN

  reset_counters();
}

Layer::Layer(const Layer& other) :
  m_index(other.m_index),
  m_comm(other.m_comm),
  m_num_neurons(other.m_num_neurons),
  m_num_neuron_dims(other.m_num_neuron_dims),
  m_neuron_dims(other.m_neuron_dims),
  m_num_prev_neurons(other.m_num_prev_neurons),
  m_num_prev_neuron_dims(other.m_num_prev_neuron_dims),
  m_prev_neuron_dims(other.m_prev_neuron_dims),
  m_parent_layers(other.m_parent_layers),
  m_child_layers(other.m_child_layers),
  m_max_num_parent_layers(other.m_max_num_parent_layers),
  m_max_num_child_layers(other.m_max_num_child_layers),
  m_execution_mode(other.m_execution_mode),
  m_neural_network_model(other.m_neural_network_model),
  m_using_gpus(other.m_using_gpus),
  m_cudnn(other.m_cudnn)
#ifdef __LIB_CUDNN
  ,
  m_mini_batch_size_per_gpu(other.m_mini_batch_size_per_gpu),
  m_max_mini_batch_size_per_gpu(other.m_max_mini_batch_size_per_gpu),
  m_copy_fp_input_to_gpus(other.m_copy_fp_input_to_gpus),
  m_copy_fp_output_from_gpus(other.m_copy_fp_output_from_gpus),
  m_copy_bp_input_to_gpus(other.m_copy_bp_input_to_gpus),
  m_copy_bp_output_from_gpus(other.m_copy_bp_output_from_gpus)
#endif // __LIB_CUDNN
{
  fp_time = other.fp_time;
  fp_compute_time = other.fp_compute_time;
  bp_time = other.bp_time;
  bp_compute_time = other.bp_compute_time;
  update_time = other.update_time;
  m_prev_activations = other.m_prev_activations->Copy();
  m_activations = other.m_activations->Copy();
  m_activations_v = other.m_activations_v->Copy();
  m_prev_error_signal = other.m_prev_error_signal->Copy();
  m_error_signal = other.m_error_signal->Copy();
  m_error_signal_v = other.m_error_signal_v->Copy();
#ifdef __LIB_CUDNN
  m_prev_activations_d = m_cudnn->copy(other.m_prev_activations_d,
                                       m_num_prev_neurons,
                                       m_mini_batch_size_per_gpu);
  m_activations_d = m_cudnn->copy(other.m_activations_d,
                                  m_num_neurons,
                                  m_mini_batch_size_per_gpu);
  m_prev_error_signal_d = m_cudnn->copy(other.m_prev_error_signal_d,
                                        m_num_neurons,
                                        m_mini_batch_size_per_gpu);
  m_error_signal_d = m_cudnn->copy(other.m_error_signal_d,
                                   m_num_prev_neurons,
                                   m_mini_batch_size_per_gpu);
  m_prev_neurons_cudnn_desc = nullptr;
  m_neurons_cudnn_desc = nullptr;
  cudnn::copy_tensor_cudnn_desc(other.m_prev_neurons_cudnn_desc,
                                m_prev_neurons_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_neurons_cudnn_desc,
                                m_neurons_cudnn_desc);
  m_name = other.m_name;
#endif // __LIB_CUDNN
}

Layer& Layer::operator=(const Layer& other) {
  m_index = other.m_index;
  m_comm = other.m_comm;
  m_num_neurons = other.m_num_neurons;
  m_num_neuron_dims = other.m_num_neuron_dims;
  m_neuron_dims = other.m_neuron_dims;
  m_num_prev_neurons = other.m_num_prev_neurons;
  m_num_prev_neuron_dims = other.m_num_prev_neuron_dims;
  m_prev_neuron_dims = other.m_prev_neuron_dims;
  m_parent_layers = other.m_parent_layers;
  m_child_layers = other.m_child_layers;
  m_max_num_parent_layers = other.m_max_num_parent_layers;
  m_max_num_child_layers = other.m_max_num_child_layers;
  m_execution_mode = other.m_execution_mode;
  m_neural_network_model = other.m_neural_network_model;
  m_using_gpus = other.m_using_gpus;
  m_cudnn = other.m_cudnn;
  fp_time = other.fp_time;
  fp_compute_time = other.fp_compute_time;
  bp_time = other.bp_time;
  bp_compute_time = other.bp_compute_time;
  update_time = other.update_time;
#ifdef __LIB_CUDNN
  m_mini_batch_size_per_gpu = other.m_mini_batch_size_per_gpu;
  m_max_mini_batch_size_per_gpu = other.m_max_mini_batch_size_per_gpu;
  m_copy_fp_input_to_gpus = other.m_copy_fp_input_to_gpus;
  m_copy_fp_output_from_gpus = other.m_copy_fp_output_from_gpus;
  m_copy_bp_input_to_gpus = other.m_copy_bp_input_to_gpus;
  m_copy_bp_output_from_gpus = other.m_copy_bp_output_from_gpus;
#endif // __LIB_CUDNN

  // Free allocated memory and copy data from other matrix
#define COPY_MATRIX(src, dst)                   \
  do {                                          \
    if(src != nullptr && dst != nullptr) {      \
      El::Copy(*src, *dst);                     \
    }                                           \
    if(src != nullptr && dst == nullptr) {      \
      dst = src->Copy();                        \
    }                                           \
    if(src == nullptr && dst != nullptr) {      \
      delete dst;                               \
      dst = nullptr;                            \
    }                                           \
  } while(false)
  COPY_MATRIX(other.m_prev_activations, m_prev_activations);
  COPY_MATRIX(other.m_activations, m_activations);
  COPY_MATRIX(other.m_prev_error_signal, m_prev_error_signal);
  COPY_MATRIX(other.m_error_signal, m_error_signal);
  COPY_MATRIX(other.m_activations_v, m_activations_v);
  COPY_MATRIX(other.m_error_signal_v, m_error_signal_v);
#undef COPY_MATRIX

#ifdef __LIB_CUDNN
  m_cudnn->deallocate_on_gpus(m_prev_activations_d);
  m_cudnn->deallocate_on_gpus(m_activations_d);
  m_cudnn->deallocate_on_gpus(m_prev_error_signal_d);
  m_cudnn->deallocate_on_gpus(m_error_signal_d);
  m_prev_activations_d = m_cudnn->copy(other.m_prev_activations_d,
                                       m_num_prev_neurons,
                                       m_mini_batch_size_per_gpu);
  m_activations_d = m_cudnn->copy(other.m_activations_d,
                                  m_num_neurons,
                                  m_mini_batch_size_per_gpu);
  m_prev_error_signal_d = m_cudnn->copy(other.m_prev_error_signal_d,
                                        m_num_neurons,
                                        m_mini_batch_size_per_gpu);
  m_error_signal_d = m_cudnn->copy(other.m_error_signal_d,
                                   m_num_prev_neurons,
                                   m_mini_batch_size_per_gpu);
  cudnn::copy_tensor_cudnn_desc(other.m_prev_neurons_cudnn_desc,
                                m_prev_neurons_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_neurons_cudnn_desc,
                                m_neurons_cudnn_desc);
  m_name = other.m_name;
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
  if(m_cudnn) {
    m_cudnn->deallocate_on_gpus(m_activations_d);
    m_cudnn->deallocate_on_gpus(m_error_signal_d);
    m_cudnn->deallocate_on_gpus(m_prev_activations_d);
    m_cudnn->deallocate_on_gpus(m_prev_error_signal_d);
    m_cudnn->unpin_matrix(*m_prev_activations);
    m_cudnn->unpin_matrix(*m_activations);
    m_cudnn->unpin_matrix(*m_prev_error_signal);
    m_cudnn->unpin_matrix(*m_error_signal);
  }
#endif // __LIB_CUDNN
  if(m_prev_activations  != nullptr) delete m_prev_activations;
  if(m_activations       != nullptr) delete m_activations;
  if(m_prev_error_signal != nullptr) delete m_prev_error_signal;
  if(m_error_signal      != nullptr) delete m_error_signal;
  if(m_activations_v     != nullptr) delete m_activations_v;
  if(m_error_signal_v    != nullptr) delete m_error_signal_v;
}

void Layer::forward_prop() {
  double fp_start = get_time();

  // Set matrix views based on current mini-batch size
  fp_set_std_matrix_view();

  // Get incoming activations and convert matrix distribution if necessary
  if(!m_parent_layers.empty()) {
    m_parent_layers.front()->get_fp_output(*m_prev_activations, this);
  } else {
    const int mini_batch_size = m_neural_network_model->get_current_mini_batch_size();
    El::Zeros(*m_prev_activations, m_num_prev_neurons, mini_batch_size);
  }

#ifdef __LIB_CUDNN
  if(m_using_gpus) {

    // Determine mini-batch size per GPU
    const int num_gpus = m_cudnn->get_num_gpus();
    const int local_mini_batch_size = m_activations_v->LocalWidth();
    m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

    // Set tensor descriptors
    cudnn::set_tensor_cudnn_desc(m_prev_neurons_cudnn_desc,
                                 m_mini_batch_size_per_gpu,
                                 m_prev_neuron_dims);
    cudnn::set_tensor_cudnn_desc(m_neurons_cudnn_desc,
                                 m_mini_batch_size_per_gpu,
                                 m_neuron_dims);

    // Transfer inputs from CPU to GPUs if needed
    if(m_copy_fp_input_to_gpus) {
      m_cudnn->scatter_to_gpus(m_prev_activations_d,
                               m_prev_activations->LockedMatrix(),
                               m_mini_batch_size_per_gpu);
    } else {
      if(!m_parent_layers.empty()) {
        m_parent_layers.front()->get_gpu_fp_output(m_prev_activations_d, this);
      }
    }

  }
#endif // __LIB_CUDNN

  // Apply layer's compute function
  double fp_compute_start = get_time();
  fp_compute();
  fp_compute_time += get_time() - fp_compute_start;

#ifdef __LIB_CUDNN
  // Transfer outputs from GPUs to CPU if needed
  if(m_using_gpus && m_copy_fp_output_from_gpus) {
    m_cudnn->gather_from_gpus(m_activations_v->Matrix(),
                              m_activations_d,
                              m_mini_batch_size_per_gpu);
    m_cudnn->synchronize();
  }
#endif // __LIB_CUDNN

  fp_time += get_time() - fp_start;
}

void Layer::back_prop() {
  double bp_start = get_time();

  // Set matrix views based on current mini-batch size
  bp_set_std_matrix_view();

  // Get incoming error signal and convert matrix distribution if necessary
  if(!m_child_layers.empty()) {
    m_child_layers.front()->get_bp_output(*m_prev_error_signal, this);
  } else {
    const int mini_batch_size = m_neural_network_model->get_current_mini_batch_size();
    El::Zeros(*m_prev_error_signal, m_num_neurons, mini_batch_size);
  }

#ifdef __LIB_CUDNN
  if(m_using_gpus) {

    // Determine mini-batch size per GPU
    const int num_gpus = m_cudnn->get_num_gpus();
    const int local_mini_batch_size = m_activations_v->LocalWidth();
    m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

    // Set tensor descriptors
    cudnn::set_tensor_cudnn_desc(m_prev_neurons_cudnn_desc,
                                 m_mini_batch_size_per_gpu,
                                 m_prev_neuron_dims);
    cudnn::set_tensor_cudnn_desc(m_neurons_cudnn_desc,
                                 m_mini_batch_size_per_gpu,
                                 m_neuron_dims);

    // Transfer inputs from CPU to GPUs if needed
    if(m_copy_bp_input_to_gpus) {
      m_cudnn->scatter_to_gpus(m_prev_error_signal_d,
                               m_prev_error_signal->LockedMatrix(),
                               m_mini_batch_size_per_gpu);
    } else {
      if(!m_child_layers.empty()) {
        m_child_layers.front()->get_gpu_bp_output(m_prev_error_signal_d, this);
      }
    }

  }
#endif // __LIB_CUDNN

  // Backprop the compute function.
  double bp_compute_start = get_time();
  bp_compute();
  bp_compute_time += get_time() - bp_compute_start;

#ifdef __LIB_CUDNN
  // Transfer outputs from GPUs to CPU if needed
  if(m_using_gpus && m_copy_bp_output_from_gpus) {
    m_cudnn->gather_from_gpus(m_error_signal_v->Matrix(),
                              m_error_signal_d,
                              m_mini_batch_size_per_gpu);
    m_cudnn->synchronize();
  }
#endif // __LIB_CUDNN

  bp_time += get_time() - bp_start;
}

bool Layer::update() {
  bool layer_done = false;
  // Apply any updates.
  double update_compute_start = get_time();
  layer_done = update_compute();
  update_time += get_time() - update_compute_start;
  return layer_done;
}

void Layer::summarize_stats(lbann_summary& summarizer, int step) {
  std::string prefix = "layer" + std::to_string(static_cast<long long>(m_index)) + "/";
  summarizer.reduce_scalar(prefix + "fp_time", fp_time, step);
  summarizer.reduce_scalar(prefix + "bp_time", bp_time, step);
  summarizer.reduce_scalar(prefix + "update_time", update_time, step);
  reset_counters();
}

void Layer::summarize_matrices(lbann_summary& summarizer, int step) {
  std::string prefix = "layer" + std::to_string(static_cast<long long>(m_index)) +
    "/activations/";
  summarizer.reduce_mean(prefix + "mean", *m_activations_v, step);
  summarizer.reduce_min(prefix + "min", *m_activations_v, step);
  summarizer.reduce_max(prefix + "max", *m_activations_v, step);
  summarizer.reduce_stdev(prefix + "stdev", *m_activations_v, step);
  prefix = "layer" + std::to_string(static_cast<long long>(m_index)) +
    "/error_signal/";
  summarizer.reduce_mean(prefix + "mean", *m_error_signal_v, step);
  summarizer.reduce_min(prefix + "min", *m_error_signal_v, step);
  summarizer.reduce_max(prefix + "max", *m_error_signal_v, step);
  summarizer.reduce_stdev(prefix + "stdev", *m_error_signal_v, step);
}

void Layer::setup() {
  setup_pointers();
  setup_dims();
  setup_data();
  setup_views();
  if (m_using_gpus) {
    setup_gpu();
  }
}

void Layer::setup_pointers() {
  // Check if the number of parents/children are valid
  if(m_max_num_parent_layers >= 0
     && (int)m_parent_layers.size() > m_max_num_parent_layers) {
    throw lbann_exception(
      std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
      "Layer: too many parent layers");
  }
  if(m_max_num_child_layers >= 0
     && (int)m_child_layers.size() > m_max_num_child_layers) {
    throw lbann_exception(
      std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
      "Layer: too many child layers");
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

void Layer::setup_data() {

  // Initialize matrices
  const int max_mini_batch_size = m_neural_network_model->get_max_mini_batch_size();
  if(max_mini_batch_size <= 0) {
    throw lbann_exception(
      std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
      "Layer: max mini-batch size is invalid");
  }
  if(m_num_prev_neurons > 0) {
    El::Zeros(*m_error_signal, m_num_prev_neurons, max_mini_batch_size);
  }
  if(m_num_neurons > 0) {
    El::Zeros(*m_activations, m_num_neurons, max_mini_batch_size);
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

  // Throw error is data layout is not data parallel
  // TODO: Design a more general interface
  if(get_data_layout() != data_layout::DATA_PARALLEL) {
    throw lbann_exception(
      std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
      "Layer: GPUs are currently only supported for data parallel layers");
  }

  // Determine whether to transfer data between CPU and GPUs
  if(m_parent_layers.empty() || !m_parent_layers.front()->using_gpus()) {
    m_copy_fp_input_to_gpus = true;
    m_copy_bp_output_from_gpus = true;
  }
  if(m_child_layers.empty() || !m_child_layers.front()->using_gpus()) {
    m_copy_fp_output_from_gpus = true;
    m_copy_bp_input_to_gpus = true;
  }
  
  // Split mini-batch amongst GPUs
  const int num_gpus = m_cudnn->get_num_gpus();
  const int num_processes = m_comm->get_procs_per_model();
  const int local_mini_batch_size =
    (m_neural_network_model->get_max_mini_batch_size() + num_processes - 1) /
    num_processes;
  m_max_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;
  m_mini_batch_size_per_gpu = m_max_mini_batch_size_per_gpu;

  // Set tensor descriptors
  cudnn::set_tensor_cudnn_desc(m_prev_neurons_cudnn_desc,
                               m_mini_batch_size_per_gpu,
                               m_prev_neuron_dims);
  cudnn::set_tensor_cudnn_desc(m_neurons_cudnn_desc,
                               m_mini_batch_size_per_gpu,
                               m_neuron_dims);

  // Allocate GPU memory
  m_cudnn->allocate_on_gpus(m_activations_d,
                            m_num_neurons,
                            m_max_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_error_signal_d,
                            m_num_prev_neurons,
                            m_max_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_prev_activations_d,
                            m_num_prev_neurons,
                            m_max_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_prev_error_signal_d,
                            m_num_neurons,
                            m_max_mini_batch_size_per_gpu);

#endif // __LIB_CUDNN
}

void Layer::check_setup() {}

bool Layer::saveToCheckpoint(int fd, const char *filename, size_t *bytes) {
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

bool Layer::saveToCheckpointShared(persist& p) {
  return true;
}

bool Layer::loadFromCheckpointShared(persist& p) {
  return true;
}

void Layer::fp_set_std_matrix_view() {
  int mini_batch_size = m_neural_network_model->get_current_mini_batch_size();
  El::View(*m_activations_v, *m_activations, El::ALL, El::IR(0, mini_batch_size));
}

void Layer::bp_set_std_matrix_view() {
  int mini_batch_size = m_neural_network_model->get_current_mini_batch_size();
  El::View(*m_activations_v, *m_activations, El::ALL, El::IR(0, mini_batch_size));
  El::View(*m_error_signal_v, *m_error_signal, El::ALL, El::IR(0, mini_batch_size));
}

#ifdef __LIB_CUDNN
void Layer::pin_data() {

  // Get maximum mini-batch size
  const int max_mini_batch_size = m_neural_network_model->get_max_mini_batch_size();

  // Flags to determine whether to pin memory
  bool pin_fp_input = false;
  bool pin_fp_output = false;
  bool pin_bp_input = false;
  bool pin_bp_output = false;

  // Pin fp input if there is no input layer and this layer uses GPUs
  if(m_parent_layers.empty() && m_using_gpus) {
    pin_fp_input = true;
  }
    
  // Pin fp input if input layer does not use GPUs, this layer uses
  // GPUs, and input layer has different distribution
  if(!m_parent_layers.empty()
     && !m_parent_layers.front()->using_gpus()
     && m_using_gpus) {
    if(m_parent_layers.front()->m_activations->DistData()
       != m_prev_activations->DistData()) {
      pin_fp_input = true;
    }
  }

  // Pin fp output if this layer does not use GPUs, output layer uses
  // GPUs, and output layer has same distribution
  if(!m_using_gpus
     && !m_parent_layers.empty()
     && m_parent_layers.front()->using_gpus()) {
    if(m_parent_layers.front()->m_prev_activations->DistData()
       != m_activations->DistData()) {
      pin_fp_output = true;
    }
  }

  // Pin fp output if this layer uses GPUs and output layer does not
  // use GPUs
  if(m_using_gpus
     && !m_child_layers.empty()
     && !m_child_layers.front()->using_gpus()) {
    pin_fp_output = true;
  }

  // Pin fp output if this layer uses GPUs and there is no output layer
  if(m_using_gpus && m_child_layers.empty()) {
    pin_fp_output = true;
  }

  // Pin bp input if there is no input layer and this layer uses GPUs
  if(m_child_layers.empty() && m_using_gpus) {
    pin_bp_input = true;
  }
    
  // Pin bp input if input layer does not use GPUs, this layer uses
  // GPUs, and input layer has different distribution
  if(!m_child_layers.empty()
     && !m_child_layers.front()->using_gpus()
     && m_using_gpus) {
    if(m_child_layers.front()->m_error_signal->DistData()
       != m_prev_error_signal->DistData()) {
      pin_bp_input = true;
    }
  }

  // Pin bp output if this layer does not use GPUs, output layer uses
  // GPUs, and output layer has same distribution
  if(!m_using_gpus
     && !m_parent_layers.empty()
     && m_parent_layers.front()->using_gpus()) {
    if(m_parent_layers.front()->m_prev_error_signal->DistData()
       != m_error_signal->DistData()) {
      pin_bp_output = true;
    }
  }

  // Pin bp output if this layer uses GPUs and output layer does not
  // use GPUs
  if(m_using_gpus
     && !m_parent_layers.empty()
     && !m_parent_layers.front()->using_gpus()) {
    pin_bp_output = true;
  }

  // Pin bp output if this layer uses GPUs and there is no output layer
  if(m_using_gpus && m_parent_layers.empty()) {
    pin_bp_output = true;
  }

  // Pin host memory if needed for GPU memory transfers
  if(pin_fp_input) {
    /// @todo If m_prev_activations is resized, we might leak pinned memory
    // m_prev_activations->Resize(m_num_prev_neurons, max_mini_batch_size);
    // m_cudnn->pin_matrix(*m_prev_activations);
  }
  if(pin_fp_output) {
    m_activations->Resize(m_num_neurons, max_mini_batch_size);
    m_cudnn->pin_matrix(*m_activations);
  }
  if(pin_bp_input) {
    /// @todo If m_prev_error_signal is resized, we might leak pinned memory
    // m_prev_error_signal->Resize(m_num_neurons, max_mini_batch_size);
    // m_cudnn->pin_matrix(*m_prev_error_signal);
  }
  if(pin_bp_output) {
    m_error_signal->Resize(m_num_prev_neurons, max_mini_batch_size);
    m_cudnn->pin_matrix(*m_error_signal);
  }

}

#endif // __LIB_CUDNN

void Layer::get_fp_output(AbsDistMat& fp_output, const Layer* next_layer) const {
  if(m_activations_v->DistData() == fp_output.DistData()) {
    El::LockedView(fp_output, *m_activations_v);
  }
  else {
    El::Copy(*m_activations_v, fp_output);
  }
}

void Layer::get_bp_output(AbsDistMat& bp_output, const Layer* prev_layer) const {
  if(m_error_signal_v->DistData() == bp_output.DistData()) {
    El::LockedView(bp_output, *m_error_signal_v);
  }
  else {
    El::Copy(*m_error_signal_v, bp_output);
  }
}

#ifdef __LIB_CUDNN

void Layer::get_gpu_fp_output(std::vector<DataType*>& fp_output, const Layer* next_layer) const {
  this->m_cudnn->copy_on_gpus(fp_output,
                              m_activations_d,
                              m_num_neurons,
                              m_mini_batch_size_per_gpu);
}

void Layer::get_gpu_bp_output(std::vector<DataType*>& bp_output, const Layer* prev_layer) const {
  this->m_cudnn->copy_on_gpus(bp_output,
                              m_error_signal_d,
                              m_num_prev_neurons,
                              m_mini_batch_size_per_gpu);
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

const vector<int> Layer::fp_output_dims(const Layer* next_layer) const {
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

}  // namespace lbann
