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

/// Matrices should be in MC,MR distributions
template <>
void Layer::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>() {
  m_prev_activations    = new DistMat(m_comm->get_model_grid());
  m_activations         = new DistMat(m_comm->get_model_grid());
  m_prev_error_signal   = new DistMat(m_comm->get_model_grid());
  m_error_signal        = new DistMat(m_comm->get_model_grid());

  /// Instantiate these view objects but do not allocate data for them
  m_prev_activations_v  = new DistMat(m_comm->get_model_grid());
  m_activations_v       = new DistMat(m_comm->get_model_grid());
  m_prev_error_signal_v = new DistMat(m_comm->get_model_grid());
  m_error_signal_v      = new DistMat(m_comm->get_model_grid());
}

/// Weight matrices should be in Star,Star and data matrices Star,VC distributions
template<>
void Layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>() {
  m_prev_activations    = new StarVCMat(m_comm->get_model_grid());
  m_activations         = new StarVCMat(m_comm->get_model_grid());
  m_prev_error_signal   = new StarVCMat(m_comm->get_model_grid());
  m_error_signal        = new StarVCMat(m_comm->get_model_grid());

  /// Instantiate these view objects but do not allocate data for them
  m_prev_activations_v  = new StarVCMat(m_comm->get_model_grid());
  m_activations_v       = new StarVCMat(m_comm->get_model_grid());
  m_prev_error_signal_v = new StarVCMat(m_comm->get_model_grid());
  m_error_signal_v      = new StarVCMat(m_comm->get_model_grid());
}

Layer::Layer(const int index, lbann_comm *comm, int mbsize)
  : m_index(index),
    m_comm(comm),
    m_execution_mode(execution_mode::training),
    m_cudnn(nullptr),
    m_mini_batch_size(mbsize),
    m_effective_mbsize(mbsize)
{
  // Initialize neuron tensor dimensions
  m_num_neurons = 0;
  m_num_neuron_dims = 1;
  m_neuron_dims = std::vector<int>(1, 0);
  m_num_prev_neurons = 0;
  m_num_prev_neuron_dims = 1;
  m_neuron_dims = std::vector<int>(1, 0);

  // Initialize model
  m_neural_network_model = NULL;

  // Initialize GPU information
  m_using_gpus = false;
#ifdef __LIB_CUDNN
  m_fp_input_pinned = false;
  m_fp_output_pinned = false;
  m_bp_input_pinned = false;
  m_bp_output_pinned = false;

  m_prev_neurons_cudnn_desc = NULL;  
  m_neurons_cudnn_desc = NULL;  
#endif

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
  m_execution_mode(other.m_execution_mode),
  m_neural_network_model(other.m_neural_network_model),
  m_prev_layer(other.m_prev_layer),
  m_next_layer(other.m_next_layer),
  m_using_gpus(other.m_using_gpus),
  m_cudnn(other.m_cudnn),
  m_mini_batch_size(other.m_mini_batch_size),
  m_effective_mbsize(other.m_effective_mbsize),
  fp_time(other.fp_time),
  fp_compute_time(other.fp_compute_time),
  bp_time(other.bp_time),
  bp_compute_time(other.bp_compute_time),
  update_time(other.update_time) {
  // No cuDNN support yet.
#ifdef __LIB_CUDNN
  throw lbann_exception("cannot copy layers with cuDNN enabled");
#endif
  m_prev_error_signal = other.m_prev_error_signal->Copy();
  m_error_signal = other.m_error_signal->Copy();
  m_prev_error_signal_v = other.m_prev_error_signal_v->Copy();
  m_error_signal_v = other.m_error_signal_v->Copy();
  m_activations = other.m_activations->Copy();
  m_prev_activations = other.m_prev_activations->Copy();
  m_activations_v = other.m_activations_v->Copy();
  m_prev_activations_v = other.m_prev_activations_v->Copy();
}

Layer& Layer::operator=(const Layer& other) {
  // No cuDNN support yet.
#ifdef __LIB_CUDNN
  throw lbann_exception("cannot copy layers with cuDNN enabled");
#endif
  m_index = other.m_index;
  m_comm = other.m_comm;
  m_num_neurons = other.m_num_neurons;
  m_num_neuron_dims = other.m_num_neuron_dims;
  m_neuron_dims = other.m_neuron_dims;
  m_num_prev_neurons = other.m_num_prev_neurons;
  m_num_prev_neuron_dims = other.m_num_prev_neuron_dims;
  m_prev_neuron_dims = other.m_prev_neuron_dims;
  m_execution_mode = other.m_execution_mode;
  m_neural_network_model = other.m_neural_network_model;
  m_prev_layer = other.m_prev_layer;
  m_next_layer = other.m_next_layer;
  m_using_gpus = other.m_using_gpus;
  m_cudnn = other.m_cudnn;
  m_mini_batch_size = other.m_mini_batch_size;
  m_effective_mbsize = other.m_effective_mbsize;
  fp_time = other.fp_time;
  fp_compute_time = other.fp_compute_time;
  bp_time = other.bp_time;
  bp_compute_time = other.bp_compute_time;
  update_time = other.update_time;
  // Free allocated memory.
  if (m_prev_error_signal) {
    delete m_prev_error_signal;
    delete m_error_signal;
    delete m_activations;
    delete m_prev_activations;
    delete m_prev_error_signal_v;
    delete m_error_signal_v;
    delete m_activations_v;
    delete m_prev_activations_v;
  }
  m_prev_error_signal = other.m_prev_error_signal->Copy();
  m_error_signal = other.m_error_signal->Copy();
  m_prev_error_signal_v = other.m_prev_error_signal_v->Copy();
  m_error_signal_v = other.m_error_signal_v->Copy();
  m_activations = other.m_activations->Copy();
  m_prev_activations = other.m_prev_activations->Copy();
  m_activations_v = other.m_activations_v->Copy();
  m_prev_activations_v = other.m_prev_activations_v->Copy();
  return *this;
}

Layer::~Layer() {
#ifdef __LIB_CUDNN
  if(m_prev_neurons_cudnn_desc) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_prev_neurons_cudnn_desc));
  }
  if(m_neurons_cudnn_desc) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_neurons_cudnn_desc));
  }
  if(m_cudnn) {
    m_cudnn->deallocate_on_gpus(m_activations_d);
    m_cudnn->deallocate_on_gpus(m_error_signal_d);
    m_cudnn->deallocate_on_gpus(m_prev_activations_d);
    m_cudnn->deallocate_on_gpus(m_prev_error_signal_d);
    if(m_fp_input_pinned) {
      m_cudnn->unpin_matrix(*m_prev_activations);
    }
    if(m_fp_output_pinned) {
      m_cudnn->unpin_matrix(*m_activations);
    }
    if(m_bp_input_pinned) {
      m_cudnn->unpin_matrix(*m_prev_error_signal);
    }
    if(m_bp_output_pinned) {
      m_cudnn->unpin_matrix(*m_error_signal);
    }
  }
#endif
  delete m_prev_error_signal;
  delete m_error_signal;
  delete m_activations;
  delete m_prev_activations;
  delete m_prev_error_signal_v;
  delete m_error_signal_v;
  delete m_activations_v;
  delete m_prev_activations_v;
}

void Layer::forwardProp() {
  double fp_start = get_time();

  // Get incoming activations and convert matrix distribution if necessary
  if(m_prev_layer != NULL) {
    const DistData& prev_dist = m_prev_layer->m_activations->DistData();
    const DistData& curr_dist = m_prev_activations->DistData();
    if(prev_dist.colDist == curr_dist.colDist
       && prev_dist.rowDist == curr_dist.rowDist) {
      View(*m_prev_activations, *m_prev_layer->m_activations);
    } else {
      Copy(*m_prev_layer->m_activations, *m_prev_activations);
    }
  }

  // Set matrix views based on current mini-batch size
  fp_set_std_matrix_view();

#ifdef __LIB_CUDNN
  // Transfer inputs from CPU to GPUs if needed
  if(m_using_gpus) {
    if(m_prev_layer == NULL || !m_prev_layer->m_using_gpus) {
      m_cudnn->scatter_to_gpus(m_prev_activations_d,
                               m_prev_activations_v->LockedMatrix(),
                               m_mini_batch_size_per_gpu);
    } else {
      m_prev_activations_d = m_prev_layer->m_activations_d;
    }
  }
#endif

  // Apply layer's compute function
  double fp_compute_start = get_time();
  fp_compute();
  fp_compute_time += get_time() - fp_compute_start;

#ifdef __LIB_CUDNN
  // Transfer outputs from GPUs to CPU if needed
  if(m_using_gpus) {
    if(m_next_layer == NULL || !m_next_layer->m_using_gpus) {
      m_cudnn->gather_from_gpus(m_activations_v->Matrix(),
                                m_activations_d,
                                m_mini_batch_size_per_gpu);
      m_cudnn->synchronize();
    }
  }
#endif

  fp_time += get_time() - fp_start;
}

void Layer::backProp() {
  double bp_start = get_time();

  // Get incoming error signal and convert matrix distribution if necessary
  if(m_next_layer != NULL) {
    const DistData& prev_dist = m_next_layer->m_error_signal->DistData();
    const DistData& curr_dist = m_prev_error_signal->DistData();
    if(prev_dist.colDist == curr_dist.colDist
       && prev_dist.rowDist == curr_dist.rowDist) {
      View(*m_prev_error_signal, *m_next_layer->m_error_signal);
    } else {
      Copy(*m_next_layer->m_error_signal, *m_prev_error_signal);
    }
  }

  // Set the view for all of the standard matrices based on the
  // current mini-batch size
  bp_set_std_matrix_view();

#ifdef __LIB_CUDNN
  // Transfer inputs from CPU to GPUs if needed
  if(m_using_gpus) {
    if(m_next_layer == NULL || !m_next_layer->m_using_gpus) {
      m_cudnn->scatter_to_gpus(m_prev_error_signal_d,
                               m_prev_error_signal_v->LockedMatrix(),
                               m_mini_batch_size_per_gpu);
    } else {
      m_prev_error_signal_d = m_next_layer->m_error_signal_d;
    }
  }
#endif

  // Backprop the compute function.
  double bp_compute_start = get_time();
  bp_compute();
  bp_compute_time += get_time() - bp_compute_start;

#ifdef __LIB_CUDNN
  // Transfer outputs from GPUs to CPU if needed
  if(m_using_gpus) {
    if(m_prev_layer == NULL || !m_prev_layer->m_using_gpus) {
      m_cudnn->gather_from_gpus(m_error_signal_v->Matrix(),
                                m_error_signal_d,
                                m_mini_batch_size_per_gpu);
      m_cudnn->synchronize();
    }
  }
#endif

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
  summarizer.reduce_mean(prefix + "mean", *m_activations, step);
  summarizer.reduce_min(prefix + "min", *m_activations, step);
  summarizer.reduce_max(prefix + "max", *m_activations, step);
  summarizer.reduce_stdev(prefix + "stdev", *m_activations, step);
  prefix = "layer" + std::to_string(static_cast<long long>(m_index)) +
    "/error_signal/";
  summarizer.reduce_mean(prefix + "mean", *m_error_signal, step);
  summarizer.reduce_min(prefix + "min", *m_error_signal, step);
  summarizer.reduce_max(prefix + "max", *m_error_signal, step);
  summarizer.reduce_stdev(prefix + "stdev", *m_error_signal, step);
}

void Layer::setup(const Layer *prev_layer, const Layer *next_layer) {
  setup_pointers(prev_layer, next_layer);

  setup_dims();
  setup_data();
  if (m_using_gpus) {
    setup_gpu();
  }
}

void Layer::setup_pointers(const Layer *prev_layer, const Layer *next_layer) {
  // Set adjacent layers
  m_prev_layer = prev_layer;
  m_next_layer = next_layer;
}

void Layer::setup_dims() {
  // Get dimensions of previous neuron tensor
  if(m_prev_layer != NULL) {
    m_num_prev_neurons = m_prev_layer->m_num_neurons;
    m_num_prev_neuron_dims = m_prev_layer->m_num_neuron_dims;
    m_prev_neuron_dims = m_prev_layer->m_neuron_dims;
  } else {
    m_num_prev_neurons = 0;
    m_num_prev_neuron_dims = 0;
    m_prev_neuron_dims.assign(1, 0);
  }
}

void Layer::setup_data() {
  // Initialize matrices
  if (m_num_neurons == 0) {
    throw lbann_exception("lbann_layer: " + std::to_string(m_index) +
                          " num_neurons is 0");
  }
  if (m_mini_batch_size == 0) {
    throw lbann_exception("lbann_layer: " + std::to_string(m_index) +
                          " mini_batch_size is 0");
  }
  if (m_num_prev_neurons > 0) {
    m_error_signal->Resize(m_num_prev_neurons, m_mini_batch_size);
  }
  if (m_num_neurons > 0) {
    m_activations->Resize(m_num_neurons, m_mini_batch_size);
  }

#ifdef __LIB_CUDNN
  // Pin host memory if needed for GPU memory transfers
  pin_data();
#endif

}

void Layer::setup_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("Layer: cuDNN not detected");
#else

  // Throw error is data layout is not data parallel
  // TODO: Design a more general interface
  if(get_data_layout() != data_layout::DATA_PARALLEL) {
    throw lbann_exception("Layer: GPUs are currently only supported for data parallel layers");
  }
  
  // Split mini-batch amongst GPUs
  const int num_gpus = m_cudnn->get_num_gpus();
  const int num_processes = m_comm->get_procs_per_model();
  const int local_mini_batch_size = (m_mini_batch_size + num_processes - 1) / num_processes;
  m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

  // Initialize descriptors
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_prev_neurons_cudnn_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_neurons_cudnn_desc));

  // Set input tensor descriptor
  std::vector<int> input_dims = m_prev_neuron_dims;
  input_dims.insert(input_dims.begin(), m_mini_batch_size_per_gpu);
  std::vector<int> input_strides(input_dims.size());
  input_strides[input_strides.size()-1]  = 1;
  for(int i=input_strides.size()-2; i>=0; --i) {
    input_strides[i] = input_strides[i+1] * input_dims[i+1];
  }
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_prev_neurons_cudnn_desc,
                                         m_cudnn->get_cudnn_data_type(),
                                         input_dims.size(),
                                         input_dims.data(),
                                         input_strides.data()));

  // Set output tensor descriptor
  std::vector<int> output_dims = m_neuron_dims;
  output_dims.insert(output_dims.begin(), m_mini_batch_size_per_gpu);
  std::vector<int> output_strides(output_dims.size());
  output_strides[output_strides.size()-1]  = 1;
  for(int i=output_strides.size()-2; i>=0; --i) {
    output_strides[i] = output_strides[i+1] * output_dims[i+1];
  }
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_neurons_cudnn_desc,
                                         m_cudnn->get_cudnn_data_type(),
                                         output_dims.size(),
                                         output_dims.data(),
                                         output_strides.data()));

  // Allocate GPU memory
  m_cudnn->allocate_on_gpus(m_activations_d,
                            m_num_neurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_error_signal_d,
                            m_num_prev_neurons,
                            m_mini_batch_size_per_gpu);
  if(m_prev_layer == NULL || !m_prev_layer->using_gpus()) {
    m_cudnn->allocate_on_gpus(m_prev_activations_d,
                              m_num_prev_neurons,
                              m_mini_batch_size_per_gpu);
  }
  if(m_prev_layer == NULL || !m_next_layer->using_gpus()) {
    m_cudnn->allocate_on_gpus(m_prev_error_signal_d,
                              m_num_neurons,
                              m_mini_batch_size_per_gpu);
  }

#endif
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
  El::Int cur_mini_batch_size = m_neural_network_model->get_current_mini_batch_size();
  View(*m_prev_activations_v, *m_prev_activations, ALL, IR(0, cur_mini_batch_size));
  View(*m_activations_v, *m_activations, ALL, IR(0, cur_mini_batch_size));

  // Update the layer's effective mini-batch size so it averages properly.
  /// @todo BVE FIXME This will cause a bug when you are on the last
  /// iteration and the size of the current mini-batch equals the normal
  /// mini-batch size.  In this case one of the ranks gets out of sync
  /// To fix this, we need a flag for when we are on the last mini-batch
  if(cur_mini_batch_size != m_mini_batch_size || 1) {
    // When the current mini-batch is partial, check with the other
    // models to figure out the entire size of the complete mini-batch
    int total_mini_batch_size = m_comm->intermodel_allreduce((int) cur_mini_batch_size);
    set_effective_minibatch_size(total_mini_batch_size);
  } else {
    set_effective_minibatch_size(cur_mini_batch_size * m_comm->get_num_models());
  }
}

void Layer::bp_set_std_matrix_view() {
  El::Int cur_mini_batch_size = m_neural_network_model->get_current_mini_batch_size();
  View(*m_prev_activations_v, *m_prev_activations, ALL, IR(0, cur_mini_batch_size));
  View(*m_activations_v, *m_activations, ALL, IR(0, cur_mini_batch_size));
  if(m_prev_error_signal->Height() > 0) {
    View(*m_prev_error_signal_v, *m_prev_error_signal, ALL,
         IR(0, cur_mini_batch_size));
  }
  View(*m_error_signal_v, *m_error_signal, ALL, IR(0, cur_mini_batch_size));
}

#ifdef __LIB_CUDNN
void Layer::pin_data() {

  // Pin fp input on host memory if needed for GPU memory transfers
  if(!m_fp_input_pinned) {
    bool pin_fp_input = false;

    // Pin fp input if there is no input layer and this layer uses GPUs
    if(m_prev_layer == NULL
       && m_using_gpus) {
      pin_fp_input = true;
    }
    
    // Pin fp input if input layer does not use GPUs, this layer uses
    // GPUs, and input layer has different distribution
    if(m_prev_layer != NULL
       && !m_prev_layer->m_using_gpus
       && m_using_gpus) {
      const El::DistData& prev_dist = m_prev_layer->m_activations->DistData();
      const El::DistData& curr_dist = m_prev_activations->DistData();
      if(!(prev_dist.colDist == curr_dist.colDist
           && prev_dist.rowDist == curr_dist.rowDist)) {
        pin_fp_input = true;
      }
    }

    // Pin fp input if needed
    if(pin_fp_input) {
      m_prev_activations->Resize(m_num_prev_neurons, m_mini_batch_size);
      m_cudnn->pin_matrix(*m_prev_activations);
      m_fp_input_pinned = true;
    }

  }

  // Pin fp output on host memory if needed for GPU memory transfers
  if(!m_fp_output_pinned) {
    bool pin_fp_output = false;

    // Pin fp output if this layer does not use GPUs, output layer uses
    // GPUs, and output layer has same distribution
    if(!m_using_gpus
       && m_next_layer != NULL
       && m_next_layer->m_using_gpus) {
      const El::DistData& next_dist = m_next_layer->m_prev_activations->DistData();
      const El::DistData& curr_dist = m_activations->DistData();
      if(next_dist.colDist == curr_dist.colDist
         && next_dist.rowDist == curr_dist.rowDist) {
        pin_fp_output = true;
      }
    }

    // Pin fp output if this layer uses GPUs and output layer does not
    // use GPUs
    if(m_using_gpus
       && m_next_layer != NULL
       && !m_next_layer->m_using_gpus) {
      pin_fp_output = true;
    }

    // Pin fp output if this layer uses GPUs and there is no output layer
    if(m_using_gpus
       && m_next_layer == NULL) {
      pin_fp_output = true;
    }

    // Pin output if needed
    if(pin_fp_output) {
      m_activations->Resize(m_num_neurons, m_mini_batch_size);
      m_cudnn->pin_matrix(*m_activations);
      m_fp_output_pinned = true;
    }

  }

  // Pin bp input on host memory if needed for GPU memory transfers
  if(!m_bp_input_pinned) {
    bool pin_bp_input = false;

    // Pin bp input if there is no input layer and this layer uses GPUs
    if(m_next_layer == NULL
       && m_using_gpus) {
      pin_bp_input = true;
    }
    
    // Pin bp input if input layer does not use GPUs, this layer uses
    // GPUs, and input layer has different distribution
    if(m_next_layer != NULL
       && !m_next_layer->m_using_gpus
       && m_using_gpus) {
      const El::DistData& prev_dist = m_next_layer->m_error_signal->DistData();
      const El::DistData& curr_dist = m_prev_error_signal->DistData();
      if(!(prev_dist.colDist == curr_dist.colDist
           && prev_dist.rowDist == curr_dist.rowDist)) {
        pin_bp_input = true;
      }
    }

    // Pin bp input if needed
    if(pin_bp_input) {
      m_prev_error_signal->Resize(m_num_neurons, m_mini_batch_size);
      m_cudnn->pin_matrix(*m_prev_error_signal);
      m_bp_input_pinned = true;
    }

  }

  // Pin bp output on host memory if needed for GPU memory transfers
  if(!m_bp_output_pinned) {
    bool pin_bp_output = false;

    // Pin bp output if this layer does not use GPUs, output layer uses
    // GPUs, and output layer has same distribution
    if(!m_using_gpus
       && m_prev_layer != NULL
       && m_prev_layer->m_using_gpus) {
      const El::DistData& next_dist = m_prev_layer->m_prev_error_signal->DistData();
      const El::DistData& curr_dist = m_error_signal->DistData();
      if(next_dist.colDist == curr_dist.colDist
         && next_dist.rowDist == curr_dist.rowDist) {
        pin_bp_output = true;
      }
    }

    // Pin bp output if this layer uses GPUs and output layer does not
    // use GPUs
    if(m_using_gpus
       && m_prev_layer != NULL
       && !m_prev_layer->m_using_gpus) {
      pin_bp_output = true;
    }

    // Pin bp output if this layer uses GPUs and there is no output layer
    if(m_using_gpus
       && m_prev_layer == NULL) {
      pin_bp_output = true;
    }

    // Pin bp output if needed
    if(pin_bp_output) {
      m_error_signal->Resize(m_num_prev_neurons, m_mini_batch_size);
      m_cudnn->pin_matrix(*m_error_signal);
      m_bp_output_pinned = true;
    }

  }

}

#endif

}  // namespace lbann
