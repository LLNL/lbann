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

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/activations/create_activation.hpp"
#include "lbann/utils/lbann_timer.hpp"
#include "lbann/models/lbann_model.hpp"
#include "lbann/io/lbann_file_io.hpp"
#include "lbann/io/lbann_persist.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::Layer::Layer(data_layout data_dist, const uint index,
                    lbann_comm *comm, optimizer *opt,
                    uint mbsize)
  : m_data_layout(data_dist), m_index(index),
    m_comm(comm), m_optimizer(opt),
    m_type(layer_type::INVALID), m_prev_layer_type(layer_type::INVALID), m_next_layer_type(layer_type::INVALID),
    m_execution_mode(execution_mode::training),
    m_cudnn(nullptr),
    m_mini_batch_size(mbsize),
    m_effective_mbsize(mbsize)
{

  fp_input = NULL;
  bp_input = NULL;
  m_neural_network_model = NULL;

  m_using_gpus = false;
  m_prev_layer_using_gpus = false;
  m_next_layer_using_gpus = false;
#ifdef __LIB_CUDNN
  fp_input_d = NULL;
  bp_input_d = NULL;
#endif

  // Setup the data distribution
  switch(data_dist) {
  case data_layout::MODEL_PARALLEL:
    initialize_model_parallel_distribution();
    break;
  case data_layout::DATA_PARALLEL:
    initialize_data_parallel_distribution();
    break;
  default:
    throw lbann_exception(std::string{} + __FILE__ + " " +
                          std::to_string(__LINE__) +
                          "Invalid data layout selected");
  }

  reset_counters();

}

lbann::Layer::~Layer() {
  delete m_weights;
  delete m_weights_gradient;
  delete m_weighted_sum;
  delete m_prev_error_signal;
  delete m_error_signal;
  delete m_activations;
  delete m_prev_activations;
  delete m_weighted_sum_v;
  delete m_prev_error_signal_v;
  delete m_error_signal_v;
  delete m_activations_v;
  delete m_prev_activations_v;
}

/// Matrices should be in MC,MR distributions
void lbann::Layer::initialize_model_parallel_distribution() {
  m_weights             = new DistMat(m_comm->get_model_grid());
  m_weights_gradient    = new DistMat(m_comm->get_model_grid());
  m_weighted_sum        = new DistMat(m_comm->get_model_grid());
  m_prev_activations    = new DistMat(m_comm->get_model_grid());
  m_activations         = new DistMat(m_comm->get_model_grid());
  m_prev_error_signal   = new DistMat(m_comm->get_model_grid());
  m_error_signal        = new DistMat(m_comm->get_model_grid());

  /// Instantiate these view objects but do not allocate data for them
  m_weighted_sum_v      = new DistMat(m_comm->get_model_grid());
  m_prev_activations_v  = new DistMat(m_comm->get_model_grid());
  m_activations_v       = new DistMat(m_comm->get_model_grid());
  m_prev_error_signal_v = new DistMat(m_comm->get_model_grid());
  m_error_signal_v      = new DistMat(m_comm->get_model_grid());
}

/// Weight matrices should be in Star,Star and data matrices Star,VC distributions
void lbann::Layer::initialize_data_parallel_distribution() {
  m_weights             = new StarMat(m_comm->get_model_grid());
  m_weights_gradient    = new StarMat(m_comm->get_model_grid());
  m_weighted_sum        = new StarVCMat(m_comm->get_model_grid());
  m_prev_activations    = new StarVCMat(m_comm->get_model_grid());
  m_activations         = new StarVCMat(m_comm->get_model_grid());
  m_prev_error_signal   = new StarVCMat(m_comm->get_model_grid());
  m_error_signal        = new StarVCMat(m_comm->get_model_grid());

  /// Instantiate these view objects but do not allocate data for them
  m_weighted_sum_v      = new StarVCMat(m_comm->get_model_grid());
  m_prev_activations_v  = new StarVCMat(m_comm->get_model_grid());
  m_activations_v       = new StarVCMat(m_comm->get_model_grid());
  m_prev_error_signal_v = new StarVCMat(m_comm->get_model_grid());
  m_error_signal_v      = new StarVCMat(m_comm->get_model_grid());
}

void lbann::Layer::forwardProp() {
  double fp_start = get_time();

  // Get incoming activations and convert matrix distribution if necessary
  if(fp_input != NULL) { // Input layers will not have a valid fp_input
    DistData curr_dist = m_prev_activations->DistData();
    DistData prev_dist = fp_input->DistData();
    if(curr_dist.colDist == prev_dist.colDist
        && curr_dist.rowDist == prev_dist.rowDist) {
      View(*m_prev_activations, *fp_input);
    } else {
      *m_prev_activations = *fp_input;
    }
  }

  // Set matrix views based on current mini-batch size
  fp_set_std_matrix_view();

#ifdef __LIB_CUDNN
  // Transfer inputs from CPU to GPUs if needed
  if(m_using_gpus) {
    if(!m_prev_layer_using_gpus) {
      m_cudnn->scatter_to_gpus(m_prev_activations_d,
                               m_prev_activations_v->LockedMatrix(),
                               m_mini_batch_size_per_gpu);
    } else {
      m_prev_activations_d = *fp_input_d;
    }
  }
#endif

  // Apply layer's compute function
  double fp_compute_start = get_time();
  fp_compute();
  fp_compute_time += get_time() - fp_compute_start;

#ifdef __LIB_CUDNN
  // Transfer outputs from GPUs to CPU if needed
  if(m_using_gpus && !m_next_layer_using_gpus) {
    m_cudnn->gather_from_gpus(m_activations_v->Matrix(),
                              m_activations_d,
                              m_mini_batch_size_per_gpu);
    m_cudnn->synchronize();
  }
#endif

  fp_time += get_time() - fp_start;
}

void lbann::Layer::backProp() {
  double bp_start = get_time();

  // Set the view for all of the standard matrices based on the
  // current mini-batch size
  //  bp_set_std_matrix_view();

  // Get incoming loss and convert matrix distribution if necessary
  if(bp_input != NULL) { // Target layers will not have a valid bp_input
    DistData curr_dist = m_prev_error_signal->DistData();
    DistData next_dist = bp_input->DistData();
    if(curr_dist.colDist == next_dist.colDist
        && curr_dist.rowDist == next_dist.rowDist) {
      View(*m_prev_error_signal, *bp_input);
      View(*m_prev_error_signal_v,
           *m_prev_error_signal,
           ALL,
           IR(0, m_prev_error_signal_v->Width()));
    } else {
      *m_prev_error_signal = *bp_input;
    }
  }

#ifdef __LIB_CUDNN
  // Transfer inputs from CPU to GPUs
  if(m_using_gpus) {
    if(!m_next_layer_using_gpus) {
      m_cudnn->scatter_to_gpus(m_prev_error_signal_d,
                               m_prev_error_signal_v->LockedMatrix(),
                               m_mini_batch_size_per_gpu);
    } else {
      m_prev_error_signal_d = *bp_input_d;
    }
  }
#endif

  // Backprop the compute function.
  double bp_compute_start = get_time();
  bp_compute();
  bp_compute_time += get_time() - bp_compute_start;

#ifdef __LIB_CUDNN
  // Transfer outputs from GPUs to CPU
  if(m_using_gpus && !m_prev_layer_using_gpus) {
    m_cudnn->gather_from_gpus(m_error_signal_v->Matrix(),
                              m_error_signal_d,
                              m_mini_batch_size_per_gpu);
    m_cudnn->synchronize();
  }
#endif

  bp_time += get_time() - bp_start;
}

bool lbann::Layer::update() {
  bool layer_done = false;
  if (m_execution_mode == execution_mode::training) {
    // Apply any updates.
    double update_compute_start = get_time();
    layer_done = update_compute();
    update_time += get_time() - update_compute_start;
  }
  return layer_done;
}

void lbann::Layer::summarize(lbann_summary& summarizer, int64_t step) {
  std::string prefix = "layer" + std::to_string(static_cast<long long>(m_index)) + "/weights/";
  // TODO: implement summarizer functions for other matrix distributions
  const ElMat& wb = get_weights_biases();
  summarizer.reduce_mean(prefix + "mean", wb, step);
  summarizer.reduce_min(prefix + "min", wb, step);
  summarizer.reduce_max(prefix + "max", wb, step);
  summarizer.reduce_stdev(prefix + "stdev", wb, step);
  prefix = "layer" + std::to_string(static_cast<long long>(m_index)) + "/weights_gradient/";
  const ElMat& wb_d = get_weights_biases_gradient();
  summarizer.reduce_mean(prefix + "mean", wb_d, step);
  summarizer.reduce_min(prefix + "min", wb_d, step);
  summarizer.reduce_max(prefix + "max", wb_d, step);
  summarizer.reduce_stdev(prefix + "stdev", wb_d, step);
  prefix = "layer" + std::to_string(static_cast<long long>(m_index)) + "/";
  summarizer.reduce_scalar(prefix + "fp_time", fp_time, step);
  summarizer.reduce_scalar(prefix + "bp_time", bp_time, step);
  summarizer.reduce_scalar(prefix + "update_time", update_time, step);
  reset_counters();
}

void lbann::Layer::setup(int num_prev_neurons) {
  m_num_prev_neurons = num_prev_neurons;
}

void lbann::Layer::check_setup() {
  // If these two are sendable, the other matrices should be fine.
  if (!lbann::lbann_comm::is_sendable(*m_weights)) {
    throw lbann::lbann_exception("Weights too large to send");
  }
  if (!lbann::lbann_comm::is_sendable(*m_activations)) {
    throw lbann::lbann_exception("Activations too large to send");
  }
}

ElMat *lbann::Layer::fp_output() {
  return m_activations;
}

ElMat *lbann::Layer::bp_output() {
  return m_error_signal;
}

void lbann::Layer::setup_fp_input(ElMat *input) {
  this->fp_input = input;
}

void lbann::Layer::setup_bp_input(ElMat *input) {
  this->bp_input = input;
}

#ifdef __LIB_CUDNN
std::vector<DataType *> *lbann::Layer::fp_output_d() {
  if(m_using_gpus) {
    return &m_activations_d;
  } else {
    return NULL;
  }
}

std::vector<DataType *> *lbann::Layer::bp_output_d() {
  if(m_using_gpus) {
    return &m_error_signal_d;
  } else {
    return NULL;
  }
}

void lbann::Layer::setup_fp_input_d(std::vector<DataType *> *fp_input_d) {
  this->fp_input_d = fp_input_d;
}

void lbann::Layer::setup_bp_input_d(std::vector<DataType *> *bp_input_d) {
  this->bp_input_d = bp_input_d;
}
#endif

void lbann::Layer::set_prev_layer_type(layer_type type) {
  this->m_prev_layer_type = type;
}

void lbann::Layer::set_next_layer_type(layer_type type) {
  this->m_next_layer_type = type;
}

bool lbann::Layer::using_gpus() const {
  return m_using_gpus;
}

void lbann::Layer::set_prev_layer_using_gpus(bool using_gpus) {
  m_prev_layer_using_gpus = using_gpus;
}

void lbann::Layer::set_next_layer_using_gpus(bool using_gpus) {
  m_next_layer_using_gpus = using_gpus;
}

bool lbann::Layer::saveToFile(int fd, const char *dirname) {
  char filepath[512];
  sprintf(filepath, "%s/weights_L%d_%03lldx%03lld", dirname, m_index, m_weights->Height()-1, m_weights->Width()-1);

  uint64_t bytes;
  return lbann::write_distmat(-1, filepath, (DistMat *)m_weights, &bytes);
}

bool lbann::Layer::loadFromFile(int fd, const char *dirname) {
  char filepath[512];
  sprintf(filepath, "%s/weights_L%d_%03lldx%03lld.bin", dirname, m_index, m_weights->Height()-1, m_weights->Width()-1);

  uint64_t bytes;
  return lbann::read_distmat(-1, filepath, (DistMat *)m_weights, &bytes);
}

bool lbann::Layer::saveToCheckpoint(int fd, const char *filename, uint64_t *bytes) {
  //writeDist(fd, filename, *m_weights, bytes);

  // Need to catch return value from function
  // m_optimizer->saveToCheckpoint(fd, filename, bytes);
  return true;
}

bool lbann::Layer::loadFromCheckpoint(int fd, const char *filename, uint64_t *bytes) {
  // TODO: implement reader for other matrix distributions
  //readDist(fd, filename, (DistMat&) *m_weights, bytes);

  // Need to catch return value from function
  // m_optimizer->loadFromCheckpoint(fd, filename, bytes);
  return true;
}

bool lbann::Layer::saveToCheckpointShared(lbann::persist& p) {
  // define name to store our parameters
  char name[512];
  sprintf(name, "weights_L%d_%lldx%lld", m_index, m_weights->Height(), m_weights->Width());

  // write out our weights to the model file
  p.write_distmat(persist_type::model, name, (DistMat *)m_weights);

  // if saving training state, also write out state of optimizer
  // m_optimizer->saveToCheckpointShared(p, m_index);

  return true;
}

bool lbann::Layer::loadFromCheckpointShared(lbann::persist& p) {
  // define name to store our parameters
  char name[512];
  sprintf(name, "weights_L%d_%lldx%lld.bin", m_index, m_weights->Height(), m_weights->Width());

  // read our weights from model file
  p.read_distmat(persist_type::model, name, (DistMat *)m_weights);

  // if loading training state, read in state of optimizer
  // m_optimizer->loadFromCheckpointShared(p, m_index);

  return true;
}

void lbann::Layer::fp_set_std_matrix_view() {
  Int cur_mini_batch_size = m_neural_network_model->get_current_mini_batch_size();

  View(*m_prev_activations_v, *m_prev_activations, ALL, IR(0, cur_mini_batch_size));
  if (m_prev_error_signal->Height() > 0) {
    // No previous error signal for the final layer.
    View(*m_prev_error_signal_v, *m_prev_error_signal, ALL,
         IR(0, cur_mini_batch_size));
  }
  View(*m_weighted_sum_v, *m_weighted_sum, ALL, IR(0, cur_mini_batch_size));
  View(*m_error_signal_v, *m_error_signal, ALL, IR(0, cur_mini_batch_size));
  View(*m_activations_v, *m_activations, ALL, IR(0, cur_mini_batch_size));

  // Update the layer's effective mini-batch size so it averages properly.
  /// @todo BVE FIXME This will cause a bug when you are on the last
  /// iteration and the size of the current mini-batch equals the normal
  /// mini-batch size.  In this case one of the ranks gets out of sync
  /// To fix this, we need a flag for when we are on the last mini-batch
  if(cur_mini_batch_size != m_mini_batch_size || 1) {
    // When the current mini-batch is partial, check with the other
    // models to figure out the entire size of the complete mini-batch
    Int total_mini_batch_size = m_comm->intermodel_allreduce((Int) cur_mini_batch_size);
    set_effective_minibatch_size(total_mini_batch_size);
  } else {
    set_effective_minibatch_size(cur_mini_batch_size * m_comm->get_num_models());
  }
}

#if 0
void lbann::Layer::bp_set_std_matrix_view() {
  int64_t cur_mini_batch_size = m_neural_network_model->get_current_mini_batch_size();

  if(m_prev_activations != NULL) { // Input layers will not have a valid fp_input
    View(*m_prev_activations_v, *m_prev_activations, IR(0, m_prev_activations->Height()), IR(0, cur_mini_batch_size));
  }
  View(*m_weighted_sum_v, *m_weighted_sum, IR(0, m_weighted_sum->Height()), IR(0, cur_mini_batch_size));
  if(m_prev_error_signal != NULL) { // Target layers will not have a valid bp_input
    View(*m_prev_error_signal_v, *m_prev_error_signal, IR(0, m_prev_error_signal->Height()), IR(0, cur_mini_batch_size));
  }
  View(*m_error_signal_v, *m_error_signal, IR(0, m_error_signal->Height()), IR(0, cur_mini_batch_size));
  View(*m_activations_v, *m_activations, IR(0, m_activations->Height()), IR(0, cur_mini_batch_size));

  // Update the layer's effective mini-batch size so it averages properly.
  if(cur_mini_batch_size != m_mini_batch_size) { /// When the current mini-batch is partial, check with the other models to figure out the entire size of the complete mini-batch
    int total_mini_batch_size = m_comm->intermodel_allreduce((int) cur_mini_batch_size);
    //    cout << "[" << m_comm->get_rank_in_world() << "] total_mini_batch_size " << total_mini_batch_size << " and cur mini batch size " << cur_mini_batch_size << endl;
    set_effective_minibatch_size(total_mini_batch_size);
  } else {
    set_effective_minibatch_size(cur_mini_batch_size * m_comm->get_num_models());
  }
}
#endif

//enum class weight_initialization {zero, uniform, normal, glorot_normal, glorot_uniform, he_normal, he_uniform};

std::string lbann::Layer::weight_initialization_name(weight_initialization id) {
  switch(id) {
  case weight_initialization::zero :
    return "zero";
    break;
  case weight_initialization::uniform :
    return "uniform";
    break;
  case weight_initialization::normal :
    return "normal";
    break;
  case weight_initialization::glorot_normal :
    return "glorot_normal";
    break;
  case weight_initialization::glorot_uniform :
    return "glorot_uniform";
    break;
  case weight_initialization::he_normal :
    return "he_normal";
    break;
  case weight_initialization::he_uniform :
    return "he_uniform";
    break;
  default:
    char b[1024];
    sprintf(b, "%s %d :: unknown weight_initialization: %d", __FILE__, __LINE__, id);
    throw lbann_exception(b);
  }
}
