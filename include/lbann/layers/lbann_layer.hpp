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
// lbann_layer .h .cpp - Parent class for all layer types
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_HPP_INCLUDED
#define LBANN_LAYER_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/layers/lbann_layer_activations.hpp"
#include "lbann/utils/lbann_summary.hpp"
#include "lbann/optimizers/lbann_optimizer.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"
#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"
#include "lbann/optimizers/lbann_optimizer_adam.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/lbann_timer.hpp"
#include "lbann/io/lbann_persist.hpp"
#include <string>
#include <vector>

namespace lbann {

// Forward-declare this.
class regularizer;
class model;

// @todo: check list of layer types
enum class layer_type {fully_connected, softmax, convolution, pooling,
                       local_response_normalization,
                       input_distributed_minibatch, input_distributed_minibatch_parallel_io,
                       input_partitioned_minibatch_parallel_io,
                       target_distributed_minibatch, target_distributed_minibatch_parallel_io,
                       target_partitioned_minibatch_parallel_io,
                       reconstruction,
                       INVALID
                      };
enum class layer_category {compute, io, SPECIAL, INVALID};

static const char *__attribute__((used)) _layer_type_to_string(layer_type l) {
  switch(l) {
  case layer_type::fully_connected:
    return "fully_connected";
  case layer_type::softmax:
    return "softmax";
  case layer_type::convolution:
    return "convolution";
  case layer_type::pooling:
    return "pooling";
  case layer_type::local_response_normalization:
    return "local_response_normalization";
  case layer_type::input_distributed_minibatch:
    return "input_distributed_minibatch";
  case layer_type::input_distributed_minibatch_parallel_io:
    return "input_distributed_minibatch_parallel_io";
  case layer_type::input_partitioned_minibatch_parallel_io:
    return "input_partitioned_minibatch_parallel_io";
  case layer_type::target_distributed_minibatch:
    return "target_distributed_minibatch";
  case layer_type::target_distributed_minibatch_parallel_io:
    return "target_distributed_minibatch_parallel_io";
  case layer_type::target_partitioned_minibatch_parallel_io:
    return "target_partitioned_minibatch_parallel_io";
  case layer_type::reconstruction:
    return "reconstruction";
  case layer_type::INVALID:
    return "INVALID";
  default:
    throw(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " Invalid layer_type specified");
  }
  return NULL;
}

static layer_category __attribute__((used)) _layer_type_to_category(layer_type l) {
  switch(l) {
  case layer_type::fully_connected:
  case layer_type::softmax:
  case layer_type::convolution:
  case layer_type::pooling:
  case layer_type::local_response_normalization:
    return layer_category::compute;
  case layer_type::input_distributed_minibatch:
  case layer_type::input_distributed_minibatch_parallel_io:
  case layer_type::input_partitioned_minibatch_parallel_io:
  case layer_type::target_distributed_minibatch:
  case layer_type::target_distributed_minibatch_parallel_io:
  case layer_type::target_partitioned_minibatch_parallel_io:
    return layer_category::io;
  case layer_type::reconstruction:
    return layer_category::SPECIAL;
  case layer_type::INVALID:
    return layer_category::INVALID;
  default:
    throw(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " Invalid layer_type specified");
  }
  return layer_category::INVALID;
}


class Layer {
 public:
  Layer(data_layout data_dist, const uint index, lbann_comm *comm, optimizer *opt,
        uint mbsize, activation_type activation=activation_type::ID,
        std::vector<regularizer *> regs= {});

  virtual ~Layer();

  static std::string weight_initialization_name(weight_initialization id);

  void initialize_model_parallel_distribution();
  void initialize_data_parallel_distribution();

  virtual void forwardProp();
  virtual void backProp();
  virtual bool update();
  virtual void summarize(lbann_summary& summarizer, int64_t step);
  /**
   * Print information at the end of an epoch.
   * This is always called on the model masters and should synchronize
   * printing if needed.
   */
  virtual void epoch_print() const {}
  /**
   * Called on every layer at the end of each epoch to give it the chance to
   * reset/clean up.
   */
  virtual void epoch_reset() {}
  virtual DataType checkGradientMB(Layer& PrevLayer, const DataType Epsilon=1e-4) {
    return 0.0;
  };

  virtual void setup(int);
  /** Validate that the setup is reasonable. */
  virtual void check_setup();

  /** Return the index of this layer. */
  inline uint get_index() const {
    return Index;
  }
  /** Return (a view of) the weights/biases matrix for this layer. */
  virtual ElMat& get_weights_biases() {
    return *m_weights;
  }
  /** Return (a view of) the weights/biases gradient matrix for this layer. */
  virtual ElMat& get_weights_biases_gradient() {
    return *m_weights_gradient;
  }
  /** Return (a view of) the activations matrix for this layer. */
  virtual ElMat& get_activations() {
    return *m_activations;
  }
  /** Return the layer's optimizer. */
  virtual optimizer *get_optimizer() const {
    return m_optimizer;
  }
  /** Reset layer stat counters. */
  virtual void reset_counters() {
    fp_time = 0.0;
    fp_linearity_time = 0.0;
    fp_nonlinearity_time = 0.0;
    bp_time = 0.0;
    bp_linearity_time = 0.0;
    bp_nonlinearity_time = 0.0;
    update_time = 0.0;
  }

  /** Return the size of mini-batch this layer uses. */
  virtual uint get_minibatch_size() const {
    return m_mini_batch_size;
  }
  /**
   * Get the "effective" size of a mini-batch.
   * This is for backward propagation, etc. when there are more updates being
   * contributed than the local mini-batch size implies (e.g. when doing
   * inter-model updates).
   */
  virtual uint get_effective_minibatch_size() const {
    return m_effective_mbsize;
  }
  /** Set the effective size of a mini-batch to size. */
  virtual void set_effective_minibatch_size(uint size) {
    m_effective_mbsize = size;
  }

  ElMat *fp_output();
  ElMat *bp_output();
  void setup_fp_input(ElMat *fp_input);
  void setup_bp_input(ElMat *bp_input);

  void set_prev_layer_type(layer_type type);
  void set_next_layer_type(layer_type type);
  bool using_gpus() const;
  void set_prev_layer_using_gpus(bool using_gpus);
  void set_next_layer_using_gpus(bool using_gpus);
#ifdef __LIB_CUDNN
  std::vector<DataType *> *fp_output_d();
  std::vector<DataType *> *bp_output_d();
  void setup_fp_input_d(std::vector<DataType *> *fp_input_d);
  void setup_bp_input_d(std::vector<DataType *> *bp_input_d);
#endif

  virtual El::Matrix<El::Int>* get_sample_indices_per_mb() { return nullptr; };

  bool saveToFile(int fd, const char *filename);
  bool loadFromFile(int fd, const char *filename);

  virtual bool saveToCheckpoint(int fd, const char *filename, uint64_t *bytes);
  virtual bool loadFromCheckpoint(int fd, const char *filename, uint64_t *bytes);

  virtual bool saveToCheckpointShared(persist& p);
  virtual bool loadFromCheckpointShared(persist& p);

 public:

  /// Layer type
  layer_type m_type;

  uint Index;      // Layer index (start with 0)
  uint NumNeurons; // # neurons
  execution_mode  m_execution_mode;
  data_layout m_data_layout;

  ElMat *m_weights;            /// Weight matrix (computes weight sum of inputs ((# neurons) x (# previous layer's neurons))
  ElMat *m_weights_gradient;   /// Gradient w.r.t. weight matrix ((# neurons) x (# previous layer's neurons))
  ElMat *m_weighted_sum;       /// Weighted sum - Output of forward pass linear transformation ((# neurons) x mini-batch size)
  ElMat *m_prev_error_signal;  /// Local copy of the error signal from "previous" layer ((# neurons) x mini-batch size)

  ElMat *m_activations;        /// Activations - non-linearity applied to weighted sum ((# neurons) x mini-batch size)

  /// Create a view of each matrix so that it can accomodate partial mini-batches
  ElMat *m_weighted_sum_v;
  ElMat *m_prev_error_signal_v;
  ElMat *m_activations_v;

  ElMat *fp_input;            /// Pointer to input for the forward propagation - no local storage
  ElMat *bp_input;            /// Pointer to the input for the backward propagation - no local storage

  model *neural_network_model;

 protected:
  /// Type of previous layer
  layer_type m_prev_layer_type;
  /// Type of next layer
  layer_type m_next_layer_type;

  Int m_num_prev_neurons; /// Number of neurons in previous layer
  activation_type m_activation_type;

  ElMat *m_error_signal;       /// Error signal to "next" layer (i.e. deltas) ((# neurons) x mini-batch size)
  ElMat *m_prev_activations;   /// Local copy of the activations from the "previous" layer ((# previous layer's neurons) x mini-batch size)

  ElMat *m_error_signal_v;
  ElMat *m_prev_activations_v;

  optimizer *m_optimizer;

  lbann_comm *comm;

 protected:

  /** Setup views of the matrices for the layer's forward propagation. */
  virtual void fp_set_std_matrix_view();
#if 0
  /** Setup views of the matrices for the layer's backward propagation. */
  virtual void bp_set_std_matrix_view();
#endif
  /** Apply the layer's linear update in forward propagation. */
  virtual void fp_linearity() {}
  /** Handle the layer's linearity in backward propagation. */
  virtual void bp_linearity() {}
  /** Apply the layer's nonlinearity in forward propagation. */
  virtual void fp_nonlinearity();
  /** Handle the layer's nonlinearity in backward propagation. */
  virtual void bp_nonlinearity();

  /** Current layer is using GPUs. */
  bool m_using_gpus;
  /** Previous layer is using GPUs. */
  bool m_prev_layer_using_gpus;
  /** Next layer is using GPUs. */
  bool m_next_layer_using_gpus;

  /// cuDNN manager
  cudnn::cudnn_manager *m_cudnn;

#ifdef __LIB_CUDNN

  /// Number of mini-batch samples per GPU
  Int m_mini_batch_size_per_gpu;

  /** GPU memory for activations from "previous" layer. */
  std::vector<DataType *> m_prev_activations_d;
  /** GPU memory for activations. */
  std::vector<DataType *> m_activations_d;
  /** GPU memory for output of forward pass linear transformation. */
  std::vector<DataType *> m_weighted_sum_d;
  /** GPU memory for error signal from "next" layer. */
  std::vector<DataType *> m_prev_error_signal_d;
  /** GPU memory for error signal. */
  std::vector<DataType *> m_error_signal_d;
  /** GPU memory for forward propagation input. */
  std::vector<DataType *> *fp_input_d;
  /** GPU memory for backward propagation input. */
  std::vector<DataType *> *bp_input_d;

#endif

  /** Activation function */
  Activation *m_activation_fn;
  /** Regularizers being applied to the layer. */
  std::vector<regularizer *> regularizers;
  /** Size of the local mini-batch. */
  uint m_mini_batch_size;
  /** "Effective" mini-batch size for backward propagation, etc.. */
  uint m_effective_mbsize;

  /** Time spent in forward propagation. */
  double fp_time;
  /** Time spent in the forward propagation linearity. */
  double fp_linearity_time;
  /** Time spent in the forward propagation nonlinearity. */
  double fp_nonlinearity_time;
  /** Time spent in backward propagation. */
  double bp_time;
  /** Time spent in the backward propagation linearity. */
  double bp_linearity_time;
  /** Time spent in the backward propagation linearity. */
  double bp_nonlinearity_time;
  /** Time spent in updates. */
  double update_time;
};
}


#endif // LBANN_LAYER_HPP_INCLUDED
