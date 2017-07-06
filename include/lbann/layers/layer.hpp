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

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/optimizers/optimizer_sgd.hpp"
#include "lbann/optimizers/optimizer_adagrad.hpp"
#include "lbann/optimizers/optimizer_rmsprop.hpp"
#include "lbann/optimizers/optimizer_adam.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/io/persist.hpp"
#include <string>
#include <vector>

namespace lbann {

// Forward-declare this.
class model;

class Layer {
 public:
  Layer(const int index, lbann_comm *comm, int mbsize);
  Layer(const Layer& other);
  Layer& operator=(const Layer& other);

  virtual ~Layer();

  template <data_layout T_layout>
  void initialize_distributed_matrices();

  virtual void forwardProp();
  virtual void backProp();
  virtual bool update();
  virtual void summarize_stats(lbann_summary& summarizer, int step);
  virtual void summarize_matrices(lbann_summary& summarizer, int step);
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
  virtual DataType checkGradientMB(Layer& PrevLayer, DataType Epsilon=1e-4) {
    return 0.0;
  };

  /**
   * This is called on every layer to set up its prev/next layer pointers,
   * dimensions, and allocate data. By default, it calls the setup_dims and
   * setup_data methods.
   */
  virtual void setup(const Layer *prev_layer, const Layer *next_layer);
  /** Validate that the setup is reasonable. */
  virtual void check_setup();

  /** Return this layer's name */
  virtual std::string get_name() const = 0;

  /** Return the layer index. */
  inline int get_index() const {
    return m_index;
  }
  /** Set the layer index. */
  inline void set_index(int i) {
    m_index = i;
  }
  /** Return the number of neurons from previous layer. */
  inline int get_num_prev_neurons() const {
    return m_num_prev_neurons;
  }
  /** Return the number of dimensions in neuron tensor from previous layer. */
  inline int get_num_prev_neuron_dims() const {
    return m_num_prev_neuron_dims;
  }
  /** Return the dimensions of neuron tensor from previous layer. */
  inline const std::vector<int>& get_prev_neuron_dims() const {
    return m_prev_neuron_dims;
  }
  /** Return the number of neurons. */
  inline int get_num_neurons() const {
    return m_num_neurons;
  }
  /** Return the number of dimensions in neuron tensor. */
  inline int get_num_neuron_dims() const {
    return m_num_neuron_dims;
  }
  /** Return the dimensions of neuron tensor. */
  inline const std::vector<int>& get_neuron_dims() const {
    return m_neuron_dims;
  }
  /** Return the execution mode. */
  inline execution_mode get_execution_mode() const {
    return m_execution_mode;
  }
  /** Set the execution mode. */
  inline void set_execution_mode(execution_mode mode) {
    m_execution_mode = mode;
  }
  /** Return the data layout of the given layer -- Every concrete
      layer has to overrride this with its T_layout template parameter */
  virtual data_layout get_data_layout() const = 0;
  /** Return (a view of) the activations matrix for this layer. */
  virtual ElMat& get_activations() {
    return *m_activations;
  }
  /** Reset layer stat counters. */
  virtual void reset_counters() {
    fp_time = 0.0;
    bp_time = 0.0;
    update_time = 0.0;
  }

  /** Return the size of mini-batch this layer uses. */
  virtual int get_minibatch_size() const {
    return m_mini_batch_size;
  }
  /**
   * Get the "effective" size of a mini-batch.
   * This is for backward propagation, etc. when there are more updates being
   * contributed than the local mini-batch size implies (e.g. when doing
   * inter-model updates).
   */
  virtual int get_effective_minibatch_size() const {
    return m_effective_mbsize;
  }
  /** Set the effective size of a mini-batch to size. */
  virtual void set_effective_minibatch_size(int size) {
    m_effective_mbsize = size;
  }

  bool using_gpus() const {
    return m_using_gpus;
  }

  /** Return the neural network model of this layer. */
  inline model* get_neural_network_model() const {
    return m_neural_network_model;
  }
  /** Set the neural network model of this layer. */
  inline void set_neural_network_model(model* const m) {
    m_neural_network_model = m;
  }
  virtual El::Matrix<El::Int>* get_sample_indices_per_mb() { return nullptr; };

  virtual bool saveToFile(int fd, const char *filename) { return true; };
  virtual bool loadFromFile(int fd, const char *filename) { return true; };

  virtual bool saveToCheckpoint(int fd, const char *filename, size_t *bytes);
  virtual bool loadFromCheckpoint(int fd, const char *filename, size_t *bytes);

  virtual bool saveToCheckpointShared(persist& p);
  virtual bool loadFromCheckpointShared(persist& p);

 protected:
  int m_index;                 ///< Layer index (start with 0)

  lbann_comm *m_comm;

  int m_num_neurons;                    ///< Number of neurons
  int m_num_neuron_dims;                ///< Number of dimensions in neuron tensor
  std::vector<int> m_neuron_dims;       ///< Neuron tensor dimensions
  int m_num_prev_neurons;               ///< Number of neurons in previous layer
  int m_num_prev_neuron_dims;           ///< Number of dimensions in previous layer's neuron tensor
  std::vector<int> m_prev_neuron_dims;  ///< Neuron tensor dimensions in previous layer

  execution_mode  m_execution_mode;

  ElMat *m_prev_error_signal;    ///< Local copy of the error signal from "previous" layer ((# neurons) x mini-batch size)
  ElMat *m_prev_error_signal_v;  ///< View of active columns in previous error signal matrix

  ElMat *m_activations;          ///< Activations - non-linearity applied to weighted sum ((# neurons) x mini-batch size)
  ElMat *m_activations_v;        ///< View of active columns in activations matrix

  model *m_neural_network_model;

  const Layer *m_prev_layer;  ///< Pointer to previous layer
  const Layer *m_next_layer;  ///< Pointer to next layer

  ElMat *m_error_signal;        ///< Error signal to "next" layer (i.e. deltas) ((# neurons) x mini-batch size)
  ElMat *m_error_signal_v;      ///< View of active columns in error signal matrix

  ElMat *m_prev_activations;    ///< Local view or copy of the activations from the "previous" layer ((# previous layer's neurons) x mini-batch size)
  ElMat *m_prev_activations_v;  ///< View of active columns in previous activations matrix

  /** Setup views of the matrices for the layer's forward propagation. */
  virtual void fp_set_std_matrix_view();
  /** Setup views of the matrices for the layer's backward propagation. */
  virtual void bp_set_std_matrix_view();
#ifdef __LIB_CUDNN
  /** Pin host memory if needed for GPU memory transfers. */
  virtual void pin_data();
#endif

  /**
   * Called by setup(), each layer should override this to call its parent and
   * set up the layer's m_num_neurons and m_neuron_dims. This base method sets
   * up m_num_prev_neurons and related methods.
   */
  virtual void setup_dims();
  /**
   * Called by setup(), each layer should override to call its parent and
   * set up the layer's data (e.g. weights). This base method sets up the
   * activations and error signal matrices. This is always called after
   * setup_dims.
   */
  virtual void setup_data();
  /**
   * Called by setup(), each layer using GPU should override this to
   * call its parent and set up the layer's GPU data
   * (e.g. weights). This base method sets up the activations and
   * error signal matrices. This is always called after setup_data.
   */
  virtual void setup_gpu();
  /** Perform the layers work / main function for forward propagation */
  virtual void fp_compute() {}
  /** Perform the layers work / main function for backward propagation */
  virtual void bp_compute() {}
  /** Perform the layers work / main function for the update step */
  virtual bool update_compute() { return true; }

  /** Current layer is using GPUs. */
  bool m_using_gpus;

  /// cuDNN manager
  cudnn::cudnn_manager *m_cudnn;

#ifdef __LIB_CUDNN

  /** Forward propagation input uses pinned memory. */
  bool m_fp_input_pinned;
  /** Forward propagation output uses pinned memory. */
  bool m_fp_output_pinned;
  /** Backward propagation input uses pinned memory. */
  bool m_bp_input_pinned;
  /** Backward propagation output uses pinned memory. */
  bool m_bp_output_pinned;

  /** Number of mini-batch samples per GPU. */
  int m_mini_batch_size_per_gpu;

  /** GPU memory for activations from "previous" layer. */
  std::vector<DataType *> m_prev_activations_d;
  /** GPU memory for activations. */
  std::vector<DataType *> m_activations_d;
  /** GPU memory for error signal from "next" layer. */
  std::vector<DataType *> m_prev_error_signal_d;
  /** GPU memory for error signal. */
  std::vector<DataType *> m_error_signal_d;

  /** cuDNN descriptor for neuron tensor from "previous" layer. */
  cudnnTensorDescriptor_t m_prev_neurons_cudnn_desc;
  /** cuDNN descriptor for neuron tensor. */
  cudnnTensorDescriptor_t m_neurons_cudnn_desc;

#endif

  /** Size of the local mini-batch. */
  int m_mini_batch_size;
  /** "Effective" mini-batch size for backward propagation, etc.. */
  int m_effective_mbsize;

  /** Time spent in forward propagation. */
  double fp_time;
  /** Time spent in the forward propagation computation. */
  double fp_compute_time;
  /** Time spent in backward propagation. */
  double bp_time;
  /** Time spent in the backward propagation computation. */
  double bp_compute_time;
  /** Time spent in updates. */
  double update_time;
};
}

#endif // LBANN_LAYER_HPP_INCLUDED
