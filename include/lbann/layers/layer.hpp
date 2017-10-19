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
  Layer(const int index, lbann_comm *comm);
  Layer(const Layer& other);
  Layer& operator=(const Layer& other);

  virtual ~Layer();

  virtual Layer* copy() const = 0;

  template <data_layout T_layout>
  void initialize_distributed_matrices();

  virtual void forward_prop();
  virtual void back_prop();
  virtual bool update();
  virtual void summarize_stats(lbann_summary& summarizer, int step);
  virtual void summarize_matrices(lbann_summary& summarizer, int step);
  /**
   * Print information at the end of an epoch.
   * This is always called on the model masters and should synchronize
   * printing if needed.
   */
  virtual void epoch_print() const {}
  virtual DataType checkGradientMB(Layer& PrevLayer, DataType Epsilon=1e-4) {
    return 0.0;
  };

  /** Setup layer dimensions and data.
   *  By default, this calls the setup_pointers, setup_dims,
   *  setup_data, setup_views, and setup_gpu (if needed)
   *  methods. Unless the setup_pointers function has been replaced in
   *  an inherited class, it is assumed that pointers to parent/child
   *  layers have already been initialized.
   */
  virtual void setup();
  /** Validate that the setup is reasonable. */
  virtual void check_setup();

  /** Return this layer's type, e.g: "fully connected," "batch normalization," etc. */
  virtual std::string get_type() const = 0;

  /** Returns this layer's name; this is an arbitrary string, e.g, assigned in a prototext file. */
  std::string get_name() const { return m_name; }

  /** Sets this layer's name; this is an arbitrary string, e.g, assigned in a prototext file. */
  void set_name(std::string name) { m_name = name; }
  
  /** Returns a description of the parameters passed to the ctor */
  virtual std::string get_description() const { 
    return std::string {} + get_type() + " - DESCRIPTION NOT IMPLEMENTED FOR THIS LAYER\n"
     + " to get a description, you need to edit the class file by adding this method:\n"
     + " virtual std::string get_descrciption() const override";
  }
  /** Returns a description of the topology */
  virtual std::string get_topo_description() const { return ""; };

  /** Returns a string description of the data_layout */
  std::string get_data_layout_string(data_layout d) const; 

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
  virtual AbsDistMat& get_activations() {
    return *m_activations_v;
  }
  /** Reset layer stat counters. */
  virtual void reset_counters() {
    fp_time = 0.0;
    fp_compute_time = 0.0;
    bp_time = 0.0;
    bp_compute_time = 0.0;
    update_time = 0.0;
  }

  bool using_gpus() const {
    return m_using_gpus;
  }

  /// Following function tells whether a layer has weights; default is false
  virtual bool is_learning_layer() { return false; }

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

  /** Get forward propagation output, as seen by next layer. */
  virtual void get_fp_output(AbsDistMat& fp_output, const Layer* next_layer = NULL) const;
  /** Get backward propagation output, as seen by previous layer. */
  virtual void get_bp_output(AbsDistMat& fp_output, const Layer* prev_layer = NULL) const;
#ifdef __LIB_CUDNN
  /** Get forward propagation output on GPUs, as seen by next layer. */
  virtual void get_gpu_fp_output(std::vector<DataType*>& fp_output, const Layer* next_layer = NULL) const;
  /** Get backward propagation output on GPUs, as seen by previous layer. */
  virtual void get_gpu_bp_output(std::vector<DataType*>& bp_output, const Layer* prev_layer = NULL) const;
#endif // __LIB_CUDNN
  /** Get forward propagation output dimensions, as seen by next layer. */
  virtual const std::vector<int> fp_output_dims(const Layer* next_layer = NULL) const;

  /** Get list of parent layers. */
  std::vector<const Layer*>& get_parent_layers();
  /** Get list of parent layers (const). */
  const std::vector<const Layer*>& get_parent_layers() const;
  /** Get list of child layers. */
  std::vector<const Layer*>& get_child_layers();
  /** Get list of child layers (const). */
  const std::vector<const Layer*>& get_child_layers() const;

  /** Add a parent layer. */
  void add_parent_layer(const Layer* parent);
  /** Add a child layer. */
  void add_child_layer(const Layer* child);

 protected:

  int m_index;                 ///< Layer index (start with 0)

  lbann_comm *m_comm;

  int m_num_neurons;                    ///< Number of neurons
  int m_num_neuron_dims;                ///< Number of dimensions in neuron tensor
  std::vector<int> m_neuron_dims;       ///< Neuron tensor dimensions
  int m_num_prev_neurons;               ///< Number of neurons in previous layer
  int m_num_prev_neuron_dims;           ///< Number of dimensions in previous layer's neuron tensor
  std::vector<int> m_prev_neuron_dims;  ///< Neuron tensor dimensions in previous layer

  AbsDistMat *m_prev_activations;   ///< Local view or copy of the activations from the "previous" layer ((# previous layer's neurons) x mini-batch size)
  AbsDistMat *m_activations;        ///< Activations - non-linearity applied to weighted sum ((# neurons) x mini-batch size)
  AbsDistMat *m_activations_v;      ///< View of active columns in activations matrix
  AbsDistMat *m_prev_error_signal;  ///< Local copy of the error signal from "previous" layer ((# neurons) x mini-batch size)
  AbsDistMat *m_error_signal;       ///< Error signal to "next" layer (i.e. deltas) ((# neurons) x mini-batch size)
  AbsDistMat *m_error_signal_v;     ///< View of active columns in error signal matrix

  /** List of parent layers. */
  std::vector<const Layer*> m_parent_layers;
  /** List of child layers. */
  std::vector<const Layer*> m_child_layers;

  /** Maximum number of parent layers.
   *  A negative value indicates no limit.
   */
  int m_max_num_parent_layers;
  /** Maximum number of child layers.
   *  A negative value indicates no limit.
   */
  int m_max_num_child_layers;

  execution_mode  m_execution_mode;
  model *m_neural_network_model;

  /** Setup views of the matrices for the layer's forward propagation. */
  virtual void fp_set_std_matrix_view();
  /** Setup views of the matrices for the layer's backward propagation. */
  virtual void bp_set_std_matrix_view();
#ifdef __LIB_CUDNN
  /** Pin host memory if needed for GPU memory transfers. */
  virtual void pin_data();
#endif // __LIB_CUDNN

  /** Setup pointers to parent and child layers.
   *  Called by the setup function. This base method just checks that
   *  the number of parents and children are valid. Pointers to the
   *  parent/child layers are assumed to be initialized already.
   */
  virtual void setup_pointers();
  /** Setup neuron tensor dimensions
   *  Called by the setup function. This base method initializes the
   *  input neuron tensor dimensions and sets the output neuron tensor
   *  dimensions equal to the input.
   */
  virtual void setup_dims();
  /** Setup layer data.
   *  Called by the setup function. This base method initializes the
   *  activations and error signal matrices.
   */
  virtual void setup_data();
  /** Setup GPU objects.
   *  Called by the setup function if GPUs are enabled. This base
   *  method initializes the activations and error signal matrices on
   *  GPUs.
   */
  virtual void setup_gpu();
  /** Setup matrix views.
   *  Called by the setup function.
   */
  virtual void setup_views() {}
  /** Perform the main computation for a forward propagation step. */
  virtual void fp_compute() {}
  /** Perform the main computation for a backward propagation step. */
  virtual void bp_compute() {}
  /** Perform the main computation for an update step. */
  virtual bool update_compute() { return true; }

  /** Whether current layer is using GPUs. */
  bool m_using_gpus;

  /// cuDNN manager
  cudnn::cudnn_manager *m_cudnn;

#ifdef __LIB_CUDNN

  /** Number of mini-batch samples per GPU. */
  int m_mini_batch_size_per_gpu;
  /** Maximum number of mini-batch samples per GPU. */
  int m_max_mini_batch_size_per_gpu;

  /** GPU memory for activations from "previous" layer. */
  std::vector<DataType*> m_prev_activations_d;
  /** GPU memory for activations. */
  std::vector<DataType*> m_activations_d;
  /** GPU memory for error signal from "next" layer. */
  std::vector<DataType*> m_prev_error_signal_d;
  /** GPU memory for error signal. */
  std::vector<DataType*> m_error_signal_d;

  /** Whether to copy forward propagation input from CPU to GPUs. */
  bool m_copy_fp_input_to_gpus;
  /** Whether to copy forward propagation output from GPUs to CPU. */
  bool m_copy_fp_output_from_gpus;
  /** Whether to copy backward propagation input from CPU to GPUs. */
  bool m_copy_bp_input_to_gpus;
  /** Whether to copy backward propagation output from GPUs to CPU. */
  bool m_copy_bp_output_from_gpus;

  /** cuDNN descriptor for neuron tensor from "previous" layer. */
  cudnnTensorDescriptor_t m_prev_neurons_cudnn_desc;
  /** cuDNN descriptor for neuron tensor. */
  cudnnTensorDescriptor_t m_neurons_cudnn_desc;

#endif // __LIB_CUDNN

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

  std::string m_name;
};
}

#endif // LBANN_LAYER_HPP_INCLUDED
