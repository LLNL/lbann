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
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/io/persist.hpp"
#include <lbann.pb.h>
#include <string>
#include <vector>

namespace lbann {

// Forward-declare this.
class model;

class Layer {
 public:

  /** Constructor. */
  Layer(lbann_comm *comm);
  /** Copy constructor. */
  Layer(const Layer& other);
  /** Copy assignment operator. */
  Layer& operator=(const Layer& other);
  /** Destructor. */
  virtual ~Layer();

  /** Copy function. */
  virtual Layer* copy() const = 0;

  /** Forward propagation step.
   *  Apply the layer's operation to the previous activations tensor
   *  to obtain the activations tensor.
   */
  virtual void forward_prop();
  /** Backward propagation step.
   *  Compute the objective function gradient w.r.t. the previous
   *  activations tensor and the weights. The gradient w.r.t. the
   *  previous activations tensor is called the error signal tensor.
   */
  virtual void back_prop();
  /** Update step.
   *  This updates the layer's internal members. The weights are
   *  updated elsewhere.
   */
  virtual bool update();

  /** Clear the error signal tensor. */
  virtual void clear_error_signals(int mini_batch_size);

  virtual void summarize_stats(lbann_summary& summarizer, int step);
  virtual void summarize_matrices(lbann_summary& summarizer, int step);

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
  virtual std::string get_topo_description() const {
    std::stringstream s;
    for (size_t i = 0; i < m_child_layers.size(); ++i) {
      if (i != 0) {
        s << ", ";
      }
      const auto& dims = get_neuron_dims(i);
      s << "activations[" << i << "]=[";
      switch (dims.size()) {
      case 0:
        s << "0"; break;
      case 2:
        s << dims[0] << "c x"
          << dims[1] << "w";
        break;
      case 3:
        s << dims[0] << "c x "
          << dims[1] << "w x "
          << dims[2] << "h";
        break;
      default:
        s << dims[0];
        for (size_t j = 1; j < dims.size(); ++j) {
          s << " x " << dims[j];
        }
      }
      s << ", " << m_activations[i]->Width() << "s]";
    }
    return s.str();
  }

  /** Returns a string description of the data_layout */
  std::string get_data_layout_string(data_layout d) const;

  /** Return the number of neurons from previous layer. */
  virtual int get_num_prev_neurons(int parent_index = 0) const {
    return m_num_prev_neurons;
  }
  /** Return the number of dimensions in neuron tensor from previous layer. */
  virtual int get_num_prev_neuron_dims(int parent_index = 0) const {
    return m_num_prev_neuron_dims;
  }
  /** Return the dimensions of neuron tensor from previous layer. */
  virtual std::vector<int> get_prev_neuron_dims(int parent_index = 0) const {
    return m_prev_neuron_dims;
  }
  /** Return the number of neurons. */
  virtual int get_num_neurons(int child_index = 0) const {
    return m_num_neurons;
  }
  /** Return the number of dimensions in neuron tensor. */
  virtual int get_num_neuron_dims(int child_index = 0) const {
    return m_num_neuron_dims;
  }
  /** Return the dimensions of neuron tensor. */
  virtual std::vector<int> get_neuron_dims(int child_index = 0) const {
    return m_neuron_dims;
  }
  /** Return the data layout of the given layer -- Every concrete
      layer has to overrride this with its T_layout template parameter */
  virtual data_layout get_data_layout() const = 0;

  /** Reset layer stat counters. */
  virtual void reset_counters();

  bool using_gpus() const {
    return m_using_gpus;
  }

  /** Get expected number of parent layers.
   *  A negative value indicates no limit.
   */
  inline int get_expected_num_parent_layers() const {
    return m_expected_num_parent_layers;
  }
  /** Get expected number of child layers.
   *  A negative value indicates no limit.
   */
  inline int get_expected_num_child_layers() const {
    return m_expected_num_child_layers;
  }

  /** Return the model that owns this layer. */
  inline model* get_model() const { return m_model; }
  /** Set the model that owns this layer. */
  inline void set_model(model* m) { m_model = m; }

  virtual El::Matrix<El::Int>* get_sample_indices_per_mb() { return nullptr; };

  virtual bool saveToFile(int fd, const char *filename) const { return true; };
  virtual bool loadFromFile(int fd, const char *filename) { return true; };

  virtual bool saveToCheckpoint(int fd, const char *filename, size_t *bytes) const;
  virtual bool loadFromCheckpoint(int fd, const char *filename, size_t *bytes);

  virtual bool save_to_checkpoint_shared(persist& p) const;
  virtual bool load_from_checkpoint_shared(persist& p);
  
  /** Write layer to proto file */
  virtual void write_proto(lbann_data::Layer* proto) const;

  /** Get forward propagation output, as seen by child layer. */
  virtual void get_fp_output(AbsDistMat& fp_output, const Layer* child) const;
  /** Get backward propagation output, as seen by parent layer. */
  virtual void get_bp_output(AbsDistMat& fp_output, const Layer* parent) const;
#ifdef __LIB_CUDNN
  /** Get forward propagation output on GPUs, as seen by child layer. */
  virtual void get_gpu_fp_output(cudnn::matrix& output_d,
                                 const Layer* child) const;
  /** Get back propagation output on GPUs, as seen by parent layer. */
  virtual void get_gpu_bp_output(cudnn::matrix& output_d,
                                 const Layer* parent) const;
#endif // __LIB_CUDNN
  /** Get forward propagation output dimensions, as seen by next layer. */
  virtual std::vector<int> fp_output_dims(const Layer* next_layer = nullptr) const;

  virtual void add_to_error_signal(const AbsDistMat& gradient,
                                   DataType scale = DataType(1),
                                   int parent_index = 0) {
    El::Axpy(scale, gradient, *m_error_signals[parent_index]);
  }

  /** Get list of parent layers. */
  std::vector<const Layer*>& get_parent_layers();
  /** Get list of parent layers (const). */
  const std::vector<const Layer*>& get_parent_layers() const;
  /** Get list of child layers. */
  std::vector<const Layer*>& get_child_layers();
  /** Get list of child layers (const). */
  const std::vector<const Layer*>& get_child_layers() const;

  /** Get number of parent layers. */
  inline int get_num_parents() const { return m_parent_layers.size(); }
  /** Get number of child layers. */
  inline int get_num_children() const { return m_child_layers.size(); }

  /** Get names in a particular list of layers */
  static std::string get_layer_names(const std::vector<const Layer*>& list);
  std::string get_child_names() const { return get_layer_names(m_child_layers); }
  std::string get_parent_names() const { return get_layer_names(m_parent_layers); }

  /** Add a parent layer. */
  void add_parent_layer(const Layer* parent);
  /** Add a child layer. */
  void add_child_layer(const Layer* child);

  /** clear the list of parent layer pointers without deallocating them. */
  void clear_parent_layers();
  /** clear the list of child layer pointers without deallocating them. */
  void clear_child_layers();

  /** Get list of pointers to other layers. */
  virtual std::vector<Layer*> get_layer_pointers();
  /** Set list of pointers to other layers. */
  virtual void set_layer_pointers(std::vector<Layer*> layers);

  /** Get list of pointers to weights. */
  std::vector<weights*> get_weights() { return m_weights; }
  /** Set list of pointers to weights. */
  void set_weights(std::vector<weights*> w) { m_weights = w; }
  /** Replace weights with another Layer's weights*/
  void replace_weights(Layer* other_layer);

  AbsDistMat& get_prev_activations(int parent_index = 0);
  AbsDistMat& get_activations(int child_index = 0);
  AbsDistMat& get_prev_error_signals(int child_index = 0);
  AbsDistMat& get_error_signals(int parent_index = 0);
  const AbsDistMat& get_prev_activations(int parent_index = 0) const;
  const AbsDistMat& get_activations(int child_index = 0) const;
  const AbsDistMat& get_prev_error_signals(int child_index = 0) const;
  const AbsDistMat& get_error_signals(int parent_index = 0) const;

  Mat& get_local_prev_activations(int parent_index = 0);
  Mat& get_local_activations(int child_index = 0);
  Mat& get_local_prev_error_signals(int child_index = 0);
  Mat& get_local_error_signals(int parent_index = 0);
  const Mat& get_local_prev_activations(int parent_index = 0) const;
  const Mat& get_local_activations(int child_index = 0) const;
  const Mat& get_local_prev_error_signals(int child_index = 0) const;
  const Mat& get_local_error_signals(int parent_index = 0) const;

 protected:

  lbann_comm *m_comm;

  int m_num_neurons;                    ///< Number of neurons
  int m_num_neuron_dims;                ///< Number of dimensions in neuron tensor
  std::vector<int> m_neuron_dims;       ///< Neuron tensor dimensions
  int m_num_prev_neurons;               ///< Number of neurons in previous layer
  int m_num_prev_neuron_dims;           ///< Number of dimensions in previous layer's neuron tensor
  std::vector<int> m_prev_neuron_dims;  ///< Neuron tensor dimensions in previous layer

  /** Activation matrices from parent layers.
   *  Forward propagation inputs, one for each parent layer. These are
   *  typically matrix views where each column is a flattened tensor
   *  corresponding to a mini-batch sample.
   */
  std::vector<AbsDistMat*> m_prev_activations;
  /** Activation matrices.
   *  Forward propagation outputs, one for each child layer. These are
   *  typically matrices where each column is a flattened tensor
   *  corresponding to a mini-batch sample.
   */
  std::vector<AbsDistMat*> m_activations;
  /** Error signal matrices from child layers.
   *  Backward propagation inputs, one for each child layer. These
   *  should be the objective function gradients w.r.t. the forward
   *  propagation outputs. These are typically matrix views where each
   *  column is a flattened tensor corresponding to a mini-batch
   *  sample.
   */
  std::vector<AbsDistMat*> m_prev_error_signals;
  /** Error signal matrices.
   *  Backward propagation outputs, one for each parent layer. These
   *  should be the objective function gradients w.r.t. the forward
   *  propagation inputs. These are typically matrices where each
   *  column is a flattened tensor corresponding to a mini-batch
   *  sample.
   */
  std::vector<AbsDistMat*> m_error_signals;

  /** List of layer weights. */
  std::vector<weights*> m_weights;

  /** List of parent layers. */
  std::vector<const Layer*> m_parent_layers;
  /** List of child layers. */
  std::vector<const Layer*> m_child_layers;

  /** Expected number of parent layers.
   *  A negative value indicates no limit.
   */
  int m_expected_num_parent_layers;
  /** Expected number of child layers.
   *  A negative value indicates no limit.
   */
  int m_expected_num_child_layers;

  model *m_model;

  /** Setup views of the matrices for the layer's forward propagation. */
  virtual void fp_setup_data(int mini_batch_size);
  /** Setup views of the matrices for the layer's backward propagation. */
  virtual void bp_setup_data(int mini_batch_size);
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
  /** Setup distributed matrix objects. */
  virtual void setup_matrices(const El::Grid& grid);
  /** Setup layer data.
   *  Called by the setup function. This base method initializes the
   *  activations and error signal matrices.
   */
  virtual void setup_data();
  /** Setup matrix views.
   *  Called by the setup function.
   */
  virtual void setup_views() {}
  /** Setup GPU objects.
   *  Called by the setup function if GPUs are enabled. This base
   *  method initializes the activations and error signal matrices on
   *  GPUs.
   */
  virtual void setup_gpu();
  /** Perform the main computation for a forward propagation step. */
  virtual void fp_compute() = 0;
  /** Perform the main computation for a backward propagation step. */
  virtual void bp_compute() = 0;
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
  std::vector<cudnn::matrix> m_prev_activations_d;
  /** GPU memory for activations. */
  std::vector<cudnn::matrix> m_activations_d;
  /** GPU memory for error signal from "next" layer. */
  std::vector<cudnn::matrix> m_prev_error_signals_d;
  /** GPU memory for error signal. */
  std::vector<cudnn::matrix> m_error_signals_d;

  /** cuDNN descriptor for activations tensor from "previous" layer. */
  cudnnTensorDescriptor_t m_prev_activations_cudnn_desc;
  /** cuDNN descriptor for activations tensor. */
  cudnnTensorDescriptor_t m_activations_cudnn_desc;
  /** cuDNN descriptor for error signal tensor from "next" layer. */
  cudnnTensorDescriptor_t m_prev_error_signals_cudnn_desc;
  /** cuDNN descriptor for error signal tensor. */
  cudnnTensorDescriptor_t m_error_signals_cudnn_desc;

#endif // __LIB_CUDNN

  /** Time spent in forward propagation. */
  EvalType m_fp_time;
  /** Time spent in the forward propagation computation. */
  EvalType m_fp_compute_time;
  /** Time spent in backward propagation. */
  EvalType m_bp_time;
  /** Time spent in the backward propagation computation. */
  EvalType m_bp_compute_time;
  /** Time spent in updates. */
  EvalType m_update_time;

  std::string m_name;

 private:

  /** Deallocate distributed matrices. */
  void deallocate_matrices();

};
}

#endif // LBANN_LAYER_HPP_INCLUDED
