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
  Layer(lbann_comm *comm);
  Layer(const Layer& other);
  Layer& operator=(const Layer& other);

  virtual ~Layer();

  virtual Layer* copy() const = 0;

  template <data_layout T_layout>
  void initialize_distributed_matrices();

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
  virtual void clear_error_signal();

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
    for (size_t h=0; h<this->m_neuron_dims.size(); h++) {
      if (h == 0) { s << "Acts=["; }
      s << this->m_neuron_dims[h] ;
      if (h == 0 && this->m_neuron_dims.size() > 1) { s << "c x "; }
      if (this->m_neuron_dims.size() == 2) {
        if (h == 1) { s << "w "; }
      }else if (this->m_neuron_dims.size() == 3) {
        if (h == 1) { s << "w x "; }
        if (h == 2) { s << "h"; }
      }else {
        if (h > 1) {
          s << " ";
        }
      }
    }
    s << ", " << m_activations->Width() << "s]";
    return s.str();;
  }

  /** Returns a string description of the data_layout */
  std::string get_data_layout_string(data_layout d) const;

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
  /** Return the data layout of the given layer -- Every concrete
      layer has to overrride this with its T_layout template parameter */
  virtual data_layout get_data_layout() const = 0;
  /** Return (a view of) the activations matrix for this layer. */
  virtual AbsDistMat& get_activations() const {
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

  /** Get maximum number of parent layers.
   *  A negative value indicates no limit.
   */
  inline int get_max_num_parent_layers() const {
    return m_max_num_parent_layers;
  }
  /** Get maximum number of child layers.
   *  A negative value indicates no limit.
   */
  inline int get_max_num_child_layers() const {
    return m_max_num_child_layers;
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

  /** Get forward propagation output, as seen by next layer. */
  virtual void get_fp_output(AbsDistMat& fp_output, const Layer* next_layer = nullptr) const;
  /** Get backward propagation output, as seen by previous layer. */
  virtual void get_bp_output(AbsDistMat& fp_output, const Layer* prev_layer = nullptr) const;
#ifdef LBANN_HAS_CUDNN
  /** Get forward propagation output on GPUs, as seen by next layer.
   *  output_dv is a view into GPU memory for the output. If the
   *  output cannot be represented as a view, the data is copied into
   *  output_d and output_dv is set as a view into it.
   */
  virtual void get_gpu_fp_output(std::vector<DataType*>& output_dv,
                                 std::vector<DataType*>& output_d,
                                 const Layer* next_layer = NULL) const;
  /** Get backward propagation output on GPUs, as seen by previous layer.
   *  output_dv is a view into GPU memory for the output. If the
   *  output cannot be represented as a view, the data is copied into
   *  output_d and output_dv is set as a view into it.
   */
  virtual void get_gpu_bp_output(std::vector<DataType*>& output_dv,
                                 std::vector<DataType*>& output_d,
                                 const Layer* prev_layer = NULL) const;
#endif // LBANN_HAS_CUDNN
  /** Get forward propagation output dimensions, as seen by next layer. */
  virtual const std::vector<int> fp_output_dims(const Layer* next_layer = nullptr) const;

  virtual void add_to_error_signal(const AbsDistMat& gradient,
                                   DataType scale = DataType(1)) {
    bp_set_std_matrix_view();
    El::Axpy(scale, gradient, *m_error_signal_v);
  }

  /** Get list of parent layers. */
  std::vector<const Layer*>& get_parent_layers();
  /** Get list of parent layers (const). */
  const std::vector<const Layer*>& get_parent_layers() const;
  /** Get list of child layers. */
  std::vector<const Layer*>& get_child_layers();
  /** Get list of child layers (const). */
  const std::vector<const Layer*>& get_child_layers() const;
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

 protected:

  lbann_comm *m_comm;

  int m_num_neurons;                    ///< Number of neurons
  int m_num_neuron_dims;                ///< Number of dimensions in neuron tensor
  std::vector<int> m_neuron_dims;       ///< Neuron tensor dimensions
  int m_num_prev_neurons;               ///< Number of neurons in previous layer
  int m_num_prev_neuron_dims;           ///< Number of dimensions in previous layer's neuron tensor
  std::vector<int> m_prev_neuron_dims;  ///< Neuron tensor dimensions in previous layer

  /** Activations matrix from the "previous" layer.
   *  This matrix is the forward propagation input. This is typically
   *  a matrix view with dimensions m_num_prev_neurons x mini-batch
   *  size.
   */
  AbsDistMat* m_prev_activations_v;
  /** Memory for activations matrix.
   *  This matrix has dimensions m_num_neurons x max mini-batch size.
   */
  AbsDistMat* m_activations;
  /** Activations matrix.
   *  This matrix is the forward propagation output. This is typically
   *  a matrix view into m_activations with dimensions m_num_neurons x
   *  mini-batch size.
   */
  AbsDistMat* m_activations_v;
  /** Error signal matrix from the "next" layer.
   *  This matrix is the backward propagation input. This is typically
   *  a matrix view with dimensions m_num_neurons x mini-batch size.
   */
  AbsDistMat* m_prev_error_signal_v;
  /** Memory for error signal matrix.
   *  This matrix has dimensions m_num_prev_neurons x max mini-batch
   *  size.
   */
  AbsDistMat* m_error_signal;
  /** Error signal matrix.
   *  This matrix is the backward propagation output. This is
   *  typically a matrix view into m_error_signal with dimensions
   *  m_num_prev_neurons x mini-batch size.
   */
  AbsDistMat *m_error_signal_v;

  /** List of layer weights. */
  std::vector<weights*> m_weights;

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

  model *m_model;

  /** Setup views of the matrices for the layer's forward propagation. */
  virtual void fp_set_std_matrix_view();
  /** Setup views of the matrices for the layer's backward propagation. */
  virtual void bp_set_std_matrix_view();
#ifdef LBANN_HAS_CUDNN
  /** Pin host memory if needed for GPU memory transfers. */
  virtual void pin_data();
#endif // LBANN_HAS_CUDNN

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

#ifdef LBANN_HAS_CUDNN

  /** Number of mini-batch samples per GPU. */
  int m_mini_batch_size_per_gpu;
  /** Maximum number of mini-batch samples per GPU. */
  int m_max_mini_batch_size_per_gpu;

  /** GPU memory for activations from "previous" layer. */
  std::vector<DataType*> m_prev_activations_d;
  /** View into GPU memory for activations from "previous" layer. */
  std::vector<DataType*> m_prev_activations_dv;
  /** GPU memory for activations. */
  std::vector<DataType*> m_activations_d;
  /** GPU memory for error signal from "next" layer. */
  std::vector<DataType*> m_prev_error_signal_d;
  /** View into GPU memory for error signal from "next" layer. */
  std::vector<DataType*> m_prev_error_signal_dv;
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

#endif // LBANN_HAS_CUDNN

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
