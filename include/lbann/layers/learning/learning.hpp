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

#ifndef LBANN_LAYER_LEARNING_HPP_INCLUDED
#define LBANN_LAYER_LEARNING_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include <string>
#include <vector>

namespace lbann {

class learning : public Layer {
 protected:
  //  data_layout m_data_layout;
  optimizer  *m_optimizer;

  ElMat *m_weights;             ///< Weight matrix (computes weight sum of inputs ((# neurons) x (# previous layer's neurons))
  ElMat *m_weights_gradient;    ///< Gradient w.r.t. weight matrix ((# neurons) x (# previous layer's neurons))

  /** Factor for L2 regularization; 0 to disable. */
  DataType m_l2_regularization_factor = DataType(0);

  /** Apply L2 regularization to the current gradient. */
  virtual void l2_regularize() {
    if (m_l2_regularization_factor > DataType(0)) {
      El::Axpy(m_l2_regularization_factor, *m_weights, *m_weights_gradient);
    }
  }

 public:
  learning(int index, 
           int numPrevNeurons,
           int numNeurons,
           int mini_batch_size,
           lbann_comm *comm,
           optimizer *opt
           )
    : Layer(index, comm, mini_batch_size), m_optimizer(opt) { 
    }

  virtual ~learning(void) {
    delete m_weights;
    delete m_weights_gradient;
  }

  /** Return the weights associated with this layer. */
  virtual ElMat& get_weights() const { return *m_weights; }
  /** Return the gradients associated with this layer. */
  virtual ElMat& get_weights_gradient() const { return *m_weights_gradient; }

  template <data_layout T_layout>
  inline void initialize_distributed_matrices();

  /// @todo BVE should the learning layer be able to initialize the
  /// matrix, or is that purely a function of the children classes
  //enum class weight_initialization {zero, uniform, normal, glorot_normal, glorot_uniform, he_normal, he_uniform};
  static std::string weight_initialization_name(weight_initialization id) {
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
      throw lbann_exception(
        std::string(__FILE__) + " " + std::to_string(__LINE__) + " :: "
        "unknown weight_initialization: " + std::to_string((int) id));
    }
  }

  virtual void setup(int numPrevNeurons) {
    Layer::setup(numPrevNeurons);
#if 0
    // Zero the weight-bias matrix
    Zeros(*m_weights, m_num_neurons, numPrevNeurons);

    /// Initialize the activations part of the weight matrix -- leave the bias term weights zero
    initialize_matrix(*m_weights, m_weight_initialization, numPrevNeurons, m_num_neurons);

    // Initialize other matrices
    Zeros(*m_weights_gradient, m_num_neurons, numPrevNeurons);
#endif
  }

  virtual void summarize(lbann_summary& summarizer, int step) {
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

    // Call parent summarizer after local results are summarized
    Layer::summarize(summarizer, step);
  }

  /** Validate that the setup is reasonable. */
  virtual void check_setup() {
    Layer::check_setup();
    // If these two are sendable, the other matrices should be fine.
    if (!lbann::lbann_comm::is_sendable(*m_weights)) {
      throw lbann::lbann_exception("Weights too large to send");
    }
    if (!lbann::lbann_comm::is_sendable(*m_activations)) {
      throw lbann::lbann_exception("Activations too large to send");
    }
  }

  /** Return (a view of) the weights/biases matrix for this layer. */
  virtual ElMat& get_weights_biases(void) {
    return *m_weights;
  }
  /** Return (a view of) the weights/biases gradient matrix for this layer. */
  virtual ElMat& get_weights_biases_gradient(void) {
    return *m_weights_gradient;
  }
  /** Return the layer's optimizer. */
  virtual optimizer *get_optimizer(void) const {
    return m_optimizer;
  }

  /** Set the layer's L2 regularization factor (0 to disable). */
  void set_l2_regularization_factor(DataType f) {
    m_l2_regularization_factor = f;
  }

  bool saveToFile(int fd, const char *dirname) {
    Layer::loadFromFile(fd, dirname);
    char filepath[512];
    sprintf(filepath, "%s/weights_L%d_%03lldx%03lld", dirname, m_index, m_weights->Height()-1, m_weights->Width()-1);
    
    uint64_t bytes;
    return lbann::write_distmat(-1, filepath, (DistMat *)m_weights, &bytes);
  }
  
  bool loadFromFile(int fd, const char *dirname) {
    Layer::loadFromFile(fd, dirname);
    char filepath[512];
    sprintf(filepath, "%s/weights_L%d_%03lldx%03lld.bin", dirname, m_index, m_weights->Height()-1, m_weights->Width()-1);
    
    uint64_t bytes;
    return lbann::read_distmat(-1, filepath, (DistMat *)m_weights, &bytes);
  }

  virtual bool saveToCheckpointShared(lbann::persist& p) {
    Layer::saveToCheckpointShared(p);
    // define name to store our parameters
    char name[512];
    sprintf(name, "weights_L%d_%lldx%lld", m_index, m_weights->Height(), m_weights->Width());

    // write out our weights to the model file
    p.write_distmat(persist_type::model, name, (DistMat *)m_weights);

    // if saving training state, also write out state of optimizer
    // m_optimizer->saveToCheckpointShared(p, m_index);

    return true;
  }

  virtual bool loadFromCheckpointShared(lbann::persist& p) {
    Layer::loadFromCheckpointShared(p);
    // define name to store our parameters
    char name[512];
    sprintf(name, "weights_L%d_%lldx%lld.bin", m_index, m_weights->Height(), m_weights->Width());

    // read our weights from model file
    p.read_distmat(persist_type::model, name, (DistMat *)m_weights);

    // if loading training state, read in state of optimizer
    // m_optimizer->loadFromCheckpointShared(p, m_index);

    return true;
  }


 protected:

  /** Setup views of the matrices for the layer's forward propagation. */
  virtual void fp_set_std_matrix_view(void) {
    Layer::fp_set_std_matrix_view();
  }

#if 0
  /** Setup views of the matrices for the layer's backward propagation. */
  virtual void bp_set_std_matrix_view(void);
#endif

};

/// Matrices should be in MC,MR distributions
template<> inline void learning::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>() {
  Layer::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_weights             = new DistMat(m_comm->get_model_grid());
  m_weights_gradient    = new DistMat(m_comm->get_model_grid());
}

/// Weight matrices should be in Star,Star and data matrices Star,VC distributions
template<> inline void learning::initialize_distributed_matrices<data_layout::DATA_PARALLEL>() {
  Layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_weights             = new StarMat(m_comm->get_model_grid());
  m_weights_gradient    = new StarMat(m_comm->get_model_grid());
}

}

#endif // LBANN_LAYER_LEARNING_HPP_INCLUDED
