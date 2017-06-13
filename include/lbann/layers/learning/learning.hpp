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

template <class T_layout>
class learning : public Layer {
#if 0
 protected:
  data_layout m_data_layout;

  ElMat *m_weights;             ///< Weight matrix (computes weight sum of inputs ((# neurons) x (# previous layer's neurons))
  ElMat *m_weights_gradient;    ///< Gradient w.r.t. weight matrix ((# neurons) x (# previous layer's neurons))
  ElMat *m_weighted_sum;        ///< Weighted sum - Output of forward pass linear transformation ((# neurons) x mini-batch size)

 public:
  /// Create a view of each matrix so that it can accomodate partial mini-batches
  ElMat *m_weighted_sum_v;
  ElMat *m_prev_error_signal_v;
  ElMat *m_activations_v;
#endif

 public:
  learning(data_layout data_dist, const uint index, 
           const int numPrevNeurons,
           const uint numNeurons,
           const uint mini_batch_size,
           lbann_comm *comm, optimizer *opt
           )
    : Layer(data_dist, index, comm, opt, mini_batch_size) {

#if 0
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
#endif
  }

#if 0
  virtual ~learning(void);

  static std::string weight_initialization_name(weight_initialization id);

  virtual void initialize_model_parallel_distribution(void);
  virtual void initialize_data_parallel_distribution(void);

  virtual void forwardProp(void);
  virtual void backProp(void);
  virtual bool update(void);
  virtual void summarize(lbann_summary& summarizer, int64_t step);

  virtual void setup(int);
  /** Validate that the setup is reasonable. */
  virtual void check_setup(void);

  /** Return (a view of) the weights/biases matrix for this layer. */
  virtual ElMat& get_weights_biases(void) {
    return *m_weights;
  }
  /** Return (a view of) the weights/biases gradient matrix for this layer. */
  virtual ElMat& get_weights_biases_gradient(void) {
    return *m_weights_gradient;
  }

  ElMat *fp_output(void);
  ElMat *bp_output(void);
  void setup_fp_input(ElMat *fp_input);
  void setup_bp_input(ElMat *bp_input);

  void set_prev_layer_type(layer_type type);
  void set_next_layer_type(layer_type type);
  bool using_gpus(void) const;
  void set_prev_layer_using_gpus(bool using_gpus);
  void set_next_layer_using_gpus(bool using_gpus);
#ifdef __LIB_CUDNN
  std::vector<DataType *> *fp_output_d(void);
  std::vector<DataType *> *bp_output_d(void);
  void setup_fp_input_d(std::vector<DataType *> *fp_input_d);
  void setup_bp_input_d(std::vector<DataType *> *bp_input_d);
#endif

#endif
#if 0
 protected:

  /** Setup views of the matrices for the layer's forward propagation. */
  virtual void fp_set_std_matrix_view(void);
#if 0
  /** Setup views of the matrices for the layer's backward propagation. */
  virtual void bp_set_std_matrix_view(void);
#endif
  /** Apply the layer's linear update in forward propagation. */
  virtual void fp_linearity(void) {}
  /** Handle the layer's linearity in backward propagation. */
  virtual void bp_linearity(void) {}
  /** Apply the layer's nonlinearity in forward propagation. */
  virtual void fp_nonlinearity(void);
  /** Handle the layer's nonlinearity in backward propagation. */
  virtual void bp_nonlinearity(void);
#endif
};
}


#endif // LBANN_LAYER_LEARNING_HPP_INCLUDED
