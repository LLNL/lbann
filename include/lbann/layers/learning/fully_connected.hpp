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
// lbann_layer_fully_connected .hpp .cpp - Dense, fully connected, layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
#define LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED

#include "lbann/layers/learning/learning.hpp"
#include "lbann/layers/activations/activations.hpp"
#include "lbann/utils/lbann_random.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <class T_layout>
class fully_connected_layer : public learning<T_layout> {
 private:

  const weight_initialization m_weight_initialization;

  /// Views of the weight matrix that allow you to separate activation weights from bias weights
  ElMat *m_activation_weights_v;
  ElMat *m_bias_weights_v;
  ElMat *m_activation_weights_gradient_v;
  ElMat *m_bias_weights_gradient_v;

  /// Special matrices to allow backprop across the bias term
  ElMat *m_bias_bp_t;
  ElMat *m_bias_bp_t_v;
  ElMat *m_bias_weights_repl;
  DataType m_bias_term;

 protected:
  //Probability of dropping neuron/input used in dropout_layer
  //Range 0 to 1; default is -1 => no dropout
  DataType  WBL2NormSum;

 public:
  ////////////////////////////////////////////////////////////////////////////////
  // fully_connected_layer : single network layer class
  ////////////////////////////////////////////////////////////////////////////////
  // WB structure: (num units "neurons / filters" x (num features + 1))
  // Each row represents a neuron / filter
  // There is a column for each feature coming in from the previous layer plus 1 for the bias
  // [W00 ...   B0]
  // [|         |]
  // [Wn0       Bn]
  //
  // WB_D structure:
  // [dW     dB]
  // D structure:
  // [D        ]
  // Z, Zs, Act, Acts structure:
  // [Acts     ]

  fully_connected_layer(T_layout data_dist,
                      const uint index,
                      const int numPrevNeurons,
                      const uint numNeurons,
                      const uint mini_batch_size,
                      const weight_initialization init,
                      lbann_comm *comm,
                        optimizer *opt)
    : learning<T_layout>(data_dist,
                         index, numPrevNeurons, 
                         numNeurons, mini_batch_size, 
                         comm, opt),
    m_weight_initialization(init) {

    this->m_type = layer_type::fully_connected;

    this->m_index = index;
    this->m_num_neurons = numNeurons;
    WBL2NormSum = 0.0;
    m_bias_term = 1.0;

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
  }

  ~fully_connected_layer(void) {
    delete m_bias_bp_t;
    delete m_bias_weights_repl;
    delete m_activation_weights_v;
    delete m_bias_weights_v;
    delete m_activation_weights_gradient_v;
    delete m_bias_weights_gradient_v;
    delete m_bias_bp_t_v;
  }

  /// Matrices should be in MC,MR distributions
  void initialize_model_parallel_distribution(void) {
    learning<T_layout>::initialize_model_parallel_distribution();
    m_bias_bp_t                      = new DistMat(this->m_comm->get_model_grid());
    m_bias_weights_repl              = new DistMatrix<DataType,MC,STAR>(this->m_comm->get_model_grid());

    /// Instantiate these view objects but do not allocate data for them
    m_activation_weights_v           = new DistMat(this->m_comm->get_model_grid());
    m_bias_weights_v                 = new DistMat(this->m_comm->get_model_grid());
    m_activation_weights_gradient_v  = new DistMat(this->m_comm->get_model_grid());
    m_bias_weights_gradient_v        = new DistMat(this->m_comm->get_model_grid());
    m_bias_bp_t_v                    = new DistMat(this->m_comm->get_model_grid());
  }

  /// Weight matrices should be in Star,Star and data matrices Star,VC distributions
  void initialize_data_parallel_distribution(void) {
    learning<T_layout>::initialize_data_parallel_distribution();
    m_bias_bp_t                      = new StarVCMat(this->m_comm->get_model_grid());
    m_bias_weights_repl              = new StarMat(this->m_comm->get_model_grid());

    /// Instantiate these view objects but do not allocate data for them
    m_activation_weights_v           = new StarMat(this->m_comm->get_model_grid());
    m_bias_weights_v                 = new StarMat(this->m_comm->get_model_grid());
    m_activation_weights_gradient_v  = new StarMat(this->m_comm->get_model_grid());
    m_bias_weights_gradient_v        = new StarMat(this->m_comm->get_model_grid());
    m_bias_bp_t_v                    = new StarVCMat(this->m_comm->get_model_grid());
  }

  void setup(int numPrevNeurons) {
    Layer::setup(numPrevNeurons);

    // Initialize matrices
    // Note: the weights-bias matrix has an extra column so it includes bias term
    Zeros(*this->m_weights, this->m_num_neurons, numPrevNeurons+1);
    Zeros(*this->m_weights_gradient, this->m_num_neurons, numPrevNeurons + 1);
    Zeros(*this->m_prev_activations, numPrevNeurons, this->m_mini_batch_size);
    Zeros(*this->m_weighted_sum, this->m_num_neurons, this->m_mini_batch_size);
    Zeros(*this->m_activations, this->m_num_neurons, this->m_mini_batch_size);
    Zeros(*this->m_prev_error_signal, this->m_num_neurons, this->m_mini_batch_size);
    Zeros(*this->m_error_signal, numPrevNeurons, this->m_mini_batch_size); // m_error_signal holds the product of m_weights^T * m_prev_error_signal

    /// Setup independent views of the weight matrix for the activations and bias terms
    View(*this->m_activation_weights_v, *this->m_weights, ALL, IR(0, numPrevNeurons));
    View(*m_bias_weights_v, *this->m_weights, ALL, IR(numPrevNeurons));

    /// Setup independent views of the weights gradient matrix for the activations and bias terms
    View(*m_activation_weights_gradient_v, *this->m_weights_gradient, ALL, IR(0, numPrevNeurons));
    View(*m_bias_weights_gradient_v, *this->m_weights_gradient, ALL, IR(numPrevNeurons));

    /// Initialize the activations part of the weight matrix -- leave the bias term weights zero
    initialize_matrix(*this->m_activation_weights_v, m_weight_initialization, numPrevNeurons, this->m_num_neurons);

    /// Create a "transposed" vector of the bias term for use in backprop
    Ones(*m_bias_bp_t, 1, this->m_mini_batch_size);

    // Initialize optimizer
    if(this->m_optimizer != NULL) {
      this->m_optimizer->setup(this->m_weights);
    }

  }

  void fp_set_std_matrix_view(void) {
    int64_t cur_mini_batch_size = this->m_neural_network_model->get_current_mini_batch_size();

    Layer::fp_set_std_matrix_view();

    /// Note that the view of the bias backprop term is transposed, so the current mini-batch size is used to
    /// limit the height, not the width
    View(*m_bias_bp_t_v, *m_bias_bp_t, ALL, IR(0, cur_mini_batch_size));
  }

  void fp_compute(void) {
    // Apply forward prop linearity

    // Apply bias
    Copy(*m_bias_weights_v, *m_bias_weights_repl);
    const Mat& local_bias_weights = m_bias_weights_repl->Matrix();
    IndexDependentFill(this->m_weighted_sum_v->Matrix(), (std::function<DataType(El::Int,El::Int)>)
                       ([&local_bias_weights](El::Int r, El::Int c)->DataType {
                         return local_bias_weights.Get(r);
                       }));
    Scale(m_bias_term, *this->m_weighted_sum_v);

    // Apply weight matrix
    switch(this->m_data_layout) {
    case data_layout::MODEL_PARALLEL:
      Gemm(NORMAL, NORMAL, DataType(1),
           *this->m_activation_weights_v,
           *this->m_prev_activations_v,
           DataType(1),
           *this->m_weighted_sum_v);
      break;
    case data_layout::DATA_PARALLEL:
      Gemm(NORMAL, NORMAL, DataType(1),
           this->m_activation_weights_v->LockedMatrix(),
           this->m_prev_activations_v->LockedMatrix(),
           DataType(1),
           this->m_weighted_sum_v->Matrix());
      break;
    }

    // Copy result to output matrix
    Copy(*this->m_weighted_sum_v, *this->m_activations_v);

  }

  void bp_compute(void) {

    switch(this->m_data_layout) {
    case data_layout::MODEL_PARALLEL:
      // Compute the partial delta update for the next lower layer
      Gemm(TRANSPOSE, NORMAL, DataType(1),
           *this->m_activation_weights_v,
           *this->m_prev_error_signal_v,
           DataType(0),
           *this->m_error_signal_v);
      // Compute update for activation weights
      Gemm(NORMAL, TRANSPOSE, DataType(1)/this->get_effective_minibatch_size(),
           *this->m_prev_error_signal_v,
           *this->m_prev_activations_v,
           DataType(0),
           *this->m_activation_weights_gradient_v);
      // Compute update for bias terms
      Gemv(NORMAL, DataType(1)/this->get_effective_minibatch_size(),
           *this->m_prev_error_signal_v,
           *m_bias_bp_t_v,
           DataType(0),
           *m_bias_weights_gradient_v);
      break;
    case data_layout::DATA_PARALLEL:
      // Compute the partial delta update for the next lower layer
      Gemm(TRANSPOSE, NORMAL, DataType(1),
           this->m_activation_weights_v->LockedMatrix(),
           this->m_prev_error_signal_v->LockedMatrix(),
           DataType(0),
           this->m_error_signal_v->Matrix());
      // Compute update for activation weights
      Gemm(NORMAL, TRANSPOSE, DataType(1)/this->get_effective_minibatch_size(),
           this->m_prev_error_signal_v->LockedMatrix(),
           this->m_prev_activations_v->LockedMatrix(),
           DataType(0),
           this->m_activation_weights_gradient_v->Matrix());
      // Compute update for bias terms
      Gemv(NORMAL, DataType(1)/this->get_effective_minibatch_size(),
           this->m_prev_error_signal_v->LockedMatrix(),
           m_bias_bp_t_v->LockedMatrix(),
           DataType(0),
           m_bias_weights_gradient_v->Matrix());
      // Add gradients from all processes
      AllReduce(*this->m_weights_gradient,
                this->m_weights_gradient->RedundantComm());
      break;
    }

  }

  DataType computeCost(DistMat& deltas) {
    DataType avg_error = 0.0, total_error = 0.0;
    // Compute the L2 norm on the deltas (activation - y)
    ColSumMat norms;
    ColumnTwoNorms(deltas, norms);
    int c = 0;
    // Sum the local, total error
    for(int r = 0; r < norms.LocalHeight(); r++) {
      total_error += norms.GetLocal(r,c);
    }
    mpi::AllReduce(total_error, norms.DistComm());
    avg_error = total_error / norms.Height();
    return avg_error;
  }

  inline DataType _sq(DataType x) {
    return (x * x);
  }
  inline DataType _sqrt(DataType x) {
    return (1 / sqrt(x + 1e-8));
  }

  bool update_compute(void) {
    double start = get_time();
    Layer::update();
    if(this->m_execution_mode == execution_mode::training) {
      this->m_optimizer->update(this->m_weights_gradient);
    }
    this->update_time += get_time() - start;
    return true;
  }

};

}


#endif // LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
