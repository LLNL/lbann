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
#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <data_layout T_layout>
class fully_connected_layer : public learning {
 private:
  weight_initialization m_weight_initialization;

  /// Views of the weight matrix that allow you to separate activation weights from bias weights
  ElMat *m_activation_weights_v;
  ElMat *m_bias_weights_v;
  ElMat *m_activation_weights_gradient_v;
  ElMat *m_bias_weights_gradient_v;

  /// Special matrices to allow backprop across the bias term
  ElMat *m_bias_weights_repl;
  ElMat *m_bias_weights_gradient_repl;
  DataType m_bias_scaling_factor;

  /**
   * Do layout-dependent forward propagation computation of the weights.
   */
  inline void fp_compute_weights();
  /**
   * Do layout-dependent backward propagation. This handles computing the error
   * signal for the next layer and the gradients for the weights.
   */
  inline void bp_compute_weights();

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

  fully_connected_layer(int index,
                        lbann_comm *comm,
                        int mini_batch_size,
                        int num_neurons,  // TODO: accept a vector for neuron dims
                        weight_initialization init,
                        optimizer *opt,
                        bool has_bias = true)
    : learning(index, comm, mini_batch_size, opt),
      m_weight_initialization(init) {

    // Setup the data distribution
    initialize_distributed_matrices();

    this->m_index = index;
    this->m_num_neurons = num_neurons;  // Set here since it's passed in.
    m_bias_scaling_factor = has_bias ? DataType(1) : DataType(0);
  }

  fully_connected_layer(const fully_connected_layer& other) :
    learning(other),
    m_weight_initialization(other.m_weight_initialization),
    m_bias_scaling_factor(other.m_bias_scaling_factor) {
    if (m_activation_weights_v) {
      delete m_bias_weights_repl;
      delete m_bias_weights_gradient_repl;
      delete m_activation_weights_v;
      delete m_activation_weights_gradient_v;
      delete m_bias_weights_v;
      delete m_bias_weights_gradient_v;
    }
    m_bias_weights_repl = other.m_bias_weights_repl->Copy();
    m_bias_weights_gradient_repl = other.m_bias_weights_gradient_repl->Copy();
    m_activation_weights_v = other.m_activation_weights_v->Copy();
    m_activation_weights_gradient_v = other.m_activation_weights_gradient_v->Copy();
    m_bias_weights_v = other.m_bias_weights_v->Copy();
    m_bias_weights_gradient_v = other.m_bias_weights_gradient_v->Copy();
  }

  fully_connected_layer& operator=(const fully_connected_layer& other) {
    learning::operator=(other);
    m_weight_initialization = other.m_weight_initialization;
    m_bias_scaling_factor = other.m_bias_scaling_factor;
    if (m_activation_weights_v) {
      delete m_bias_weights_repl;
      delete m_bias_weights_gradient_repl;
      delete m_activation_weights_v;
      delete m_activation_weights_gradient_v;
      delete m_bias_weights_v;
      delete m_bias_weights_gradient_v;
    }
    m_bias_weights_repl = other.m_bias_weights_repl->Copy();
    m_bias_weights_gradient_repl = other.m_bias_weights_gradient_repl->Copy();
    m_activation_weights_v = other.m_activation_weights_v->Copy();
    m_activation_weights_gradient_v = other.m_activation_weights_gradient_v->Copy();
    m_bias_weights_v = other.m_bias_weights_v->Copy();
    m_bias_weights_gradient_v = other.m_bias_weights_gradient_v->Copy();
    return *this;
  }

  ~fully_connected_layer() {
    delete m_bias_weights_repl;
    delete m_bias_weights_gradient_repl;
    delete m_activation_weights_v;
    delete m_activation_weights_gradient_v;
    delete m_bias_weights_v;
    delete m_bias_weights_gradient_v;
  }

  fully_connected_layer* copy() const {
    return new fully_connected_layer(*this);
  }

  std::string get_name() const { return "fully connected"; }

  virtual inline void initialize_distributed_matrices();
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_dims() {
    learning::setup_dims();
    this->m_num_neuron_dims = 1;
    this->m_neuron_dims.assign(1, this->m_num_neurons);
  }

  void setup_data() {
    learning::setup_data();
    // Initialize matrices
    // Note: the weights-bias matrix has an extra column so it includes bias term
    El::Zeros(*this->m_weights, this->m_num_neurons, this->m_num_prev_neurons+1);
    this->m_weights_gradient->Resize(this->m_num_neurons, this->m_num_prev_neurons + 1);

    /// Setup independent views of the weight matrix for the activations
    El::View(*this->m_activation_weights_v, *this->m_weights, ALL, IR(0, this->m_num_prev_neurons));

    /// Setup independent views of the weights gradient matrix for the activations
    El::View(*m_activation_weights_gradient_v, *this->m_weights_gradient, ALL, IR(0, this->m_num_prev_neurons));

    /// Setup independent views of the weights and gradient matrix for the bias terms
    El::View(*m_bias_weights_v, *this->m_weights, ALL, IR(this->m_num_prev_neurons));
    El::View(*m_bias_weights_gradient_v, *this->m_weights_gradient, ALL, IR(this->m_num_prev_neurons));

    /// Initialize the activations part of the weight matrix -- leave the bias term weights zero
    initialize_matrix(*this->m_activation_weights_v, m_weight_initialization, this->m_num_prev_neurons, this->m_num_neurons);

    // Initialize optimizer
    if (this->m_optimizer != NULL) {
      this->m_optimizer->setup(this->m_weights);
    }
  }

  void fp_compute() {
    // Apply weight matrix
    fp_compute_weights();

    // Apply bias if needed
    if(m_bias_scaling_factor != DataType(0)) {
      El::Copy(*m_bias_weights_v, *m_bias_weights_repl);
      const Mat& local_bias_weights = m_bias_weights_repl->Matrix();
      El::IndexDependentMap(this->m_activations_v->Matrix(),
                            (std::function<DataType(El::Int,El::Int,const DataType&)>)
                            ([this,&local_bias_weights](El::Int r, El::Int c,const DataType& z)->DataType {
                              return z + m_bias_scaling_factor * local_bias_weights.Get(r, 0);
                            }));
    }
  }

  void bp_compute() {
    // Compute the error signal and gradients.
    bp_compute_weights();

    // Compute bias update if needed
    if(m_bias_scaling_factor != DataType(0)) {
      El::RowSum(*this->m_prev_error_signal_v,
                 *m_bias_weights_gradient_repl);
      El::Scale(m_bias_scaling_factor / this->get_effective_minibatch_size(),
                *m_bias_weights_gradient_v);
      El::Copy(*m_bias_weights_gradient_repl, *m_bias_weights_gradient_v);
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

  bool update_compute() {
    if(this->m_execution_mode == execution_mode::training) {
      this->l2_regularize();
      this->m_optimizer->update(this->m_weights_gradient);
    }
    return true;
  }

};

/// Matrices should be in MC,MR distributions
template<> inline void fully_connected_layer<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  learning::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_bias_weights_repl = new El::DistMatrix<DataType,MC,STAR>(this->m_comm->get_model_grid());
  m_bias_weights_gradient_repl = new El::DistMatrix<DataType,MC,STAR>(this->m_comm->get_model_grid());

  /// Instantiate these view objects but do not allocate data for them
  m_activation_weights_v           = new DistMat(this->m_comm->get_model_grid());
  m_activation_weights_gradient_v  = new DistMat(this->m_comm->get_model_grid());
  m_bias_weights_v                 = new DistMat(this->m_comm->get_model_grid());
  m_bias_weights_gradient_v        = new DistMat(this->m_comm->get_model_grid());
}

/// Weight matrices should be in Star,Star and data matrices Star,VC distributions
template<> inline void fully_connected_layer<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  learning::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_bias_weights_repl = new StarMat(this->m_comm->get_model_grid());
  m_bias_weights_gradient_repl = new StarMat(this->m_comm->get_model_grid());

  /// Instantiate these view objects but do not allocate data for them
  m_activation_weights_v           = new StarMat(this->m_comm->get_model_grid());
  m_activation_weights_gradient_v  = new StarMat(this->m_comm->get_model_grid());
  m_bias_weights_v                 = new StarMat(this->m_comm->get_model_grid());
  m_bias_weights_gradient_v        = new StarMat(this->m_comm->get_model_grid());
}

template<> inline void
fully_connected_layer<data_layout::MODEL_PARALLEL>::fp_compute_weights() {
  El::Gemm(NORMAL, NORMAL, DataType(1),
           *this->m_activation_weights_v,
           *this->m_prev_activations_v,
           DataType(0),
           *this->m_activations_v);
}

template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::fp_compute_weights() {
  El::Gemm(NORMAL, NORMAL, DataType(1),
           this->m_activation_weights_v->LockedMatrix(),
           this->m_prev_activations_v->LockedMatrix(),
           DataType(0),
           this->m_activations_v->Matrix());
}

template<> inline void
fully_connected_layer<data_layout::MODEL_PARALLEL>::bp_compute_weights() {
  // Compute the partial delta update for the next lower layer
  El::Gemm(El::TRANSPOSE, El::NORMAL, DataType(1),
           *this->m_activation_weights_v,
           *this->m_prev_error_signal_v,
           DataType(0),
           *this->m_error_signal_v);

  // Compute update for activation weights
  El::Gemm(El::NORMAL, El::TRANSPOSE, DataType(1)/this->get_effective_minibatch_size(),
           *this->m_prev_error_signal_v,
           *this->m_prev_activations_v,
           DataType(0),
           *this->m_activation_weights_gradient_v);
}

template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::bp_compute_weights() {
  El::Gemm(El::TRANSPOSE, El::NORMAL, DataType(1),
           this->m_activation_weights_v->LockedMatrix(),
           this->m_prev_error_signal_v->LockedMatrix(),
           DataType(0),
           this->m_error_signal_v->Matrix());

  // Compute update for activation weights
  El::Gemm(El::NORMAL, El::TRANSPOSE, DataType(1)/this->get_effective_minibatch_size(),
           this->m_prev_error_signal_v->LockedMatrix(),
           this->m_prev_activations_v->LockedMatrix(),
           DataType(0),
           this->m_activation_weights_gradient_v->Matrix());
  El::AllReduce(*this->m_activation_weights_gradient_v,
                this->m_activation_weights_gradient_v->RedundantComm());
}

}  // namespace lbann

#endif  // LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
