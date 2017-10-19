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
// lbann_dropout .cpp .hpp - Dropout implementation
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"

namespace lbann {

/**
 * Dropout: probabilistically drop units from a layer.
 * See this paper for full details:
 * Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks
 * from overfitting." Journal of Machine Learning Research 15.1 (2014).
 * This implementation uses the approach noted in section 10 of that paper of
 * multiplying weights by 1/(keep probability) at training time and not
 * modifying them at test time.
 * The implementation recommends a keep probability of 0.5 for fully-connected
 * layers and 0.8 for input layers as good starting points.
 */
template <data_layout T_layout>
class dropout : public regularizer_layer {
 public:
  /** Keep units with probabiliy keep_prob. */
  dropout(int index,
          lbann_comm *comm,
          float keep_prob=0.5f) :
    regularizer_layer(index, comm),
    m_keep_prob(keep_prob) {
    // Setup the data distribution
    initialize_distributed_matrices();
  }

  dropout(const dropout& other) :
    regularizer_layer(other),
    m_keep_prob(other.m_keep_prob) {
    m_mask = other.m_mask->Copy();
  }

  dropout& operator=(const dropout& other) {
    regularizer_layer::operator=(other);
    m_keep_prob = other.m_keep_prob;
    if(m_mask) {
      delete m_mask;
    }
    m_mask = other.m_mask->Copy();
    return *this;
  }

  ~dropout() {
    delete m_mask;
  }

  dropout* copy() const { return new dropout(*this); }

  std::string get_type() const { return "dropout"; }

  std::string get_description() const {
    return " dropout keep_prob: " + std::to_string(m_keep_prob) 
           + " dataLayout: " + get_data_layout_string(get_data_layout());
  }

  virtual inline void initialize_distributed_matrices();
  virtual data_layout get_data_layout() const { return T_layout; }

 protected:
  /** Drop out units in forward propagation. */
  void fp_compute() {

    // Copy previous activations if dropout is disabled
    if (this->get_execution_mode() != execution_mode::training
        || m_keep_prob < 0.0f) {
      El::Copy(*this->m_prev_activations, *this->m_activations_v);
      return;
    }

    // Construct mask matrix
    const DataType scale = DataType(1) / m_keep_prob;
    const int height = this->m_activations_v->Height();
    const int width = this->m_activations_v->Width();
    m_mask->Resize(height, width);
#ifdef LBANN_PROCDET_DROPOUT
    bernoulli_fill_procdet(*m_mask, height, width, m_keep_prob);
    *m_mask *= scale;
#else
    El::EntrywiseMap(*m_mask,
                     (std::function<DataType(const DataType&)>)
                     ([this,scale](const DataType& z)->DataType {
                       auto& gen = get_fast_generator();
                       std::bernoulli_distribution dist(m_keep_prob);
                       return dist(gen) ? scale : DataType(0);
                     }));
#endif // LBANN_PROCDET_DROPOUT

    // Apply mask matrix to get activations
    El::Hadamard(*this->m_prev_activations, *m_mask, *this->m_activations_v);

  }

  /** Adjust gradients for dropout in backprop. */
  void bp_compute() {

    // Copy previous error signal if dropout is disabled
    if (this->get_execution_mode() != execution_mode::training
        || m_keep_prob < 0.0f) {
      El::Copy(*this->m_prev_error_signal, *this->m_error_signal_v);
      return;
    }

    // Apply mask matrix to error signal
    El::Hadamard(*this->m_prev_error_signal, *m_mask, *this->m_error_signal_v);

  }

  /** Probability of keeping each unit. */
  float m_keep_prob;
  /** Current dropout mask (a scaled Bernoulli random matrix). */
  AbsDistMat *m_mask;

};

template<> inline void dropout<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  regularizer_layer::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_mask = new DistMat(m_comm->get_model_grid());
}

template<> inline void dropout<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  regularizer_layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_mask = new StarVCMat(m_comm->get_model_grid());
}

}  // namespace lbann

#endif  // LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
