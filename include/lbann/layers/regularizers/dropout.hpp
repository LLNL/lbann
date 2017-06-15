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
  dropout(const uint index, const uint num_neurons, lbann_comm *comm,
          uint mini_batch_size, float keep_prob=0.5f) :
    regularizer_layer(index, comm, mini_batch_size),
    m_keep_prob(keep_prob) {
    // Setup the data distribution
    initialize_distributed_matrices();
    this->m_type = layer_type::dropout;
    this->m_num_neurons = num_neurons;
  }
  ~dropout() {
    delete m_cur_mask;
  }

  virtual inline void initialize_distributed_matrices();
  virtual inline data_layout get_data_layout() { return T_layout; }

  void setup(int num_prev_neurons) {
    regularizer_layer::setup(num_prev_neurons);
    this->m_num_neurons = num_prev_neurons;
    Zeros(*(this->m_activations), this->m_num_neurons, this->m_mini_batch_size);
    Zeros(*(this->m_error_signal), num_prev_neurons, this->m_mini_batch_size);
  }
 protected:
  /** Drop out units in forward propagation. */
  void fp_compute() {
    if (this->get_execution_mode() != execution_mode::training ||
        m_keep_prob < 0.0f) {
      // Copy previous activations over.
      El::Copy(*(this->m_prev_activations), *(this->m_activations));
      return;
    }
    ElMat *input_acts = this->m_prev_activations;
    const El::Int local_height = input_acts->LocalHeight();
    const El::Int local_width = input_acts->LocalWidth();

#ifdef LBANN_PROCDET_DROPOUT
    const El::Int global_height = input_acts->Height();
    bernoulli_fill_procdet(*m_cur_mask, input_acts->Height(),
                           input_acts->Width(), m_keep_prob);
    *m_cur_mask *= DataType(1.0) / m_keep_prob;
    if (input_acts->GlobalRow(local_height - 1) == global_height - 1) {
      for (El::Int col = 0; col < local_width; ++col) {
        m_cur_mask->SetLocal(local_height - 1, col, DataType(1.0));
      }
    }
    El::Hadamard(*input_acts, *m_cur_mask, *(this->m_activations));
#else
    Mat& local_input_acts = input_acts->Matrix();
    Mat& local_output_acts = this->m_activations->Matrix();

    // Construct dropout mask
    // Note: Construct Bernoulli matrix and scale by 1/m_keep_prob.
    m_cur_mask->Resize(local_height, local_width);
    El::EntrywiseMap(*m_cur_mask,
                     (std::function<DataType(const DataType&)>)
                     ([this](const DataType& z)->DataType {
                       auto& gen = get_fast_generator();
                       std::bernoulli_distribution dist(m_keep_prob);
                       return dist(gen) ? DataType(1) / m_keep_prob : DataType(0);
                     }));
    // Apply dropout mask to local activations
    El::Hadamard(local_input_acts, *m_cur_mask, local_output_acts);
#endif  // LBANN_PROCDET_DROPOUT
  }

  /** Adjust gradients for dropout in backprop. */
  void bp_compute() {
    // Terminate early when not training.
    if (this->get_execution_mode() != execution_mode::training) {
      return;
    }
    if (m_keep_prob < 0.0f) {
      // Copy error signal through.
      El::Copy(*(this->m_prev_error_signal), *(this->m_error_signal));
      return;
    }

#ifdef LBANN_PROCDET_DROPOUT
    El::Hadamard(*(this->m_prev_error_signal), *m_cur_mask, *(this->m_error_signal));
#else
    // Re-weight the incoming loss using dropout mask
    Mat& local_prev_error_signal = this->m_prev_error_signal->Matrix();
    Mat& local_error_signal = this->m_error_signal->Matrix();
    El::Hadamard(local_prev_error_signal, *m_cur_mask, local_error_signal);
#endif  // LBANN_PROCDET_DROPOUT
  }

  /** Probability of keeping each unit. */
  float m_keep_prob;
#ifdef LBANN_PROCDET_DROPOUT
  /** Current dropout mask (a scaled Bernoulli random matrix). */
  ElMat *m_cur_mask;
#else
  /** Current dropout mask (a scaled Bernoulli random matrix). */
  Mat *m_cur_mask;
#endif
};

template<> inline void dropout<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  regularizer_layer::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
#ifdef LBANN_PROCDET_DROPOUT
  m_cur_mask = new DistMat(m_comm->get_model_grid());
#else
  m_cur_mask = new Mat;
#endif
}

template<> inline void dropout<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  regularizer_layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
#ifdef LBANN_PROCDET_DROPOUT
  m_cur_mask = new StarMat(m_comm->get_model_grid());
#else
  m_cur_mask = new Mat;
#endif
}

}  // namespace lbann

#endif  // LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
