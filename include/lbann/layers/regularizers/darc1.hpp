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
// lbann_darc1 .cpp .hpp - DARC1 implementation
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_REGULARIZER_DARC1_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_DARC1_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"

namespace lbann {

template <data_layout T_layout>
class darc1 : public regularizer_layer {
 public:

  darc1(int index,
          lbann_comm *comm,
          double scaling_factor) :
    regularizer_layer(index, comm),
    m_scaling_factor(scaling_factor),
    m_max_row(0) {
    initialize_distributed_matrices();
  }

  darc1(const darc1& other) :
    regularizer_layer(other),
    m_scaling_factor(other.m_scaling_factor),
    m_max_row(other.m_max_row) {
    m_row_norms = other.m_row_norms->Copy();
  }

  darc1& operator=(const darc1& other) {
    regularizer_layer::operator=(other);
    m_scaling_factor = other.m_scaling_factor;
    m_max_row = other.m_max_row;
    if (m_row_norms) {
      delete m_row_norms;
    }
    m_row_norms = other.m_row_norms->Copy();
    return *this;
  }

  ~darc1() {
    delete m_row_norms;
  }

  darc1* copy() const { return new darc1(*this); }

  std::string get_type() const { return "darc1"; }

  std::string get_description() const {
    return " darc1 scaling_factor: " + std::to_string(m_scaling_factor) 
           + " dataLayout: " + get_data_layout_string(get_data_layout());
  }

  virtual inline void initialize_distributed_matrices();
  virtual data_layout get_data_layout() const { return T_layout; }

 protected:

  void setup_data() {
    regularizer_layer::setup_data();
    El::Zeros(*m_row_norms, this->m_num_neurons, 1);
  }

  void fp_compute() {

    // Forward prop does not affect activations
    El::LockedView(*this->m_activations_v, *this->m_prev_activations);

    // Compute DARC1 regularization term
    if (m_scaling_factor != DataType(0)) {

      // Get local matrices
      const Mat& activations_local = this->m_activations_v->LockedMatrix();
      Mat& row_norms_local = m_row_norms->Matrix();
      const int local_height = activations_local.Height();
      const int local_width = activations_local.Width();

      // Compute row-wise 1-norms
      El::Zero(row_norms_local);
      const int block_size = std::max((int) (64 / sizeof(DataType)), 1);
      #pragma omp parallel for
      for (int block_start = 0;
           block_start < local_height;
           block_start += block_size) {
        const int block_end = std::min(block_start + block_size,
                                       local_height);
        for (int col = 0; col < local_width; ++col) {
          for (int row = block_start; row < block_end; ++row) {
            row_norms_local(row, 0) += std::fabs(activations_local(row, col));
          }
        }
      }
      El::AllReduce(m_row_norms->Matrix(),
                    m_row_norms->RedundantComm(),
                    El::mpi::SUM);
      
      // Get row with largest 1-norm
      auto max_row_norm = El::VectorMaxAbsLoc(*m_row_norms);
      m_max_row = max_row_norm.index;

      // Apply DARC1 regularization term to objective function
      const int mini_batch_size = this->m_activations_v->Width();
      const DataType regularization_term = (m_scaling_factor / mini_batch_size
                                            * max_row_norm.value);
      this->m_neural_network_model->m_obj_fn->add_to_value(regularization_term);

    }
  }

  void bp_compute() {

    // Copy previous error signal
    El::Copy(*this->m_prev_error_signal, *this->m_error_signal_v);
    
    // Apply gradient from DARC1 regularization term
    if (m_scaling_factor != DataType(0)
        && m_error_signal_v->IsLocalRow(m_max_row)) {

      // Get local matrices
      const Mat& activations_local = this->m_activations_v->LockedMatrix();
      Mat& error_signal_local = this->m_error_signal_v->Matrix();
      const int local_width = error_signal_local.Width();

      // Apply DARC1 regularization gradient
      const int mini_batch_size = this->m_activations_v->Width();
      const DataType gradient_term = m_scaling_factor / mini_batch_size;
      const int row = this->m_error_signal_v->LocalRow(m_max_row);
      #pragma omp parallel for
      for (int col = 0; col < local_width; ++col) {
        const DataType activations_entry = activations_local(row, col);
        DataType& error_signal_entry = error_signal_local(row, col);
        if (activations_entry > DataType(0)) {
          error_signal_entry += gradient_term;
        }
        if (activations_entry < DataType(0)) {
          error_signal_entry -= gradient_term;
        }
      }
      
    }

  }

  /** Scaling factor for DARC1 regularization term. */
  DataType m_scaling_factor;
  /** Activation matrix row with maximum 1-norm. */
  int m_max_row;
  /** 1-norms for rows of the activation matrix. */
  AbsDistMat *m_row_norms;

};

template<> inline void darc1<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  regularizer_layer::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_row_norms = new RowSumMat(m_comm->get_model_grid());
}

template<> inline void darc1<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  regularizer_layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_row_norms = new StarMat(m_comm->get_model_grid());
}

}  // namespace lbann

#endif  // LBANN_LAYER_REGULARIZER_DARC1_HPP_INCLUDED
