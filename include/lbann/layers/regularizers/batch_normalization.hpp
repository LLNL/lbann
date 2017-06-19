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
// lbann_batch_normalization .cpp .hpp - Batch normalization implementation
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/utils/lbann_statistics.hpp"

namespace lbann {

/**
 * Batch normalization: normalize layers to zero mean/unit standard deviation.
 * See paper:
 * Sergey Ioffe and Christian Szegedy. "Batch Normalization:
 * Accelerating Deep Network Training by Reducing Internal Covariate
 * Shift." ICML 2015.  This keeps a running mean and standard
 * deviation (with exponential decay) instead of computing it over the
 * data at test time. This approach seems to have become standard.
 * See also:
 * https://cthorey.github.io/backpropagation/
 */
template <data_layout T_layout>
class batch_normalization : public regularizer_layer {
 public:
  /**
   * Set up batch normalization.
   * @param decay Controls the momentum of the running mean/standard
   * deviation averages.
   * @param gamma The initial value for gamma. The paper recommends 1.0
   * as a starting point, but other papers have had better results with
   * a smaller value (e.g. 0.1).
   * @param beta The initial value for beta. This should almost always
   * stay at zero.
   */
  batch_normalization(const uint index, const uint num_neurons,
                      lbann_comm *comm, uint mini_batch_size,
                      DataType decay=0.9, DataType gamma=1.0, DataType beta=0.0)
    : regularizer_layer(index, comm, mini_batch_size),
      m_gamma_init(gamma), m_beta_init(beta), m_decay(decay) {
    set_name("batch_normalization");
    // Setup the data distribution
    initialize_distributed_matrices();
    this->m_type = layer_type::batch_normalization;
    this->m_num_neurons = num_neurons;
  }

  ~batch_normalization() {
    delete m_gamma;
    delete m_beta;
    delete m_dgamma;
    delete m_dbeta;
    delete m_mean;
    delete m_stdev;
    delete m_running_mean;
    delete m_running_stdev;
  }

  virtual inline void initialize_distributed_matrices();
  virtual inline data_layout get_data_layout() { return T_layout; }

  /** Initializes matrices. */
  void setup(int num_prev_neurons) {
    regularizer_layer::setup(num_prev_neurons);
    this->m_num_neurons = num_prev_neurons;
    Zeros(*(this->m_activations), this->m_num_neurons, this->m_mini_batch_size);
    Zeros(*(this->m_error_signal), num_prev_neurons, this->m_mini_batch_size);
    Ones(*m_gamma, this->get_num_neurons(), 1);
    Scale(m_gamma_init, *m_gamma);
    Ones(*m_beta, this->get_num_neurons(), 1);
    Scale(m_beta_init, *m_beta);
    Zeros(*m_dgamma, this->get_num_neurons(), 1);
    Zeros(*m_dbeta, this->get_num_neurons(), 1);
    Zeros(*m_mean, this->get_num_neurons(), 1);
    Zeros(*m_stdev, this->get_num_neurons(), 1);
    Zeros(*m_running_mean, this->get_num_neurons(), 1);
    Zeros(*m_running_stdev, this->get_num_neurons(), 1);
    m_gamma_optimizer = this->get_neural_network_model()->create_optimizer();
    m_beta_optimizer = this->get_neural_network_model()->create_optimizer();
    m_gamma_optimizer->setup(m_gamma);
    m_beta_optimizer->setup(m_beta);
  }

  void fp_compute() {
    ElMat *input_acts = this->m_prev_activations;
    Mat& input_acts_local = input_acts->Matrix();
    Mat& acts_local = this->m_activations->Matrix();
    Mat& gamma_local = m_gamma->Matrix();
    Mat& beta_local = m_beta->Matrix();
    const El::Int local_width = acts_local.Width();
    const El::Int local_height = acts_local.Height();
    if (this->get_execution_mode() == execution_mode::training) {
      // Compute row-wise mean and standard deviation
      rowwise_mean_and_stdev(*input_acts, *m_mean, *m_stdev);
      Mat& mean_local = m_mean->Matrix();
      Mat& stdev_local = m_stdev->Matrix();
      // Compute transformed activations xhat = (x-mean)/stdev
#pragma omp parallel for collapse(2)
      for (El::Int col = 0; col < local_width; ++col) {
        for (El::Int row = 0; row < local_height; ++row) {
          // Normalize.
          acts_local(row, col) = input_acts_local(row, col) - mean_local(row, 0);
          acts_local(row, col) /= stdev_local(row, 0) + DataType(1e-7);
          // Apply scale/shift.
          acts_local(row, col) *= gamma_local(row, 0);
          acts_local(row, col) += beta_local(row, 0);
        }
      }
      // Update the running averages.
      Scale(m_decay, *m_running_mean);
      Scale(m_decay, *m_running_stdev);
      Axpy(DataType(1) - m_decay, *m_mean, *m_running_mean);
      Axpy(DataType(1) - m_decay, *m_stdev, *m_running_stdev);
    } else if (this->get_execution_mode() == execution_mode::validation ||
        this->get_execution_mode() == execution_mode::testing) {
      // Use the running mean/standard deviation to normalize.
      const Mat& mean_local = m_running_mean->LockedMatrix();
      const Mat& stdev_local = m_running_stdev->LockedMatrix();
#pragma omp parallel for collapse(2)
      for (El::Int col = 0; col < local_width; ++col) {
        for (El::Int row = 0; row < local_height; ++row) {
          // Normalize.
          acts_local(row, col) = input_acts_local(row, col) - mean_local(row, 0);
          acts_local(row, col) /= stdev_local(row, 0) + DataType(1e-7);
          // Apply scale/shift.
          acts_local(row, col) *= gamma_local(row, 0);
          acts_local(row, col) += beta_local(row, 0);
        }
      }
    }
  }

  void bp_compute() {
    // No backprop when not training.
    if (this->get_execution_mode() != execution_mode::training) {
      return;
    }
    const ElMat *input_bpsignal = this->m_prev_error_signal;
    const ElMat *acts = this->m_prev_activations;
    const El::Int mbsize = acts->Width();
    const Mat& input_bp_local = input_bpsignal->LockedMatrix();
    Mat& bp_local = this->m_error_signal->Matrix();
    const Mat& acts_local = acts->LockedMatrix();
    const El::Int local_height = input_bp_local.Height();
    const El::Int local_width = input_bp_local.Width();
    const Mat& mean_local = m_mean->LockedMatrix();
    const Mat& stdev_local = m_stdev->LockedMatrix();
    const Mat& gamma_local = m_gamma->LockedMatrix();
    Mat& dgamma_local = m_dgamma->Matrix();
    Mat& dbeta_local = m_dbeta->Matrix();
    // Compute the derivatives of gamma and beta.
#pragma omp parallel for
    for (El::Int row = 0; row < local_height; ++row) {
      dbeta_local(row, 0) = 0.0;
      dgamma_local(row, 0) = 0.0;
      for (El::Int col = 0; col < local_width; ++col) {
        dbeta_local(row, 0) += input_bp_local(row, col);
        dgamma_local(row, 0) +=
          ((acts_local(row, col) - mean_local(row, 0)) /
           (stdev_local(row, 0) + DataType(1e-7))) *
          input_bp_local(row, col);
      }
    }
    AllReduce(*m_dbeta, m_dbeta->RedundantComm(), mpi::SUM);
    AllReduce(*m_dgamma, m_dgamma->RedundantComm(), mpi::SUM);
    // Update the backprop gradient signal.
#pragma omp parallel for collapse(2)
    for (El::Int row = 0; row < local_height; ++row) {
      for (El::Int col = 0; col < local_width; ++col) {
        bp_local(row, col) = mbsize * input_bp_local(row, col);
        bp_local(row, col) -= dbeta_local(row, 0);
        bp_local(row, col) -=
          ((acts_local(row, col) - mean_local(row, 0)) /
           (stdev_local(row, 0) + DataType(1e-7))) *
          dgamma_local(row, 0);
        bp_local(row, col) /= DataType(mbsize);
        bp_local(row, col) *= gamma_local(row, 0);
        bp_local(row, col) /= stdev_local(row, 0) + DataType(1e-7);
        bp_local(row, col) += DataType(1e-8);  // Avoid very small values.
      }
    }
  }

  bool update_compute() {
    if (this->get_execution_mode() == execution_mode::training) {
      m_gamma_optimizer->update(m_dgamma);
      m_beta_optimizer->update(m_dbeta);
    }
    return true;
  }

 protected:
  /** For learning gamma and beta. */
  optimizer *m_gamma_optimizer;
  optimizer *m_beta_optimizer;
  /** Default initialization value for gamma. */
  DataType m_gamma_init;
  /** Default initialization value for beta. */
  DataType m_beta_init;
  /** Scale parameter (one scale for each activation). */
  ElMat *m_gamma;
  /** Shift parameter (one shift for each activation). */
  ElMat *m_beta;
  /** Gradients of gamma. */
  ElMat *m_dgamma;
  /** Gradients of beta. */
  ElMat *m_dbeta;
  /** Decay rate for the running mean/standard deviation. */
  DataType m_decay;
  /** Current minibatch mean of activations. */
  ElMat *m_mean;
  /** Current minibatch standard deviation of activations. */
  ElMat *m_stdev;
  /** Running mean of activations (for inference). */
  ElMat *m_running_mean;
  /** Running standard deviation of activations (for inference). */
  ElMat *m_running_stdev;
};

template<> inline void batch_normalization<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  regularizer_layer::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_gamma = new RowSumMat(this->m_comm->get_model_grid());
  m_beta = new RowSumMat(this->m_comm->get_model_grid());
  m_dgamma = new RowSumMat(this->m_comm->get_model_grid());
  m_dbeta = new RowSumMat(this->m_comm->get_model_grid());
  m_mean = new RowSumMat(this->m_comm->get_model_grid());
  m_stdev = new RowSumMat(this->m_comm->get_model_grid());
  m_running_mean = new RowSumMat(this->m_comm->get_model_grid());
  m_running_stdev = new RowSumMat(this->m_comm->get_model_grid());
}
template<> inline void batch_normalization<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  regularizer_layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_gamma = new StarMat(this->m_comm->get_model_grid());
  m_beta = new StarMat(this->m_comm->get_model_grid());
  m_dgamma = new StarMat(this->m_comm->get_model_grid());
  m_dbeta = new StarMat(this->m_comm->get_model_grid());
  m_mean = new StarMat(this->m_comm->get_model_grid());
  m_stdev = new StarMat(this->m_comm->get_model_grid());
  m_running_mean = new StarMat(this->m_comm->get_model_grid());
  m_running_stdev = new StarMat(this->m_comm->get_model_grid());
}

}  // namespace lbann

#endif  // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
