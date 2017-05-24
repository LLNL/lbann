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

#include "lbann/lbann_base.hpp"
#include "lbann/models/lbann_model.hpp"
#include "lbann/regularization/lbann_batch_normalization.hpp"

using namespace El;

namespace lbann {

batch_normalization::batch_normalization(data_layout data_dist,
                                         lbann_comm* comm, DataType decay,
                                         DataType gamma, DataType beta) :
  comm(comm), m_gamma_init(gamma), m_beta_init(beta), m_decay(decay) {
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

batch_normalization::~batch_normalization() {
  delete m_gamma;
  delete m_beta;
  delete m_dgamma;
  delete m_dbeta;
  delete m_mean;
  delete m_var;
  delete m_running_mean;
  delete m_running_var;
}

void batch_normalization::initialize_model_parallel_distribution() {
  m_gamma = new RowSumMat(comm->get_model_grid());
  m_beta = new RowSumMat(comm->get_model_grid());
  m_dgamma = new RowSumMat(comm->get_model_grid());
  m_dbeta = new RowSumMat(comm->get_model_grid());
  m_mean = new RowSumMat(comm->get_model_grid());
  m_var = new RowSumMat(comm->get_model_grid());
  m_running_mean = new RowSumMat(comm->get_model_grid());
  m_running_var = new RowSumMat(comm->get_model_grid());
  m_weighted_sum_copy = new DistMat(comm->get_model_grid());
}

void batch_normalization::initialize_data_parallel_distribution() {
  m_gamma = new StarMat(comm->get_model_grid());
  m_beta = new StarMat(comm->get_model_grid());
  m_dgamma = new StarMat(comm->get_model_grid());
  m_dbeta = new StarMat(comm->get_model_grid());
  m_mean = new StarMat(comm->get_model_grid());
  m_var = new StarMat(comm->get_model_grid());
  m_running_mean = new StarMat(comm->get_model_grid());
  m_running_var = new StarMat(comm->get_model_grid());
}

void batch_normalization::fp_weights() {
  // Get output from linearity.
  ElMat* acts = m_layer->m_activations_v;
  Copy(*(m_layer->m_weighted_sum_v), *m_weighted_sum_copy);
  Int mbsize = acts->Width();
  Mat& acts_local = acts->Matrix();
  Mat& gamma_local = m_gamma->Matrix();
  Mat& beta_local = m_beta->Matrix();
  const Int local_width = acts_local.Width();
  const Int local_height = acts_local.Height();
  if (m_layer->m_execution_mode == execution_mode::training) {
    Mat& mean_local = m_mean->Matrix();
    Mat& var_local = m_var->Matrix();
    // Compute the mean and variance of the activations over the minibatch.
    // Todo: Better access pattern for locality.
    for (Int row = 0; row < local_height; ++row) {
      DataType sum = 0.0;
      DataType sqsum = 0.0;
      for (Int col = 0; col < local_width; ++col) {
        sum += acts_local(row, col);
        sqsum += acts_local(row, col) * acts_local(row, col);
      }
      mean_local(row, 0) = sum;
      var_local(row, 0) = sqsum;
    }
    AllReduce(*m_mean, m_mean->RedundantComm(), mpi::SUM);
    AllReduce(*m_var, m_var->RedundantComm(), mpi::SUM);
    for (Int row = 0; row < local_height; ++row) {
      mean_local(row, 0) /= mbsize;
      var_local(row, 0) = Sqrt(var_local(row, 0) / mbsize -
                               (mean_local(row, 0) * mean_local(row, 0)));
    }
    // Compute transformed activations xhat = (x-mean)/sqrt(var)
    for (Int col = 0; col < local_width; ++col) {
      for (Int row = 0; row < local_height; ++row) {
        // Normalize.
        acts_local(row, col) -= mean_local(row, 0);
        acts_local(row, col) /= var_local(row, 0) + DataType(1e-7);
        // Apply scale/shift.
        acts_local(row, col) *= gamma_local(row, 0);
        acts_local(row, col) += beta_local(row, 0);
      }
    }
    // Update the running averages.
    Scale(m_decay, *m_running_mean);
    Scale(m_decay, *m_running_var);
    Axpy(DataType(1) - m_decay, *m_mean, *m_running_mean);
    Axpy(DataType(1) - m_decay, *m_var, *m_running_var);
  } else if (m_layer->m_execution_mode == execution_mode::validation ||
             m_layer->m_execution_mode == execution_mode::testing) {
    // Use the running mean/variance to normalize.
    const Mat& mean_local = m_running_mean->LockedMatrix();
    const Mat& var_local = m_running_var->LockedMatrix();
    for (Int col = 0; col < local_width; ++col) {
      for (Int row = 0; row < local_height; ++row) {
        acts_local(row, col) -= mean_local(row, 0);
        acts_local(row, col) /= var_local(row, 0) + DataType(1e-7);
        acts_local(row, col) *= gamma_local(row, 0);
        acts_local(row, col) += beta_local(row, 0);
      }
    }
  }
}

void batch_normalization::bp_weights() {
  // No backprop when not training.
  if (m_layer->m_execution_mode != execution_mode::training) return;
  ElMat* bpsignal = m_layer->m_prev_error_signal_v;
  // "activations" here are from *before* the nonlinearity is applied.
  //const ElMat* acts = m_layer->m_weighted_sum_v;
  const ElMat* acts = m_weighted_sum_copy;
  const Int mbsize = acts->Width();
  Mat& bp_local = bpsignal->Matrix();
  const Mat& acts_local = acts->LockedMatrix();
  const Int local_height = bp_local.Height();
  const Int local_width = bp_local.Width();
  const Mat& mean_local = m_mean->LockedMatrix();
  const Mat& var_local = m_var->LockedMatrix();
  const Mat& gamma_local = m_gamma->LockedMatrix();
  Mat& dgamma_local = m_dgamma->Matrix();
  Mat& dbeta_local = m_dbeta->Matrix();
  // Compute the derivatives of gamma and beta.
  for (Int row = 0; row < local_height; ++row) {
    dbeta_local(row, 0) = 0.0;
    dgamma_local(row, 0) = 0.0;
    for (Int col = 0; col < local_width; ++col) {
      dbeta_local(row, 0) += bp_local(row, col);
      dgamma_local(row, 0) +=
        ((acts_local(row, col) - mean_local(row, 0)) /
         (var_local(row, 0) + DataType(1e-7))) *
        bp_local(row, col);
    }
  }
  AllReduce(*m_dbeta, m_dbeta->RedundantComm(), mpi::SUM);
  AllReduce(*m_dgamma, m_dgamma->RedundantComm(), mpi::SUM);
  // Update the backprop gradient signal.
  for (Int row = 0; row < local_height; ++row) {
    for (Int col = 0; col < local_width; ++col) {
      bp_local(row, col) = mbsize * bp_local(row, col);
      bp_local(row, col) -= dbeta_local(row, 0);
      bp_local(row, col) -=
        ((acts_local(row, col) - mean_local(row, 0)) /
         (var_local(row, 0) + DataType(1e-7))) *
        dgamma_local(row, 0);
      bp_local(row, col) /= DataType(mbsize);
      bp_local(row, col) *= gamma_local(row, 0);
      bp_local(row, col) /= var_local(row, 0) + DataType(1e-7);
    }
  }
}

void batch_normalization::setup(Layer* l) {
  regularizer::setup(l);
  Ones(*m_gamma, l->NumNeurons, 1);
  Scale(m_gamma_init, *m_gamma);
  Ones(*m_beta, l->NumNeurons, 1);
  Scale(m_beta_init, *m_beta);
  Zeros(*m_dgamma, l->NumNeurons, 1);
  Zeros(*m_dbeta, l->NumNeurons, 1);
  Zeros(*m_mean, l->NumNeurons, 1);
  Zeros(*m_var, l->NumNeurons, 1);
  Zeros(*m_running_mean, l->NumNeurons, 1);
  Zeros(*m_running_var, l->NumNeurons, 1);
  m_gamma_optimizer = l->neural_network_model->create_optimizer();
  m_beta_optimizer = l->neural_network_model->create_optimizer();
  m_gamma_optimizer->setup(m_gamma);
  m_beta_optimizer->setup(m_beta);
}

void batch_normalization::update() {
  regularizer::update();
  if (m_layer->m_execution_mode == execution_mode::training) {
    m_gamma_optimizer->update(m_dgamma);
    m_beta_optimizer->update(m_dbeta);
  }
}

}  // namespace lbann

