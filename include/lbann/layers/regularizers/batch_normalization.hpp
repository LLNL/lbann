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
#include "lbann/utils/statistics.hpp"

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
   * @param scale The initial value for scaling parameter
   * \f$\gamma$\f$. The paper recommends 1.0 as a starting point, but
   * other papers have had better results with a smaller value
   * (e.g. 0.1).
   * @param bias The initial value for bias parameter
   * \f$\beta\f$. This should almost always stay at zero.
   */
  batch_normalization(int index,
                      lbann_comm *comm,
                      int mini_batch_size,
                      DataType decay=0.9,
                      DataType scale_init=1.0,
                      DataType bias_init=0.0)
    : regularizer_layer(index, comm, mini_batch_size),
      m_scale_init(scale_init),
      m_bias_init(bias_init),
      m_decay(decay) {
    // Setup the data distribution
    initialize_distributed_matrices();
  }

  batch_normalization(const batch_normalization& other) :
    regularizer_layer(other),
    m_scale_init(other.m_scale_init),
    m_bias_init(other.m_bias_init),
    m_decay(other.m_decay) {
    m_scale = other.m_scale->Copy();
    m_bias = other.m_bias->Copy();
    m_scale_gradient = other.m_scale_gradient->Copy();
    m_bias_gradient = other.m_bias_gradient->Copy();
    m_mean = other.m_mean->Copy();
    m_stdev = other.m_stdev->Copy();
    m_running_mean = other.m_running_mean->Copy();
    m_running_stdev = other.m_running_stdev->Copy();
    m_scale_optimizer = other.m_scale_optimizer->copy();
    m_bias_optimizer = other.m_bias_optimizer->copy();
    if (m_scale_optimizer->get_parameters()) {
      m_scale_optimizer->set_parameters(m_scale);
      m_bias_optimizer->set_parameters(m_bias);
    }
  }

  batch_normalization& operator=(const batch_normalization& other) {
    regularizer_layer::operator=(other);
    m_scale_init = other.m_scale_init;
    m_bias_init = other.m_bias_init;
    m_decay = other.m_decay;
    if (m_scale) {
      delete m_scale;
      delete m_bias;
      delete m_scale_gradient;
      delete m_bias_gradient;
      delete m_mean;
      delete m_stdev;
      delete m_running_mean;
      delete m_running_stdev;
    }
    m_scale = other.m_scale->Copy();
    m_bias = other.m_bias->Copy();
    m_scale_gradient = other.m_scale_gradient->Copy();
    m_bias_gradient = other.m_bias_gradient->Copy();
    m_mean = other.m_mean->Copy();
    m_stdev = other.m_stdev->Copy();
    m_running_mean = other.m_running_mean->Copy();
    m_running_stdev = other.m_running_stdev->Copy();
    if (m_scale_optimizer) {
      delete m_scale_optimizer;
      delete m_bias_optimizer;
    }
    if (other.m_scale_optimizer) {
      m_scale_optimizer = other.m_scale_optimizer->copy();
      m_bias_optimizer = other.m_bias_optimizer->copy();
      m_scale_optimizer->set_parameters(m_scale);
      m_bias_optimizer->set_parameters(m_bias);
    }
    return *this;
  }

  ~batch_normalization() {
    delete m_scale;
    delete m_bias;
    delete m_scale_gradient;
    delete m_bias_gradient;
    delete m_mean;
    delete m_stdev;
    delete m_running_mean;
    delete m_running_stdev;
  }

  batch_normalization* copy() const { return new batch_normalization(*this); }

  std::string get_name() const { return "batch normalization"; }

  virtual inline void initialize_distributed_matrices();
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_data() {
    regularizer_layer::setup_data();
    // Initialize parameters
    m_scale->Resize(this->get_num_neurons(), 1);
    El::Fill(*m_scale, m_scale_init);
    m_bias->Resize(this->get_num_neurons(), 1);
    El::Fill(*m_bias, m_bias_init);
    El::Zeros(*m_scale_gradient, this->get_num_neurons(), 1);
    El::Zeros(*m_bias_gradient, this->get_num_neurons(), 1);
    El::Zeros(*m_mean, this->get_num_neurons(), 1);
    El::Zeros(*m_stdev, this->get_num_neurons(), 1);
    El::Zeros(*m_running_mean, this->get_num_neurons(), 1);
    El::Zeros(*m_running_stdev, this->get_num_neurons(), 1);
    m_scale_optimizer = this->get_neural_network_model()->create_optimizer();
    m_bias_optimizer = this->get_neural_network_model()->create_optimizer();
    m_scale_optimizer->setup(m_scale);
    m_bias_optimizer->setup(m_bias);
  }

  void fp_compute() {
    ElMat *input_acts = this->m_prev_activations;
    Mat& input_acts_local = input_acts->Matrix();
    Mat& acts_local = this->m_activations->Matrix();
    Mat& scale_local = m_scale->Matrix();
    Mat& bias_local = m_bias->Matrix();
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
          acts_local(row, col) *= scale_local(row, 0);
          acts_local(row, col) += bias_local(row, 0);
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
          acts_local(row, col) *= scale_local(row, 0);
          acts_local(row, col) += bias_local(row, 0);
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
    const Mat& scale_local = m_scale->LockedMatrix();
    Mat& scale_gradient_local = m_scale_gradient->Matrix();
    Mat& bias_gradient_local = m_bias_gradient->Matrix();
    // Compute the derivatives of scale and bias.
#pragma omp parallel for
    for (El::Int row = 0; row < local_height; ++row) {
      bias_gradient_local(row, 0) = 0.0;
      scale_gradient_local(row, 0) = 0.0;
      for (El::Int col = 0; col < local_width; ++col) {
        bias_gradient_local(row, 0) += input_bp_local(row, col);
        scale_gradient_local(row, 0) +=
          ((acts_local(row, col) - mean_local(row, 0)) /
           (stdev_local(row, 0) + DataType(1e-7))) *
          input_bp_local(row, col);
      }
    }
    AllReduce(*m_bias_gradient, m_bias_gradient->RedundantComm(), mpi::SUM);
    AllReduce(*m_scale_gradient, m_scale_gradient->RedundantComm(), mpi::SUM);
    // Update the backprop gradient signal.
#pragma omp parallel for collapse(2)
    for (El::Int row = 0; row < local_height; ++row) {
      for (El::Int col = 0; col < local_width; ++col) {
        bp_local(row, col) = mbsize * input_bp_local(row, col);
        bp_local(row, col) -= bias_gradient_local(row, 0);
        bp_local(row, col) -=
          ((acts_local(row, col) - mean_local(row, 0)) /
           (stdev_local(row, 0) + DataType(1e-7))) *
          scale_gradient_local(row, 0);
        bp_local(row, col) /= DataType(mbsize);
        bp_local(row, col) *= scale_local(row, 0);
        bp_local(row, col) /= stdev_local(row, 0) + DataType(1e-7);
        bp_local(row, col) += DataType(1e-8);  // Avoid very small values.
      }
    }
  }

  bool update_compute() {
    if (this->get_execution_mode() == execution_mode::training) {
      m_scale_optimizer->update(m_scale_gradient);
      m_bias_optimizer->update(m_bias_gradient);
    }
    return true;
  }

 protected:
  /** For learning scale and bias. */
  optimizer *m_scale_optimizer;
  optimizer *m_bias_optimizer;
  /** Default initialization value for scale. */
  DataType m_scale_init;
  /** Default initialization value for bias. */
  DataType m_bias_init;
  /** Scale parameter (one scale for each activation). */
  ElMat *m_scale;
  /** Shift parameter (one shift for each activation). */
  ElMat *m_bias;
  /** Gradients of scale. */
  ElMat *m_scale_gradient;
  /** Gradients of bias. */
  ElMat *m_bias_gradient;
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
  m_scale = new RowSumMat(this->m_comm->get_model_grid());
  m_bias = new RowSumMat(this->m_comm->get_model_grid());
  m_scale_gradient = new RowSumMat(this->m_comm->get_model_grid());
  m_bias_gradient = new RowSumMat(this->m_comm->get_model_grid());
  m_mean = new RowSumMat(this->m_comm->get_model_grid());
  m_stdev = new RowSumMat(this->m_comm->get_model_grid());
  m_running_mean = new RowSumMat(this->m_comm->get_model_grid());
  m_running_stdev = new RowSumMat(this->m_comm->get_model_grid());
}
template<> inline void batch_normalization<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  regularizer_layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_scale = new StarMat(this->m_comm->get_model_grid());
  m_bias = new StarMat(this->m_comm->get_model_grid());
  m_scale_gradient = new StarMat(this->m_comm->get_model_grid());
  m_bias_gradient = new StarMat(this->m_comm->get_model_grid());
  m_mean = new StarMat(this->m_comm->get_model_grid());
  m_stdev = new StarMat(this->m_comm->get_model_grid());
  m_running_mean = new StarMat(this->m_comm->get_model_grid());
  m_running_stdev = new StarMat(this->m_comm->get_model_grid());
}

}  // namespace lbann

#endif  // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
