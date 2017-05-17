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

#ifndef LBANN_REGULARIZATION_BATCH_NORMALIZATION_HPP_INCLUDED
#define LBANN_REGULARIZATION_BATCH_NORMALIZATION_HPP_INCLUDED

#include "lbann/regularization/lbann_regularizer.hpp"

namespace lbann {

/**
 * Batch normalization: normalize layers to zero mean/unit variance.
 * See paper:
 * Sergey Ioffe and Christian Szegedy. "Batch Normalization: Accelerating Deep
 * Network Training by Reducing Internal Covariate Shift." ICML 2015.
 * This keeps a running mean and variance (with exponential decay) instead of
 * computing it over the data at test time. This approach seems to have become
 * standard.
 * See also:
 * https://cthorey.github.io/backpropagation/
 */
class batch_normalization : public regularizer {
public:
  /**
   * Set up batch normalization.
   * @param decay Controls the momentum of the running mean/variance averages.
   * @param gamma The initial value for gamma. The paper recommends 1.0 as a
   * starting point, but other papers have had better results with a smaller
   * value (e.g. 0.1).
   * @param beta The initial value for beta. This should almost always stay at
   * zero.
   */
  batch_normalization(data_layout data_dist, lbann_comm* comm,
                      DataType decay=0.9, DataType gamma=1.0, DataType beta=0.0);
  ~batch_normalization();
  void initialize_model_parallel_distribution();
  void initialize_data_parallel_distribution();
  void fp_weights();
  void bp_weights();
  /** Initializes matrices. */
  void setup(Layer* l);
  void update();
protected:
  lbann_comm* comm;
  /** For learning gamma and beta. */
  optimizer* m_gamma_optimizer;
  optimizer* m_beta_optimizer;
  /** Default initialization value for gamma. */
  DataType m_gamma_init;
  /** Default initialization value for beta. */
  DataType m_beta_init;
  /** Scale parameter (one scale for each activation). */
  ElMat* m_gamma;
  /** Shift parameter (one shift for each activation). */
  ElMat* m_beta;
  /** Gradients of gamma. */
  ElMat* m_dgamma;
  /** Gradients of beta. */
  ElMat* m_dbeta;
  /** Decay rate for the running mean/variance. */
  DataType m_decay;
  /** Current minibatch mean of activations. */
  ElMat* m_mean;
  /** Current minibatch variance of activations. */
  ElMat* m_var;
  /** Running mean of activations (for inference). */
  ElMat* m_running_mean;
  /** Running variance of activations (for inference). */
  ElMat* m_running_var;
  ElMat* m_weighted_sum_copy;
};

}  // namespace lbann

#endif  // LBANN_REGULARIZATION_BATCH_NORMALIZATION_HPP_INCLUDED
