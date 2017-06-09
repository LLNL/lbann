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

#ifndef LBANN_REGULARIZATION_DROPOUT_HPP_INCLUDED
#define LBANN_REGULARIZATION_DROPOUT_HPP_INCLUDED

#include "lbann/regularization/lbann_regularizer.hpp"

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
class dropout : public regularizer {
 public:
  /** Keep units with probabiliy keep_prob. */
  dropout(data_layout data_dist, lbann_comm *comm, float keep_prob=0.5f);
  ~dropout();
  void initialize_model_parallel_distribution();
  void initialize_data_parallel_distribution();
  /** Drop out units in forward propagation. */
  void fp_activations();
  /** Adjust gradients for dropout in backprop. */
  void bp_activations();
 protected:
  lbann_comm *m_comm;
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

}  // namespace lbann

#endif  // LBANN_REGULARIZATION_DROPOUT_HPP_INCLUDED
