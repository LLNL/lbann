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
// lbann_optimizer_adam .hpp .cpp - SGD with Adam
// Reference:
// Kingma, D. and Ba, J. 2014. Adam: A Method for Stochastic Optimization.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_ADAM_HPP
#define LBANN_OPTIMIZER_ADAM_HPP

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

/// Adam optimizer
class adam : public optimizer {
 public:
  /// Constructor
  adam
  (lbann_comm *comm,
   DataType learning_rate,
   DataType beta1 = DataType(0.9),
   DataType beta2 = DataType(0.99),
   DataType eps = DataType(1e-8),
   cudnn::cudnn_manager *cudnn=nullptr);
  adam(const adam& other);
  adam& operator=(const adam& other);
  /// Destructor
  ~adam();
  
  /// Returns the optimizer's name
  std::string get_name() const  { return "adam"; }

  /** Returns description of ctor params */
  std::string get_description() const {
    return std::string {} +
     " adam; learning_rate: "
     + std::to_string(m_learning_rate) 
     + " beta1: " + std::to_string(m_beta1)
     + " beta2: " + std::to_string(m_beta2)
     + " eps: " + std::to_string(m_eps);
  }

  adam* copy() const { return new adam(*this); }
  /// Set parameters to optimize and initialize optimizer
  void setup(AbsDistMat *parameters);
  void setup_gpu(AbsDistMat *parameters,
                 const std::vector<DataType *> &parameters_d);
  /// Update parameters using objective function gradient
  void update(const AbsDistMat *gradient);
#ifdef __LIB_CUDA
  void update_gpu(const std::vector<DataType *> &gradient_d);
#endif  
  std::string name() const { return "adam"; }
 private:
  /// Update factor for first moment estimate
  DataType m_beta1;
  /// Update factor for second moment estimate
  DataType m_beta2;
  /// Small factor to avoid division by zero
  DataType m_eps;
  /// beta1 ^ iteration
  DataType m_current_beta1;
  /// beta2 ^ iteration
  DataType m_current_beta2;
  /// First moment estimates
  AbsDistMat *m_moment1;
  // GPU memory for m_moment1;
  std::vector<DataType *> m_moment1_d;
  /// Second moment estimates
  AbsDistMat *m_moment2;
  // GPU memory for m_moment2;
  std::vector<DataType *> m_moment2_d;
};

/// Factory for Adam optimizer
class adam_factory : public optimizer_factory {
 public:
  /// Constructor
  adam_factory
  (lbann_comm *comm,
   DataType learning_rate,
   DataType beta1 = DataType(0.9),
   DataType beta2 = DataType(0.99),
   DataType eps = DataType(1e-8),
   cudnn::cudnn_manager *cudnn=nullptr);
  /// Destructor
  virtual ~adam_factory();
  /// Create Adam optimizer
  optimizer *create_optimizer();
 private:
  /// Learning rate
  DataType m_learning_rate;
  /// Update factor for first moment estimate
  DataType m_beta1;
  /// Update factor for second moment estimate
  DataType m_beta2;
  /// Small factor to avoid division by zero
  DataType m_eps;
  cudnn::cudnn_manager *m_cudnn;
};

} // namespace lbann

#endif  // LBANN_OPTIMIZER_ADAM_HPP
