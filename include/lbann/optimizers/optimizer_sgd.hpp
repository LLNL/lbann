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
// lbann_optimizer_sgd .hpp .cpp - Stochastic gradient descent
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_SGD_HPP
#define LBANN_OPTIMIZER_SGD_HPP

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

/// Stochastic gradient descent optimizer
/** Supports momentum, learning rate decay, and Nesterov acceleration.
 */
class sgd : public optimizer {

 public:
  /// Constructor
  sgd
  (lbann_comm *comm,
   DataType learning_rate,
   DataType momentum = DataType(0),
   DataType decay_rate = DataType(0),
   bool nesterov = false);
  sgd(const sgd& other);
  sgd& operator=(const sgd& other);
  /// Destructor
  ~sgd();
  
  /// Returns the optimizer's name
  std::string get_name() const  { return "sgd"; }

  /** Returns description of ctor params */
  std::string get_description() const {
    return std::string {} +
     " sgd; learning_rate: " + std::to_string(m_learning_rate) 
     + " momentum: " + std::to_string(m_momentum)
     + " decay_rate: " + std::to_string(m_decay)
     + " nesterov: " + std::to_string(m_nesterov);
  }

  sgd* copy() const { return new sgd(*this); }
  /// Set parameters to optimize and initialize optimizer
  void setup(AbsDistMat *parameters);
  /// Update parameters using objective function gradient
  void update(const AbsDistMat *gradient);
  std::string name() const { return "sgd"; }
 private:
  /// Number of iterations
  int m_iterations;
  /// Momentum
  DataType m_momentum;
  /// Learning rate decay
  DataType m_decay;
  /// Nesterov acceleration
  bool m_nesterov;
  /// Velocity term for momentum SGD
  AbsDistMat *m_velocity;

};

/// Factory for stochastic gradient descent optimizer
class sgd_factory : public optimizer_factory {
 public:
  /// Constructor
  sgd_factory
  (lbann_comm *comm,
   DataType learning_rate,
   DataType momentum = DataType(0),
   DataType decay = DataType(0),
   bool nesterov = false);
  /// Destructor
  virtual ~sgd_factory();
  /// Create SGD optimizer
  optimizer *create_optimizer();
 private:
  /// Learning rate
  DataType m_learning_rate;
  /// Momentum
  DataType m_momentum;
  /// Learning rate decay
  DataType m_decay;
  /// Nesterov acceleration
  bool m_nesterov;
};

} // namespace lbann

#endif // LBANN_OPTIMIZER_SGD_HPP
