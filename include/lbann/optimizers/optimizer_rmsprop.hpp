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
// lbann_optimizer_rmsprop .hpp .cpp - SGD with RMSprop
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_RMSPROP_HPP
#define LBANN_OPTIMIZER_RMSPROP_HPP

#include "lbann/optimizers/optimizer.hpp"
#include <sys/stat.h>

namespace lbann {

/// RMSprop optimizer
class rmsprop : public optimizer {
 public:
  /// Constructor
  rmsprop
  (lbann_comm *comm,
   DataType learning_rate,
   DataType decay_rate,
   DataType eps = DataType(1e-8));
  rmsprop(const rmsprop& other);
  rmsprop& operator=(const rmsprop& other);
  /// Destructor
  ~rmsprop();
  
  /// Returns the optimizer's name
  std::string get_name() const  { return "rmsprop"; }

  /** Returns description of ctor params */
  std::string get_description() const {
    return std::string {} +
     " rmsprop; learning_rate: " + std::to_string(m_learning_rate) 
     + " decay_rate: " + std::to_string(m_decay_rate)
     + " eps: " + std::to_string(m_eps);
  }

  rmsprop* copy() const { return new rmsprop(*this); }
  /// Set parameters to optimize and initialize optimizer
  void setup(AbsDistMat *parameters);
  /// Update parameters using objective function gradient
  void update(const AbsDistMat *gradient);
  std::string name() const { return "rmsprop"; }
 private:
  /// Decay rate
  DataType m_decay_rate;
  /// Small factor to avoid division by zero
  DataType m_eps;
  /// RMSprop cache
  AbsDistMat *m_cache;
};

/// Factory for RMSprop optimizer
class rmsprop_factory : public optimizer_factory {
 public:
  /// Constructor
  rmsprop_factory
  (lbann_comm *comm,
   DataType learning_rate,
   DataType decay_rate = DataType(0.9),
   DataType eps = DataType(1e-8));
  /// Destructor
  virtual ~rmsprop_factory();
  /// Create RMSprop optimizer
  optimizer *create_optimizer();
 private:
  /// Learning rate
  DataType m_learning_rate;
  /// Decay rate
  DataType m_decay_rate;
  /// Small factor to avoid division by zero
  DataType m_eps;
};

} // namespace lbann

#endif // LBANN_OPTIMIZER_RMSPROP_HPP
