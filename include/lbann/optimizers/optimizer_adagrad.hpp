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
// lbann_optimizer_adagrad .hpp .cpp - SGD with AdaGrad optimizer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_ADAGRAD_HPP
#define LBANN_OPTIMIZER_ADAGRAD_HPP

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

/// AdaGrad optimizer
class adagrad : public optimizer {
 public:
  /// Constructor
  adagrad
  (lbann_comm *comm,
   DataType learning_rate,
   DataType eps = DataType(1e-8));
  adagrad(const adagrad& other);
  adagrad& operator=(const adagrad& other);
  /// Destructor
  ~adagrad();

  /// Returns the optimizer's name
  std::string get_name() const  { return "adagrad"; }

  /** Returns description of ctor params */
  std::string get_description() const {
    return std::string {} +
     " adagrad; learning_rate: "
     + std::to_string(m_learning_rate) + " eps: " + std::to_string(m_eps);
  }

  adagrad* copy() const { return new adagrad(*this); }

  /// Set parameters to optimize and initialize optimizer
  void setup(AbsDistMat *parameters);
  /// Update parameters using objective function gradient
  void update(const AbsDistMat *gradient);
  std::string name() const { return "adagrad"; }
 private:
  /// Small factor to avoid division by zero
  DataType m_eps;
  /// AdaGrad cache
  AbsDistMat *m_cache;
};

/// Factory for AdaGrad optimizer
class adagrad_factory : public optimizer_factory {
 public:
  /// Constructor
  adagrad_factory
  (lbann_comm *comm,
   DataType learning_rate,
   DataType eps = DataType(1e-8));
  /// Destructor
  virtual ~adagrad_factory();
  /// Create AdaGrad optimizer
  optimizer *create_optimizer();
 private:
  /// Small factor to avoid division by zero
  DataType m_eps;
  /// Learning rate
  DataType m_learning_rate;
};

} // namespace lbann

#endif // LBANN_OPTIMIZER_ADAGRAD_HPP
