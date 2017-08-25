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
// lbann_optimizer .hpp .cpp - Abstract optimizer class
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_HPP
#define LBANN_OPTIMIZER_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include <string>

namespace lbann {

/// Optimizer base class
class optimizer {

 public:

  /// Constructor
  optimizer(lbann_comm *comm,
            DataType learning_rate = DataType(0),
            cudnn::cudnn_manager *cudnn=nullptr);
  optimizer(const optimizer&) = default;
  optimizer& operator=(const optimizer&) = default;

  /// Destructor
  virtual ~optimizer();

  /**
   * Virtual copy operator. Returns a copy of the true class.
   */
  virtual optimizer* copy() const = 0;

  /// Set parameters to optimize and initialize optimizer
  virtual void setup(AbsDistMat *parameters);
  virtual void setup_gpu(AbsDistMat *parameters,
                         const std::vector<DataType *> &parameters_d);

  /// Returns the optimizer's name
  virtual std::string get_name() const = 0;

  virtual std::string get_description() const = 0;

  /// Update parameters using objective function gradient
  virtual void update(const AbsDistMat *gradient) = 0;
  
  virtual void update_gpu(const std::vector<DataType *> &gradient_d) {
    throw new lbann_exception("update_gpu not supported");
  }

  /// Get learning rate
  virtual DataType get_learning_rate() const {
    return m_learning_rate;
  }

  /// Set learning rate
  virtual void set_learning_rate(DataType learning_rate) {
    m_learning_rate = learning_rate;
  };

  /// Get parameters
  AbsDistMat *get_parameters() const {
    return m_parameters;
  }

  /**
   * Set parameters to optimize.
   * Undefined if parameters is different dimensions or distribution than what
   * was originally set!
   */
  void set_parameters(AbsDistMat *parameters) {
    m_parameters = parameters;
  }
  
  void set_parameters_gpu(AbsDistMat *parameters,
                          const std::vector<DataType *> &parameters_d) {
    set_parameters(parameters);
    m_parameters_d = parameters_d;
  }

  /// Get optimizer name
  virtual std::string name() const = 0;

 protected:
  /// LBANN communicator
  lbann_comm *m_comm;
  /// cuDNN manager
  cudnn::cudnn_manager *m_cudnn;
  /// Parameters to optimize
  AbsDistMat *m_parameters;
  /// m_parameters on GPU memory
  std::vector<DataType *> m_parameters_d;
  
  /// Parameter matrix height
  int m_height;
  /// Parameter matrix width
  int m_width;
  /// Parameter matrix format
  matrix_format m_matrix_format;
  /// Learning rate
  DataType m_learning_rate;
};

/// Optimizer factory base class
class optimizer_factory {
 public:
  /// Constructor
  optimizer_factory
  (lbann_comm *comm,
   const std::string name);
  /// Destructor
  virtual ~optimizer_factory();
  /// Create optimizer; caller is responsible for freeing memory.
  virtual optimizer *create_optimizer() = 0;
  /// Get optimizer name
  virtual const std::string name() const {
    return m_name;
  };
 protected:
  /// LBANN communicator
  lbann_comm *m_comm;
 private:
  /// Optimizer name
  const std::string m_name;
};

} // namespace lbann

#endif // LBANN_OPTIMIZER_HPP
