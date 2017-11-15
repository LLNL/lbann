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
#include "lbann/weights/weights.hpp"
#include <string>

namespace lbann {

/** Abstract optimizer. */
class optimizer {
 public:

  /** Constructor. */
  optimizer(DataType learning_rate = DataType(0),
            cudnn::cudnn_manager *cudnn = nullptr);

  /** Copy constructor. */
  optimizer(const optimizer& other);
  /** Copy assignment operator. */
  optimizer& operator=(const optimizer& other);
  /** Destructor. */
  virtual ~optimizer();
  /** Create a copy of the optimizer. */
  virtual optimizer* copy() const = 0;

  /** Get the optimizer name. */
  virtual std::string get_type() const = 0;
  /** Get a human-readable description of the optimizer. */
  virtual std::string get_description() const;

  /** Get weights being optimized. */
  weights& get_weights();
  /** Set weights being optimized. */
  void set_weights(weights& w) { m_weights = &w; }
  /** Get learning rate. */
  DataType get_learning_rate() const { return m_learning_rate; }
  /** Set learning rate. */
  void set_learning_rate(DataType learning_rate) {
    m_learning_rate = learning_rate;
  };

  /** Get gradient matrix. */
  AbsDistMat& get_gradient();
  /** Get gradient matrix (const). */
  const AbsDistMat& get_gradient() const;
  
  /** Clear gradient matrix. */
  void clear_gradient();
  /** Add to the gradient matrix. */
  void add_to_gradient(const AbsDistMat& gradient) {
    El::Axpy(DataType(1), gradient, get_gradient());
  }
  /** Allreduce and add to gradient matrix.
   *  The input is added to a staging matrix. When an optimization
   *  step is applied, an allreduce is applied over the redundant
   *  communicator of the staging matrix and the result is added to
   *  the gradient.
   */
  void allreduce_and_add_to_gradient(const AbsDistMat& gradient);

  /** Setup optimizer. */
  virtual void setup(weights& w);

  /** Apply an optimization step. */
  void step();
  /** Perform the computation in an optimization step.
   *  It can be assumed that values and gradient are the same size and
   *  have the same matrix distribution.
   */
  virtual void step_compute(AbsDistMat& values, AbsDistMat& gradient) = 0;

 protected:
 
  /** cuDNN manager. */
  cudnn::cudnn_manager* m_cudnn;

  /** Weights being optimized. */
  weights* m_weights;

  /** Learning rate. */
  DataType m_learning_rate;

  /** Gradient matrix. */
  AbsDistMat* m_gradient;

  /** Staging matrix for gradient allreduce.
   *  When an optimization step is applied, an allreduce is applied
   *  over the redundant communicator of the staging matrix and the
   *  result is added to the gradient matrix.
   */
  AbsDistMat* m_gradient_staging;

};

} // namespace lbann

#endif // LBANN_OPTIMIZER_HPP
