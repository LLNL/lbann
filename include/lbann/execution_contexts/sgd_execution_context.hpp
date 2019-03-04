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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_SGD_EXECUTION_CONTEXT_HPP
#define LBANN_SGD_EXECUTION_CONTEXT_HPP

#include "lbann/execution_contexts/execution_context.hpp"

namespace lbann {

class sgd_termination_criteria : public termination_criteria {
public:
  El::Int num_epochs;
};

class sgd_execution_context : public execution_context {
public:
  /** Constructor. */
  sgd_execution_context(observing_ptr<trainer> trainer, lbann_comm *comm, execution_mode mode, int mini_batch_size);

  /** Copy constructor. */
  sgd_execution_context(const sgd_execution_context& other) = default;
  /** Copy assignment operator. */
  sgd_execution_context& operator=(const sgd_execution_context& other) = default;
  /** Move constructor. */
  sgd_execution_context(sgd_execution_context&& other) = default;
  /** Move assignment operator. */
  sgd_execution_context& operator=(sgd_execution_context&& other) = default;
  /** Destructor. */
  virtual ~sgd_execution_context() = default;
  /** Copy sgd_execution_context. */
  //  virtual sgd_execution_context* copy() const = default;

  /** Number of times the training set has been traversed. */
  inline El::Int get_epoch() const noexcept { return m_epoch; }

  /** @brief Increment the current epoch in the execution context
    *  @detailed Increment the counter tracking the number of times
    *  that the data set has been traversed.
    */
  virtual void inc_epoch() noexcept { ++m_epoch; }

  /** @brief Current mini-batch step for execution mode.
    *  @detailed Step counts are not reset after each epoch.
    */
  El::Int get_step() const noexcept { return execution_context::get_step(); }

  /** Set the trainer's current mini-batch size. */
  inline void set_current_mini_batch_size(int mini_batch_size) {
    m_current_mini_batch_size = mini_batch_size;
  }
  /** Get the trainer's current mini-batch size. */
  inline int get_current_mini_batch_size() const {
    return m_current_mini_batch_size;
  }
  /** Get the trainer's effective mini-batch size. */
  inline int get_effective_mini_batch_size() const {
    return m_effective_mini_batch_size;
  }
  /** Set the trainer's effective mini-batch size. */
  inline void set_effective_mini_batch_size(int mini_batch_size) {
    m_effective_mini_batch_size = mini_batch_size;
  }

public:
  /** Number of times the training data set has been traversed. */
  El::Int m_epoch = 0;

  /** Size of the current mini-batch in the model. */
  int m_current_mini_batch_size;

  /** The "effective" size of a minibatch.
   *
   *  This is the size of the minibatch across all models and used for
   *  e.g.  correctly averaging gradients from multiple models.
   */
  int m_effective_mini_batch_size;
};

}  // namespace lbann

#endif  // LBANN_SGD_EXECUTION_CONTEXT_HPP
