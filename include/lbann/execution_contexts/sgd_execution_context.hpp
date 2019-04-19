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

struct sgd_termination_criteria : public termination_criteria {
  El::Int num_epochs;
};


/** @brief SGD Uses the step to track the Current mini-batch step for
  *  execution mode.
  *  @detailed Step counts are not reset after each epoch.
  */
class sgd_execution_context final : public execution_context {
public:
  /** Constructor. */
  sgd_execution_context(observing_ptr<trainer> trainer, lbann_comm *comm, execution_mode mode, int mini_batch_size);
  /** Destructor. */
  virtual ~sgd_execution_context() = default;

  /** Copy constructor. */
  sgd_execution_context(const sgd_execution_context& other) = default;
  /** Copy assignment operator. */
  sgd_execution_context& operator=(const sgd_execution_context& other) = default;
  /** Move constructor. */
  sgd_execution_context(sgd_execution_context&& other) = default;
  /** Move assignment operator. */
  sgd_execution_context& operator=(sgd_execution_context&& other) = default;
  /** Copy sgd_execution_context. */
  virtual std::unique_ptr<execution_context> copy_execution_context() const { return make_unique<sgd_execution_context>(*this); }

  /** Number of times the training set has been traversed. */
  inline El::Int get_epoch() const noexcept { return m_epoch; }

  /** @brief Increment the current epoch in the execution context
    *  @detailed Increment the counter tracking the number of times
    *  that the data set has been traversed.
    */
  void inc_epoch() noexcept { ++m_epoch; }

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

  /** Checkpoint training_algorithm to given file descriptor, return number of bytes written */
  virtual bool save_to_checkpoint_shared(persist& p);
  /** Restore training_algorithm by reading checkpoint from given file descriptor, return number of bytes read */
  virtual bool load_from_checkpoint_shared(persist& p);
  virtual bool save_to_checkpoint_distributed(persist& p);
  virtual bool load_from_checkpoint_distributed(persist& p);

private:
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
