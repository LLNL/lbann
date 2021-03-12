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

class sgd_termination_criteria : public termination_criteria
{
public:
  ~sgd_termination_criteria() = default;
  size_t num_epochs;
};

/** @brief SGD Uses the step to track the Current mini-batch step for
 *  execution mode.
 *  @details Step counts are not reset after each epoch.
 */
class sgd_execution_context final : public execution_context
{
public:
  /** Constructor. */
  sgd_execution_context(trainer& trainer,
                        training_algorithm& training_alg,
                        execution_mode mode,
                        size_t mini_batch_size);
  /** Destructor. */
  virtual ~sgd_execution_context() = default;

  /** Copy constructor. */
  sgd_execution_context(const sgd_execution_context& other) = default;
  /** Copy assignment operator. */
  sgd_execution_context&
  operator=(const sgd_execution_context& other) = default;
  /** Move constructor. */
  sgd_execution_context(sgd_execution_context&& other) = default;
  /** Move assignment operator. */
  sgd_execution_context& operator=(sgd_execution_context&& other) = default;
  /** Copy sgd_execution_context. */
  std::unique_ptr<execution_context> copy_execution_context() const override
  {
    return make_unique<sgd_execution_context>(*this);
  }

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize(Archive& ar);

  /** @brief Return the state of the execution context as a string */
  std::string get_state_string() const noexcept override
  {
    return build_string("sgd.",
                        to_string(get_execution_mode()),
                        ".epoch.",
                        get_epoch(),
                        ".step.",
                        get_step());
  }

  /** Number of times the training set has been traversed. */
  inline size_t get_epoch() const noexcept { return m_epoch; }

  /** @brief Increment the current epoch in the execution context
   *  @details Increment the counter tracking the number of times
   *  that the data set has been traversed.
   */
  void inc_epoch() noexcept { ++m_epoch; }

  /** Set the trainer's current mini-batch size. */
  inline void set_current_mini_batch_size(size_t mini_batch_size)
  {
    m_current_mini_batch_size = mini_batch_size;
  }
  /** Get the trainer's current mini-batch size. */
  inline size_t get_current_mini_batch_size() const
  {
    return m_current_mini_batch_size;
  }
  /** Get the trainer's effective mini-batch size. */
  inline size_t get_effective_mini_batch_size() const
  {
    return m_effective_mini_batch_size;
  }
  /** Set the trainer's effective mini-batch size. */
  inline void set_effective_mini_batch_size(size_t mini_batch_size)
  {
    m_effective_mini_batch_size = mini_batch_size;
  }

  /** Checkpoint training_algorithm to given file descriptor  */
  void save_to_checkpoint_shared(persist& p) override;
  /** Restore training_algorithm by reading checkpoint from given file
   * descriptor */
  void load_from_checkpoint_shared(persist& p) override;
  void save_to_checkpoint_distributed(persist& p) override;
  void load_from_checkpoint_distributed(persist& p) override;

private:
  friend class cereal::access;
  sgd_execution_context() = default;

private:
  /** Number of times the training data set has been traversed. */
  size_t m_epoch = 0;

  /** Size of the current mini-batch in the model. */
  size_t m_current_mini_batch_size;

  /** The "effective" size of a minibatch.
   *
   *  This is the size of the minibatch across all models and used for
   *  e.g.  correctly averaging gradients from multiple models.
   */
  size_t m_effective_mini_batch_size;
};

} // namespace lbann

#endif // LBANN_SGD_EXECUTION_CONTEXT_HPP
