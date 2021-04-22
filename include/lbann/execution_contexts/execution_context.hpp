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

#ifndef LBANN_EXECUTION_CONTEXT_HPP
#define LBANN_EXECUTION_CONTEXT_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/threads/thread_pool.hpp"

// Forward declaration
namespace cereal {
class access;
}

namespace lbann {

// Forward-declare this.
class trainer;
class training_algorithm;

class execution_context
{
public:
  /** Constructor. */
  execution_context();

  /** Destructor. */
  virtual ~execution_context() = default;

  /** Get a "clean" execution_context of the same type. */
  virtual std::unique_ptr<execution_context> get_new() const = 0;

  /** @brief Get a string identifying the type of execution context.
   *  @details Should match the training algorithm.
   *  @todo Absorb completely into `get_state_string()`.
   */
  virtual std::string get_type() const = 0;

  /** @brief Return the state of the execution context as a string */
  virtual std::string get_state_string() const noexcept = 0;

  virtual execution_mode get_execution_mode() const noexcept
  {
    return execution_mode::invalid;
  }

  /** @brief Current step in the training algorithm
   *  @details Step counts the number of iterations in the training
   *  algorithm's internal state
   */
  size_t get_step() const noexcept { return m_step; }

  /** @brief Increment the current step in the training algorithm
   *  @details Increment the step count in the training
   *  algorithm's internal state
   */
  void inc_step() noexcept { ++m_step; }

  /** Return true if the flag to stop training is set. */
  bool get_terminate_training() const { return m_terminate_training; }
  /** Set the terminate training flag (on or off). */
  void set_terminate_training(bool f) { m_terminate_training = f; }

  /** @name Checkpointing and Serialization */
  ///@{

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize(Archive& ar);

  /** @brief Checkpoint exection_context to a shared checkpoint. */
  virtual void save_to_checkpoint_shared(persist& p) = 0;
  /** @brief Restore execution_context from a shared checkpoint. */
  virtual void load_from_checkpoint_shared(persist& p) = 0;
  /** @brief Checkpoint exection_context to a distributed checkpoint. */
  virtual void save_to_checkpoint_distributed(persist& p) = 0;
  /** @brief Restore execution_context from a distributed checkpoint. */
  virtual void load_from_checkpoint_distributed(persist& p) = 0;
  ///@}

protected:
  friend class cereal::access;
  /** Copy constructor. */
  execution_context(const execution_context& other) = default;
  /** Copy assignment operator. */
  execution_context& operator=(const execution_context& other) = default;
  /** Move constructor. */
  execution_context(execution_context&& other) = default;
  /** Move assignment operator. */
  execution_context& operator=(execution_context&& other) = default;

private:

  /** @brief Current step in the training algorithm
   *  @details Step counts the number of iterations in the training
   *  algorithm's internal state
   */
  size_t m_step = 0UL;

  /** @brief Whether to terminate training.
   *  @details If true, training will terminate immediately before
   *  the next epoch.
   */
  bool m_terminate_training = false;
};

class termination_criteria
{
public:
  termination_criteria(size_t max_steps)
    : m_max_num_steps{max_steps}
  {}
  virtual ~termination_criteria() = default;
  bool satisfied(execution_context const& ctxt) const
  {
    return (ctxt.get_step() >= m_max_num_steps);
  }

  size_t max_num_steps() const noexcept { return m_max_num_steps; }

private:
  size_t m_max_num_steps;
};

} // namespace lbann
#endif // LBANN_EXECUTION_CONTEXT_HPP
