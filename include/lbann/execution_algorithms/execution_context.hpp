////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_EXECUTION_ALGORITHMS_EXECUTION_CONTEXT_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_EXECUTION_CONTEXT_HPP_INCLUDED

#include "lbann/base.hpp"

#include <memory>
#include <string>

// Forward declaration
namespace cereal {
class access;
}

namespace lbann {

// Forward-declare this.
class persist;
class trainer;
class TrainingAlgorithm;

class ExecutionContext
{
public:
  /** Constructor. */
  ExecutionContext();

  /** Destructor. */
  virtual ~ExecutionContext() = default;

  /** Get a "clean" execution_context of the same type. */
  virtual std::unique_ptr<ExecutionContext> get_new() const = 0;

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

  /** @name Checkpointing and Serialization */
  ///@{

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

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
  ExecutionContext(const ExecutionContext& other) = delete;
  /** Copy assignment operator. */
  ExecutionContext& operator=(const ExecutionContext& other) = delete;
  /** Move constructor. */
  ExecutionContext(ExecutionContext&& other) = default;
  /** Move assignment operator. */
  ExecutionContext& operator=(ExecutionContext&& other) = default;

private:
  /** @brief Current step in the training algorithm
   *  @details Step counts the number of iterations in the training
   *  algorithm's internal state
   */
  size_t m_step = 0UL;
};

/** @brief Specifies when to stop a training algorithm.
 *
 *  The stopping criteria must be compatible with the training
 *  algorithm, and specifically its execution context, but can
 *  otherwise be anything meaningful in the context of that algorithm.
 */
class TerminationCriteria
{
public:
  TerminationCriteria() = default;
  virtual ~TerminationCriteria() = default;
  virtual bool operator()(ExecutionContext const& c) const = 0;
};

} // namespace lbann

#endif // LBANN_EXECUTION_ALGORITHMS_EXECUTION_CONTEXT_HPP_INCLUDED
