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

namespace lbann {

// Forward-declare this.
class trainer;
class training_algorithm;

class termination_criteria {
public:
  size_t num_steps;
};

class execution_context {
public:
  /** Constructor. */
  execution_context(trainer& trainer, training_algorithm& training_alg,
                    lbann_comm *comm, execution_mode mode);
  /** Destructor. */
  virtual ~execution_context() = default;

  /** Copy execution_context. */
  virtual std::unique_ptr<execution_context> copy_execution_context() const {
    // Use explicit construction of unique pointer since copy
    // constructor is protected and cannot be accessed in make_unique
    return std::unique_ptr<execution_context>{new execution_context(*this)};
  }

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize( Archive & ar );

  /** @brief Return the state of the execution context as a string */
  virtual std::string get_state_string() const noexcept {
    return build_string("ec.", to_string(get_execution_mode()),
                        ".step.", get_step());
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

  /** Get the mode that the trainer is currenting executing. */
  inline void set_execution_mode(execution_mode mode) noexcept {
    m_execution_mode = mode;
  }

  /** Get the mode that the trainer is currenting executing. */
  inline execution_mode get_execution_mode() const noexcept {
    return m_execution_mode;
  }

  /** Return true if the flag to stop training is set. */
  bool get_terminate_training() const {
    return m_terminate_training;
  }
  /** Set the terminate training flag (on or off). */
  void set_terminate_training(bool f) {
    m_terminate_training = f;
  }

  /** Grab the trainer from the execution context */
  const trainer& get_trainer() const {
    return *m_trainer;
  }

  trainer& get_trainer() {
    return const_cast<trainer&>(static_cast<const execution_context&>(*this).get_trainer());
  }

  const training_algorithm& get_training_algorithm() const {
    return *m_training_algorithm;
  }

  training_algorithm& get_training_algorithm() {
    return const_cast<training_algorithm&>(static_cast<const execution_context&>(*this).get_training_algorithm());
  }

  thread_pool& get_io_thread_pool() const;

  lbann_comm& get_comm() const {
    if (!m_comm) { LBANN_ERROR("m_comm is null"); }
    return *m_comm;
  };

  /** Checkpoint training_algorithm to given file descriptor */
  virtual void save_to_checkpoint_shared(persist& p);
  /** Restore training_algorithm by reading checkpoint from given file descriptor */
  virtual void load_from_checkpoint_shared(persist& p);
  virtual void save_to_checkpoint_distributed(persist& p);
  virtual void load_from_checkpoint_distributed(persist& p);

protected:
  /** Copy constructor. */
  execution_context(const execution_context& other) = default;
  /** Copy assignment operator. */
  execution_context& operator=(const execution_context& other) = default;
  /** Move constructor. */
  execution_context(execution_context&& other) = default;
  /** Move assignment operator. */
  execution_context& operator=(execution_context&& other) = default;

private:
  /** Pointer to the training context (execution environment) for the training algorithm */
  trainer* m_trainer;

  training_algorithm* m_training_algorithm;

  /** LBANN communicator. */
  observer_ptr<lbann_comm> m_comm;

  /** The trainer's current execution mode. */
  execution_mode m_execution_mode = execution_mode::training;

  /** @brief Current step in the training algorithm
    *  @details Step counts the number of iterations in the training
    *  algorithm's internal state
    */
  size_t m_step = 0;

  /** @brief Whether to terminate training.
   *  @details If true, training will terminate immediately before
   *  the next epoch.
   */
  bool m_terminate_training = false;
};

}  // namespace lbann

#endif  // LBANN_EXECUTION_CONTEXT_HPP
