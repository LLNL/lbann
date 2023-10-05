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

#ifndef LBANN_SGD_EXECUTION_CONTEXT_HPP
#define LBANN_SGD_EXECUTION_CONTEXT_HPP

#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

/** @brief SGD Uses the step to track the Current mini-batch step for
 *  execution mode.
 *  @details Step counts are not reset after each epoch.
 */
class SGDExecutionContext final : public ExecutionContext
{
public:
  /** Constructor. */
  SGDExecutionContext(execution_mode mode, size_t mini_batch_size);
  /** Destructor. */
  virtual ~SGDExecutionContext() = default;

  /** Move constructor. */
  SGDExecutionContext(SGDExecutionContext&& other) = default;
  /** Move assignment operator. */
  SGDExecutionContext& operator=(SGDExecutionContext&& other) = default;
  /** @brief Get a clean sgd_execution_context. */

  /** Copy constructor -- deleted. */
  SGDExecutionContext(const SGDExecutionContext& other) = delete;
  /** Copy assignment operator -- deleted. */
  SGDExecutionContext& operator=(const SGDExecutionContext& other) = delete;

  std::unique_ptr<ExecutionContext> get_new() const override
  {
    return std::make_unique<SGDExecutionContext>(execution_mode::invalid, 0UL);
  }

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

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

  /** Checkpoint training_algorithm to given file descriptor  */
  void save_to_checkpoint_shared(persist& p) override;
  /** Restore training_algorithm by reading checkpoint from given file
   * descriptor */
  void load_from_checkpoint_shared(persist& p) override;
  void save_to_checkpoint_distributed(persist& p) override;
  void load_from_checkpoint_distributed(persist& p) override;

  std::string get_type() const override;

  /** Get the mode that the trainer is currenting executing. */
  void set_execution_mode(execution_mode mode) noexcept
  {
    m_execution_mode = mode;
  }

  /** Get the mode that the trainer is currenting executing. */
  execution_mode get_execution_mode() const noexcept override
  {
    return m_execution_mode;
  }

  void set_early_stop(bool stop) noexcept { m_stop_early = stop; }
  bool get_early_stop() const noexcept { return m_stop_early; }

  void start_timer() noexcept { m_timer.start(); }
  void stop_timer() noexcept { m_timer.stop(); }
  double get_current_execution_time() const noexcept { return m_timer.check(); }

private:
  friend class cereal::access;
  SGDExecutionContext() = default;

private:
  /** @brief Timer tracking execution time. */
  lbann::Timer m_timer;

  /** Number of times the training data set has been traversed. */
  size_t m_epoch = 0;

  /** Size of the current mini-batch in the model.
   *
   *  Number of samples being processed in the current step (iteration),
   *  used for correctly averaging gradients.
   */
  size_t m_current_mini_batch_size;

  execution_mode m_execution_mode;

  bool m_stop_early = false;
};

/** @brief Base class for SGD stopping. */
class SGDTerminationCriteria
  : public TerminationCriteria,
    public Cloneable<HasAbstractFunction<SGDTerminationCriteria>>
{
public:
  SGDTerminationCriteria() = default;
  virtual ~SGDTerminationCriteria() = default;
  bool operator()(ExecutionContext const& c_in) const final
  {
    auto const& c = dynamic_cast<SGDExecutionContext const&>(c_in);
    return c.get_early_stop() || this->is_done(c);
  }

private:
  virtual bool is_done(SGDExecutionContext const& c) const noexcept = 0;
};

/** @brief Stop SGD based on a fixed batch count.
 *
 *  The training algorithm still tracks the epoch count for other
 *  parts of the code (e.g. at_epoch_begin/end callbacks).
 */
class BatchTerminationCriteria
  : public Cloneable<BatchTerminationCriteria, SGDTerminationCriteria>
{
public:
  BatchTerminationCriteria(size_t num_batches) : m_max_batches{num_batches} {}

private:
  bool is_done(SGDExecutionContext const& c) const noexcept final
  {
    return c.get_step() >= m_max_batches;
  }

private:
  size_t m_max_batches;
};

class EpochTerminationCriteria
  : public Cloneable<EpochTerminationCriteria, SGDTerminationCriteria>
{
public:
  EpochTerminationCriteria(size_t num_epochs) : m_max_epochs{num_epochs} {}

private:
  bool is_done(SGDExecutionContext const& c) const noexcept final
  {
    return c.get_epoch() >= m_max_epochs;
  }

private:
  size_t m_max_epochs;
};

class SecondsTerminationCriteria
  : public Cloneable<SecondsTerminationCriteria, SGDTerminationCriteria>
{
public:
  SecondsTerminationCriteria(double seconds) : m_max_seconds{seconds} {}

private:
  bool is_done(SGDExecutionContext const& c) const noexcept final;

private:
  double m_max_seconds;
};
} // namespace lbann

#endif // LBANN_SGD_EXECUTION_CONTEXT_HPP
