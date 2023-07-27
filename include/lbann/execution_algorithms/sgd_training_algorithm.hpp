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

#ifndef LBANN_SGD_TRAINING_ALGORITHM_HPP
#define LBANN_SGD_TRAINING_ALGORITHM_HPP

#include "lbann/base.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/execution_algorithms/factory.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/timer_map.hpp"
#ifdef LBANN_HAS_GPU
#include "lbann/utils/gpu/helpers.hpp"
#endif // LBANN_HAS_GPU

#include <google/protobuf/message.h>

#include <memory>

namespace lbann {

/** @brief Base class for LBANN SGD-family training algorithms. */
class SGDTrainingAlgorithm : public TrainingAlgorithm
{
public:
  /** @brief Construct with a name. */
  SGDTrainingAlgorithm(std::string name,
                       std::unique_ptr<SGDTerminationCriteria> stop,
                       bool suppress_timer_output);

  SGDTrainingAlgorithm(const SGDTrainingAlgorithm& other) = delete;
  SGDTrainingAlgorithm& operator=(const SGDTrainingAlgorithm& other) = delete;

  SGDTrainingAlgorithm(SGDTrainingAlgorithm&& other) = default;
  SGDTrainingAlgorithm& operator=(SGDTrainingAlgorithm&& other) = default;

  virtual ~SGDTrainingAlgorithm() = default;
  /** Copy training_algorithm. */
  //  virtual sgd_training_algorithm* copy() const = default;

  std::string get_type() const override;

  // ===========================================
  // Execution
  // ===========================================

  /** Apply the training algorithm to the model with the provided
      context and execution mode */
  void apply(ExecutionContext& c,
             model& model,
             data_coordinator& dc,
             execution_mode mode) override;

  /** Train a model using an iterative SGD solver. */
  void train(SGDExecutionContext& c,
             model& model,
             data_coordinator& dc,
             SGDTerminationCriteria const& term);

  /** Evaluate a model using the forward pass of an SGD solver. */
  void evaluate(SGDExecutionContext& c,
                model& model,
                data_coordinator& dc,
                execution_mode mode,
                SGDTerminationCriteria const& term);

  /** @brief Get a default-initialized execution context.
   *  @note This method participates in the
   *        "covariant-smart-pointer-return" pattern. In particular,
   *        it hides the base-class method to give the illusion of a
   *        covariant return.
   */
  std::unique_ptr<SGDExecutionContext> get_new_execution_context() const;

protected:
  /** Train model on one step / mini-batch of an SGD forward pass */
  bool train_mini_batch(SGDExecutionContext& c,
                        model& model,
                        data_coordinator& dc,
                        ScopeTimer timer);

  /** Evaluate model on one step / mini-batch of an SGD forward pass */
  bool evaluate_mini_batch(SGDExecutionContext& c,
                           model& model,
                           data_coordinator& dc,
                           execution_mode mode,
                           ScopeTimer timer);

  ////////////////////////////////////////////////////////////
  // Callbacks
  ////////////////////////////////////////////////////////////

  /** Execute callbacks at start of training. */
  void do_train_begin_cbs(model& model, ScopeTimer timer);
  /** Execute callbacks at end of training. */
  void do_train_end_cbs(model& model, ScopeTimer timer);
  /** Execute callbacks at start of evaluation. */
  void
  do_evaluate_begin_cbs(model& model, execution_mode mode, ScopeTimer timer);
  /** Execute callbacks at end of evaluation. */
  void do_evaluate_end_cbs(model& model, execution_mode mode, ScopeTimer timer);
  /** Execute callbacks at start of epoch. */
  void do_epoch_begin_cbs(model& model, ScopeTimer timer);
  /** Execute callbacks at end of epoch. */
  void do_epoch_end_cbs(model& model, ScopeTimer timer);
  /** Execute callbacks at start of mini-batch. */
  void do_batch_begin_cbs(model& model, execution_mode mode, ScopeTimer timer);
  /** Execute callbacks at end of mini-batch. */
  void do_batch_end_cbs(model& model, execution_mode mode, ScopeTimer timer);

  SGDExecutionContext* do_get_new_execution_context() const override;

private:
  TimerMap m_timers;
  std::unique_ptr<SGDTerminationCriteria> m_stopping_criteria;

  // FIXME (trb 07/20/21): This is a hack. These aren't actually
  // copyable objects (it wouldn't make sense), so when the training
  // algorithm is copied, these are reset to defaults. "In the
  // future", we'll externalize validation and this won't be an issue.
  SGDExecutionContext m_validation_context;
  size_t m_validation_epochs;

  /** @brief Suppress timer output.
   *  @deprecated This is a temporary way to disable timer
   *              output. This will be more configurable in the
   *              future.
   */
  bool m_suppress_timer = false;

#ifdef LBANN_HAS_GPU
  gpu_lib::event_wrapper m_data_prefetch_sync_event;
#endif // LBANN_HAS_GPU
};

template <>
std::unique_ptr<SGDTrainingAlgorithm>
make<SGDTrainingAlgorithm>(google::protobuf::Message const& params);

} // namespace lbann

#endif // LBANN_SGD_TRAINING_ALGORITHM_HPP
