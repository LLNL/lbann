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

#ifndef LBANN_SGD_TRAINING_ALGORITHM_HPP
#define LBANN_SGD_TRAINING_ALGORITHM_HPP

#include "lbann/base.hpp"
#include "lbann/execution_algorithms/factory.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/memory.hpp"
#include <google/protobuf/message.h>
#include <memory>

namespace lbann {

/** @brief Base class for LBANN SGD-family training algorithms. */
class SGDTrainingAlgorithm
  : public Cloneable<SGDTrainingAlgorithm, TrainingAlgorithm>
{
  using BaseType = Cloneable<SGDTrainingAlgorithm, TrainingAlgorithm>;

public:
  /** @brief Construct with a name. */
  SGDTrainingAlgorithm(std::string name,
                         std::unique_ptr<SGDTerminationCriteria> stop)
    : BaseType{std::move(name)},
      m_stopping_criteria{std::move(stop)},
      m_validation_context{execution_mode::validation, 1UL},
      m_validation_epochs{1UL}
  {}

  SGDTrainingAlgorithm(const SGDTrainingAlgorithm& other);
  SGDTrainingAlgorithm&
  operator=(const SGDTrainingAlgorithm& other);

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
  std::unique_ptr<SGDExecutionContext>
  get_new_execution_context() const
  {
    return to_unique_ptr(this->do_get_new_execution_context());
  }

protected:
  /** Train model on one step / mini-batch of an SGD forward pass */
  virtual bool train_mini_batch(SGDExecutionContext& c,
                                model& model,
                                data_coordinator& dc);

  /** Evaluate model on one step / mini-batch of an SGD forward pass */
  virtual bool evaluate_mini_batch(SGDExecutionContext& c,
                                   model& model,
                                   data_coordinator& dc,
                                   execution_mode mode);

  ////////////////////////////////////////////////////////////
  // Callbacks
  ////////////////////////////////////////////////////////////

  /** Execute callbacks at start of training. */
  virtual void do_train_begin_cbs(model& model);
  /** Execute callbacks at end of training. */
  virtual void do_train_end_cbs(model& model);
  /** Execute callbacks at start of evaluation. */
  virtual void do_evaluate_begin_cbs(model& model, execution_mode mode);
  /** Execute callbacks at end of evaluation. */
  virtual void do_evaluate_end_cbs(model& model, execution_mode mode);
  /** Execute callbacks at start of epoch. */
  virtual void do_epoch_begin_cbs(model& model);
  /** Execute callbacks at end of epoch. */
  virtual void do_epoch_end_cbs(model& model);
  /** Execute callbacks at start of mini-batch. */
  virtual void do_batch_begin_cbs(model& model, execution_mode mode);
  /** Execute callbacks at end of mini-batch. */
  virtual void do_batch_end_cbs(model& model, execution_mode mode);

  SGDExecutionContext*
  do_get_new_execution_context() const override;

private:
  std::unique_ptr<SGDTerminationCriteria> m_stopping_criteria;

  // FIXME (trb 07/20/21): This is a hack. These aren't actually
  // copyable objects (it wouldn't make sense), so when the training
  // algorithm is copied, these are reset to defaults. "In the
  // future", we'll externalize validation and this won't be an issue.
  SGDExecutionContext m_validation_context;
  size_t m_validation_epochs;
};

template <>
std::unique_ptr<SGDTrainingAlgorithm>
make<SGDTrainingAlgorithm>(google::protobuf::Message const& params);

} // namespace lbann

#endif // LBANN_SGD_TRAINING_ALGORITHM_HPP
