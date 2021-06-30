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

#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"
#include "lbann/base.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/memory.hpp"

#include <training_algorithm.pb.h>

#include <cstddef>
#include <limits>

namespace lbann {

sgd_training_algorithm::sgd_training_algorithm(
  sgd_training_algorithm const& other)
  : BaseType(other.get_name()),
    m_stopping_criteria{other.m_stopping_criteria->clone()}
{}

sgd_training_algorithm&
sgd_training_algorithm::operator=(sgd_training_algorithm const& other)
{
  BaseType::operator=(other);
  m_stopping_criteria = other.m_stopping_criteria->clone();
  return *this;
}

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

void sgd_training_algorithm::apply(execution_context& context,
                                   model& model,
                                   data_coordinator& dc,
                                   execution_mode mode)
{
  sgd_execution_context& sgd_context =
    dynamic_cast<sgd_execution_context&>(context);
  const sgd_termination_criteria& sgd_term = *m_stopping_criteria;
  switch (mode) {
  case execution_mode::training:
    train(sgd_context, model, dc, sgd_term);
    break;
  case execution_mode::validation:
  case execution_mode::testing:
  case execution_mode::prediction:
    evaluate(sgd_context, model, dc, mode, sgd_term);
    break;
  default:
    LBANN_ERROR(std::string{} + "Illegal mode: " + to_string(mode));
  }
}

void sgd_training_algorithm::train(sgd_execution_context& c,
                                   model& model,
                                   data_coordinator& dc,
                                   sgd_termination_criteria const& term)
{
  // Setup a "training-global" validation context:
  using ValidationContext = sgd_execution_context;
  ValidationContext evaluation_context(
    execution_mode::validation,
    dc.get_mini_batch_size(execution_mode::validation));
  size_t num_validation_epochs = 1UL;

  // Initialize some state so it knows we're training now.
  c.set_execution_mode(execution_mode::training);
  model.reset_mode(c, execution_mode::training);
  dc.reset_mode(c);

  // Run callbacks.
  do_train_begin_cbs(model);

  // Start iterating
  bool is_start_of_epoch = true;
  c.start_timer();
  while (!term(c)) {

    if (is_start_of_epoch) {
      // Initialize epoch
      model.reset_mode(c, execution_mode::training);
      model.reset_epoch_statistics(execution_mode::training);
      dc.reset_mode(c);
      do_epoch_begin_cbs(model);
      is_start_of_epoch = false;
    }

    // Train a mini batch. Returns "true" if the data_coordinator
    // detects the end of an epoch.
    if (train_mini_batch(c, model, dc)) {
      // Finalize epoch
      c.inc_epoch();
      model.reconcile_weight_values();
      do_epoch_end_cbs(model);

      // Evaluate on validation set
      //
      // FIXME (trb 05/04/2021): Upon further refactor, this should
      // move out of the main training cycle and become part of an
      // "evaluation policy" or something of that nature, ideally with
      // its own context that we needn't know about.
      if (dc.is_execution_mode_valid(execution_mode::validation)) {
        evaluate(evaluation_context,
                 model,
                 dc,
                 execution_mode::validation,
                 epoch_termination_criteria(num_validation_epochs));
        ++num_validation_epochs;

        // FIXME (trb 06/07/21): The early stopping callback is part
        // of the evaluation callbacks but it's meant to affect
        // training. This fixes a bug in which the training context
        // was meant to stop but was never properly told.
        c.set_early_stop(evaluation_context.get_early_stop());
      }

      // Trigger new epoch stuff next iteration (if there is one).
      is_start_of_epoch = true;
    }
  }
  c.stop_timer();

  // Reset the model back to the training execution context prior to
  // end of training callbacks
  model.reset_mode(c, execution_mode::training);
  do_train_end_cbs(model);
}

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

// Returns "true" if the data_coordinator detects the end of an epoch.
bool sgd_training_algorithm::train_mini_batch(sgd_execution_context& c,
                                              model& model,
                                              data_coordinator& dc)
{
  model.reset_mode(c, execution_mode::training);
  dc.reset_mode(c);
  do_batch_begin_cbs(model, execution_mode::training);

  bool finished = false;

  dc.fetch_data(execution_mode::training);

#if defined(LBANN_HAVE_OMP_TASKLOOP)
  LBANN_OMP_PARALLEL
  {
#pragma omp single
    {
#endif
      // Forward prop step
      model.clear_gradients();
      model.forward_prop(execution_mode::training);
      // check if the data coordinator has finished the epoch and kickoff
      // background I/O
      finished = dc.epoch_complete(execution_mode::training);

      // Result is not needed until the end of the mini-batch.
      model.get_objective_function()->start_evaluation(
        execution_mode::training,
        c.get_current_mini_batch_size());

      // Backward prop step
      model.get_objective_function()->differentiate();
      model.backward_prop();
      model.get_objective_function()->compute_weight_regularization();

      // Finish evaluation.
      model.get_objective_function()->finish_evaluation(
        execution_mode::training,
        c.get_current_mini_batch_size());
      model.evaluate_metrics(execution_mode::training,
                             c.get_current_mini_batch_size());

      // Update step
      model.update_weights();
      model.update_layers();
#if defined(LBANN_HAVE_OMP_TASKLOOP)
    }
  }
#endif

  c.inc_step();
  do_batch_end_cbs(model, execution_mode::training);
  return finished;
}

void sgd_training_algorithm::evaluate(sgd_execution_context& c,
                                      model& model,
                                      data_coordinator& dc,
                                      execution_mode mode,
                                      sgd_termination_criteria const& term)
{
  /// @todo BVE FIXME this state needs to be set for inference-only
  /// workflows -- however, if the model will bail due to a lack of a
  /// valid mode, the state of the data coordinator is not
  /// consistent.  Fix this once the data coordinator is fully
  /// decoupled from the input layer.
  model.reset_epoch_statistics(mode);
  model.reset_mode(c, mode);
  // Ensure that the data coordinator has the right execution context
  dc.reset_mode(c);
  // Return early if execution mode is invalid
  if (!dc.is_execution_mode_valid(mode))
    return;
  if (mode != execution_mode::validation &&
      mode != execution_mode::tournament && mode != execution_mode::testing) {
    LBANN_ERROR("invalid execution mode for evaluation");
  }

  // Evaluate on all mini-batches
  do_evaluate_begin_cbs(model, mode);
  while (!term(c)) {
    if (evaluate_mini_batch(c, model, dc, mode))
      c.inc_epoch();
  }
  do_evaluate_end_cbs(model, mode);
}

bool sgd_training_algorithm::evaluate_mini_batch(sgd_execution_context& c,
                                                 model& model,
                                                 data_coordinator& dc,
                                                 execution_mode mode)
{
  model.reset_mode(c, mode);
  dc.reset_mode(c);
  do_batch_begin_cbs(model, mode);
  dc.fetch_data(mode);
  model.forward_prop(mode);
  // check if the data coordinator has finished the epoch and kickoff
  // background I/O
  const bool finished = dc.epoch_complete(mode);

  model.get_objective_function()->start_evaluation(
    mode,
    c.get_current_mini_batch_size());
  model.get_objective_function()->finish_evaluation(
    mode,
    c.get_current_mini_batch_size());
  model.evaluate_metrics(mode, c.get_current_mini_batch_size());
  model.update_layers();
  c.inc_step();
  do_batch_end_cbs(model, mode);
  return finished;
}

////////////////////////////////////////////////////////////
// Callbacks
////////////////////////////////////////////////////////////

void sgd_training_algorithm::do_train_begin_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_begin(&model);
  }
}

void sgd_training_algorithm::do_train_end_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_end(&model);
  }
}

void sgd_training_algorithm::do_evaluate_begin_cbs(model& model,
                                                   execution_mode mode)
{
  for (const auto& cb : model.get_callbacks()) {
    switch (mode) {
    case execution_mode::validation:
      cb->on_validation_begin(&model);
      break;
    case execution_mode::tournament:
      cb->on_validation_begin(&model);
      break;
    case execution_mode::testing:
      cb->on_test_begin(&model);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void sgd_training_algorithm::do_evaluate_end_cbs(model& model,
                                                 execution_mode mode)
{
  for (const auto& cb : model.get_callbacks()) {
    switch (mode) {
    case execution_mode::validation:
      cb->on_validation_end(&model);
      break;
    case execution_mode::tournament:
      cb->on_validation_end(&model);
      break;
    case execution_mode::testing:
      cb->on_test_end(&model);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void sgd_training_algorithm::do_epoch_begin_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_begin(&model);
  }
}

void sgd_training_algorithm::do_epoch_end_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_end(&model);
  }
}

void sgd_training_algorithm::do_batch_begin_cbs(model& model,
                                                execution_mode mode)
{
  sgd_execution_context& c =
    static_cast<sgd_execution_context&>(model.get_execution_context());

  for (const auto& cb : model.get_callbacks()) {
    switch (mode) {
    case execution_mode::training:
      if (c.get_step() % cb->get_batch_interval() == 0) {
        cb->on_batch_begin(&model);
      }
      break;
    case execution_mode::validation:
    case execution_mode::tournament:
    case execution_mode::testing:
      cb->on_batch_evaluate_begin(&model);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void sgd_training_algorithm::do_batch_end_cbs(model& model, execution_mode mode)
{
  sgd_execution_context& c =
    static_cast<sgd_execution_context&>(model.get_execution_context());

  for (const auto& cb : model.get_callbacks()) {
    switch (mode) {
    case execution_mode::training:
      if (c.get_step() % cb->get_batch_interval() == 0) {
        cb->on_batch_end(&model);
      }
      break;
    case execution_mode::validation:
    case execution_mode::tournament:
    case execution_mode::testing:
      cb->on_batch_evaluate_end(&model);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

std::string sgd_training_algorithm::get_type() const { return "sgd"; }

sgd_execution_context*
sgd_training_algorithm::do_get_new_execution_context() const
{
  return new sgd_execution_context(execution_mode::invalid, 0);
}
} // namespace lbann

template <>
std::unique_ptr<lbann::sgd_training_algorithm>
lbann::make<lbann::sgd_training_algorithm>(
  google::protobuf::Message const& msg_in)
{
  auto const& params =
    dynamic_cast<lbann_data::TrainingAlgorithm const&>(msg_in);

  lbann_data::SGD sgd_params;
  LBANN_ASSERT(params.parameters().UnpackTo(&sgd_params));

  auto const& stopping_criteria = sgd_params.stopping_criteria();
  std::unique_ptr<lbann::sgd_termination_criteria> stopping;
  switch (stopping_criteria.criterion_case()) {
  case lbann_data::SGD::TerminationCriteria::kMaxBatches:
    stopping = lbann::make_unique<lbann::batch_termination_criteria>(
      stopping_criteria.max_batches());
    break;
  case lbann_data::SGD::TerminationCriteria::kMaxEpochs:
    stopping = lbann::make_unique<lbann::epoch_termination_criteria>(
      stopping_criteria.max_epochs());
    break;
  case lbann_data::SGD::TerminationCriteria::kMaxSeconds:
    stopping = lbann::make_unique<lbann::seconds_termination_criteria>(
      stopping_criteria.max_seconds());
    //LBANN_ERROR("Time-based training not yet supported in SGD.");
    break;
  default:
    LBANN_ERROR("No stopping criteria specified.");
  }
  return make_unique<sgd_training_algorithm>(params.name(),
                                             std::move(stopping));
}
