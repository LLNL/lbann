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

#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"

#include "lbann/base.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/timer_map.hpp"

#include "lbann/proto/training_algorithm.pb.h"

// FIXME (trb): This is a quick hack for a short-term deliverable.
#ifdef LBANN_HAS_CALIPER
#include <adiak.hpp>
#endif

#include <iostream>
#include <memory>
#include <string>

namespace lbann {

SGDTrainingAlgorithm::SGDTrainingAlgorithm(
  std::string name,
  std::unique_ptr<SGDTerminationCriteria> stop,
  bool suppress_timer)
  : TrainingAlgorithm{std::move(name)},
    m_timers{"<default>"},
    m_stopping_criteria{std::move(stop)},
    m_validation_context{execution_mode::validation},
    m_validation_epochs{1UL},
    m_suppress_timer{suppress_timer}
{}

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

void SGDTrainingAlgorithm::apply(ExecutionContext& context,
                                 model& model,
                                 data_coordinator& dc,
                                 execution_mode mode)
{
  m_timers = TimerMap{build_string("SGD::",
                                   this->get_name(),
                                   " (trainer:",
                                   get_trainer().get_comm()->get_trainer_rank(),
                                   ")")};
  SGDExecutionContext& sgd_context =
    dynamic_cast<SGDExecutionContext&>(context);
  const SGDTerminationCriteria& sgd_term = *m_stopping_criteria;
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
    LBANN_ERROR("Illegal mode: ", to_string(mode));
  }
  if (!m_suppress_timer && model.get_comm()->am_trainer_master())
    m_timers.print(std::cout);
}

void SGDTrainingAlgorithm::train(SGDExecutionContext& c,
                                 model& model,
                                 data_coordinator& dc,
                                 SGDTerminationCriteria const& term)
{
  ScopeTimer train_timer(m_timers, "train()");

  auto& evaluation_context = m_validation_context;
  auto& num_validation_epochs = m_validation_epochs;

  // Initialize some state so it knows we're training now.
  c.set_execution_mode(execution_mode::training);
  model.reset_mode(c, execution_mode::training);
  dc.reset_mode(c);

  if (get_trainer().background_io_activity_allowed()) {
    // if (model.background_io_activity_allowed()) {
    // Fetch the first batch
    dc.fetch_active_batch_synchronous(execution_mode::training);
    El::Int current_mini_batch_size =
      dc.get_current_mini_batch_size(execution_mode::training);
    model.set_current_mini_batch_size(current_mini_batch_size);
  }

  // Run callbacks.
  do_train_begin_cbs(model, ScopeTimer{train_timer, "train_begin callbacks"});

  // Start iterating
  [[maybe_unused]] size_t total_batches = 0;
  c.start_timer();
  LBANN_CALIPER_LOOP_BEGIN(train_epoch, "train_epoch");
  for ([[maybe_unused]] size_t epoch = 0; !term(c); ++epoch) {

    LBANN_CALIPER_LOOP_ITER(train_epoch, epoch);

    // Initialize epoch
    model.reset_mode(c, execution_mode::training);
    model.reset_epoch_statistics(execution_mode::training);
    dc.reset_mode(c);
    do_epoch_begin_cbs(model, ScopeTimer{train_timer, "epoch_begin callbacks"});

    // Train a mini batch. Returns "true" if the data_coordinator
    // detects the end of an epoch. The termination criteria must be
    // checked every iteration as there may be a batch limit rather
    // than an epoch limit.
    LBANN_CALIPER_LOOP_BEGIN(
      train_batch,
      (epoch == 0UL ? "train_minibatch_epoch_0" : "train_minibatch"));
    bool end_of_epoch = false;
    while (!term(c) && !end_of_epoch) {
      LBANN_CALIPER_LOOP_ITER(train_batch, c.get_step());
      end_of_epoch =
        train_mini_batch(c,
                         model,
                         dc,
                         ScopeTimer{train_timer, "train minibatch"});
      ++total_batches;
    }
    LBANN_CALIPER_LOOP_END(train_batch);

    // Finalize epoch
    c.inc_epoch();

    // FIXME (tbennun 10/09/23): This is weight averaging and should not be done
    // unless requested. Commented out for now to test for any regressions.
    // model.reconcile_weight_values();

    do_epoch_end_cbs(model, ScopeTimer{train_timer, "epoch_end callbacks"});

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
               EpochTerminationCriteria(num_validation_epochs));
      ++num_validation_epochs;

      // FIXME (trb 06/07/21): The early stopping callback is part
      // of the evaluation callbacks but it's meant to affect
      // training. This fixes a bug in which the training context
      // was meant to stop but was never properly told.
      c.set_early_stop(evaluation_context.get_early_stop());
    }
  }
  LBANN_CALIPER_LOOP_END(train_epoch);
  c.stop_timer();
#ifdef LBANN_HAS_CALIPER
  if (is_caliper_initialized()) {
    adiak::value("total training epochs", c.get_epoch());
    adiak::value("total training batches", total_batches);
  }
#endif

  // Reset the model back to the training execution context prior to
  // end of training callbacks
  model.reset_mode(c, execution_mode::training);
  do_train_end_cbs(model, ScopeTimer{train_timer, "train_end callbacks"});
}

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

// Returns "true" if the data_coordinator detects the end of an epoch.
bool SGDTrainingAlgorithm::train_mini_batch(SGDExecutionContext& c,
                                            model& model,
                                            data_coordinator& dc,
                                            ScopeTimer timer)
{
  model.reset_mode(c, execution_mode::training);
  dc.reset_mode(c);
  do_batch_begin_cbs(model,
                     execution_mode::training,
                     ScopeTimer{timer, "batch_begin callbacks"});

  bool finished = false;

#ifdef LBANN_HAS_GPU
  m_data_prefetch_sync_event.synchronize();
#endif // LBANN_HAS_GPU

  if (get_trainer().background_io_activity_allowed()) {
    dc.fetch_data(execution_mode::training);
  }
  else {
    dc.fetch_active_batch_synchronous(execution_mode::training);
  }

  El::Int current_mini_batch_size =
    dc.get_current_mini_batch_size(execution_mode::training);
  model.set_current_mini_batch_size(current_mini_batch_size);

#if defined(LBANN_HAVE_OMP_TASKLOOP)
  LBANN_OMP_PARALLEL
  {
#pragma omp single
    {
#endif
      // Forward prop step
      model.clear_gradients();
      {
        ScopeTimer _{timer, "forward prop*"};
        model.forward_prop(execution_mode::training);
      }

#ifdef LBANN_HAS_GPU
      m_data_prefetch_sync_event.record(
        El::SyncInfo<El::Device::GPU>{}.Stream());
#endif // LBANN_HAS_GPU

      // Result is not needed until the end of the mini-batch.
      model.get_objective_function()->start_evaluation(execution_mode::training,
                                                       current_mini_batch_size);

      // Backward prop step
      model.get_objective_function()->differentiate();
      {
        ScopeTimer _{timer, "back prop*"};
        model.backward_prop();
      }
      model.get_objective_function()->compute_weight_regularization();

      // Swap buffers in data coordinator
      finished = dc.ready_for_next_fetch(execution_mode::training);

      // Finish evaluation.
      model.get_objective_function()->finish_evaluation(
        execution_mode::training,
        current_mini_batch_size);
      model.evaluate_metrics(execution_mode::training, current_mini_batch_size);

      // Update step
      model.update_weights();
      model.update_layers();
#if defined(LBANN_HAVE_OMP_TASKLOOP)
    }
  }
#endif

  c.inc_step();
  do_batch_end_cbs(model,
                   execution_mode::training,
                   ScopeTimer{timer, "batch_end callbacks"});
  return finished;
}

[[maybe_unused]] static char const* loop_label(execution_mode mode)
{
  switch (mode) {
  case execution_mode::validation:
    return "validation_batch";
  case execution_mode::testing:
    return "testing_batch";
  case execution_mode::prediction:
    return "prediction_batch";
  case execution_mode::tournament:
    return "tournament_batch";
  default:
    return "unknown_evaluation_batch";
  }
}

void SGDTrainingAlgorithm::evaluate(SGDExecutionContext& c,
                                    model& model,
                                    data_coordinator& dc,
                                    execution_mode mode,
                                    SGDTerminationCriteria const& term)
{
  ScopeTimer eval_timer{m_timers,
                        build_string("evaluate(", to_string(mode), ")")};

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

  if (get_trainer().background_io_activity_allowed()) {
    // Fetch the first step in an evaluation
    dc.fetch_active_batch_synchronous(mode);
    El::Int current_mini_batch_size = dc.get_current_mini_batch_size(mode);
    model.set_current_mini_batch_size(current_mini_batch_size);
  }

  // Evaluate on all mini-batches
  do_evaluate_begin_cbs(model,
                        mode,
                        ScopeTimer{eval_timer, "eval_begin callbacks"});
  LBANN_CALIPER_LOOP_BEGIN(eval_batch, loop_label(mode));
  while (!term(c)) {
    LBANN_CALIPER_LOOP_ITER(eval_batch, c.get_step());
    if (evaluate_mini_batch(c,
                            model,
                            dc,
                            mode,
                            ScopeTimer{eval_timer, "eval minibatch"}))
      c.inc_epoch();
  }
  LBANN_CALIPER_LOOP_END(eval_batch);
  do_evaluate_end_cbs(model,
                      mode,
                      ScopeTimer{eval_timer, "eval_end callbacks"});
}

bool SGDTrainingAlgorithm::evaluate_mini_batch(SGDExecutionContext& c,
                                               model& model,
                                               data_coordinator& dc,
                                               execution_mode mode,
                                               ScopeTimer timer)
{
  model.reset_mode(c, mode);
  dc.reset_mode(c);
  do_batch_begin_cbs(model, mode, ScopeTimer{timer, "batch_begin callbacks"});
  if (get_trainer().background_io_activity_allowed()) {
    dc.fetch_data(mode);
  }
  else {
    dc.fetch_active_batch_synchronous(mode);
  }
  El::Int current_mini_batch_size = dc.get_current_mini_batch_size(mode);
  model.set_current_mini_batch_size(current_mini_batch_size);
  // LBANN_MSG("evaluate mb thinks that the current mini-batch is ",
  //           current_mini_batch_size);
  model.forward_prop(mode);
  bool const finished = dc.ready_for_next_fetch(mode);

  model.get_objective_function()->start_evaluation(mode,
                                                   current_mini_batch_size);
  model.get_objective_function()->finish_evaluation(mode,
                                                    current_mini_batch_size);
  model.evaluate_metrics(mode, current_mini_batch_size);
  model.update_layers();
  c.inc_step();
  do_batch_end_cbs(model, mode, ScopeTimer{timer, "batch_end callbacks"});
  return finished;
}

std::unique_ptr<SGDExecutionContext>
SGDTrainingAlgorithm::get_new_execution_context() const
{
  return to_unique_ptr(this->do_get_new_execution_context());
}

////////////////////////////////////////////////////////////
// Callbacks
////////////////////////////////////////////////////////////

void SGDTrainingAlgorithm::do_train_begin_cbs(model& model, ScopeTimer timer)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_begin(&model);
  }
}

void SGDTrainingAlgorithm::do_train_end_cbs(model& model, ScopeTimer timer)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_end(&model);
  }
}

void SGDTrainingAlgorithm::do_evaluate_begin_cbs(model& model,
                                                 execution_mode mode,
                                                 ScopeTimer timer)
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

void SGDTrainingAlgorithm::do_evaluate_end_cbs(model& model,
                                               execution_mode mode,
                                               ScopeTimer timer)
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

void SGDTrainingAlgorithm::do_epoch_begin_cbs(model& model, ScopeTimer timer)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_begin(&model);
  }
}

void SGDTrainingAlgorithm::do_epoch_end_cbs(model& model, ScopeTimer timer)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_end(&model);
  }
}

void SGDTrainingAlgorithm::do_batch_begin_cbs(model& model,
                                              execution_mode mode,
                                              ScopeTimer timer)
{
  SGDExecutionContext& c =
    static_cast<SGDExecutionContext&>(model.get_execution_context());

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

void SGDTrainingAlgorithm::do_batch_end_cbs(model& model,
                                            execution_mode mode,
                                            ScopeTimer timer)
{
  SGDExecutionContext& c =
    static_cast<SGDExecutionContext&>(model.get_execution_context());

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

std::string SGDTrainingAlgorithm::get_type() const { return "sgd"; }

SGDExecutionContext* SGDTrainingAlgorithm::do_get_new_execution_context() const
{
  return new SGDExecutionContext(execution_mode::invalid);
}
} // namespace lbann

namespace {
using TermCriteria = lbann_data::SGD::TerminationCriteria;
using StoppingCriteriaFactory = lbann::generic_factory<
  lbann::SGDTerminationCriteria,
  int,
  lbann::generate_builder_type<lbann::SGDTerminationCriteria,
                               TermCriteria const&>>;

StoppingCriteriaFactory make_factory()
{
  using namespace lbann;
  using namespace std;
  StoppingCriteriaFactory factory;
  factory.register_builder(TermCriteria::kMaxBatches,
                           [](TermCriteria const& msg) {
                             return make_unique<BatchTerminationCriteria>(
                               msg.max_batches());
                           });
  factory.register_builder(TermCriteria::kMaxEpochs,
                           [](TermCriteria const& msg) {
                             return make_unique<EpochTerminationCriteria>(
                               msg.max_epochs());
                           });
  factory.register_builder(TermCriteria::kMaxSeconds,
                           [](TermCriteria const& msg) {
                             return make_unique<SecondsTerminationCriteria>(
                               msg.max_seconds());
                           });
  return factory;
}

StoppingCriteriaFactory& term_criteria_factory()
{
  static StoppingCriteriaFactory factory = make_factory();
  return factory;
}

} // namespace
template <>
std::unique_ptr<lbann::SGDTrainingAlgorithm>
lbann::make<lbann::SGDTrainingAlgorithm>(
  google::protobuf::Message const& msg_in)
{
  auto const& params =
    dynamic_cast<lbann_data::TrainingAlgorithm const&>(msg_in);

  lbann_data::SGD sgd_params;
  LBANN_ASSERT(params.parameters().UnpackTo(&sgd_params));

  auto const& stopping_criteria = sgd_params.stopping_criteria();
  auto stopping =
    term_criteria_factory().create_object(stopping_criteria.criterion_case(),
                                          stopping_criteria);
  return std::make_unique<SGDTrainingAlgorithm>(
    params.name(),
    std::move(stopping),
    sgd_params.suppress_timer_output());
}
