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

#include "lbann/execution_algorithms/kfac.hpp"
#include "lbann/execution_algorithms/kfac/execution_context.hpp"
#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"
#include "lbann/base.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/memory.hpp"

#include <training_algorithm.pb.h>

#include <cstddef>
#include <limits>

namespace lbann {

KFAC::KFAC(std::string name,
           std::unique_ptr<TermCriteriaType> stop)
  : BaseType{std::move(name)},
    m_stopping_criteria{std::move(stop)}
{}

KFAC::KFAC(KFAC const& other)
  : BaseType(other.get_name()),
    m_stopping_criteria{other.m_stopping_criteria->clone()}
{}

KFAC& KFAC::operator=(KFAC const& other)
{
  BaseType::operator=(other);
  m_stopping_criteria = other.m_stopping_criteria->clone();
  return *this;
}

// =============================================
// Evaluation and training
// =============================================

void KFAC::apply(
  execution_context& context_,
  model& model,
  data_coordinator& dc,
  execution_mode mode)
{
  ExeContextType& context = dynamic_cast<ExeContextType&>(context_);
  if (mode == execution_mode::training) {
    train(context, model, dc, *m_stopping_criteria);
  }
  else {
    sgd_training_algorithm eval_algo(
      this->get_name()+"_eval",
      m_stopping_criteria->clone());
    auto& eval_context = context.get_sgd_execution_context();
    eval_algo.apply(eval_context, model, dc, mode);
  }
}

void KFAC::train(
  ExeContextType& kfac_context,
  model& model,
  data_coordinator& dc,
  TermCriteriaType const& term)
{
  // Initialize some state so it knows we're training now.
  auto& sgd_context = kfac_context.get_sgd_execution_context();
  sgd_context.set_execution_mode(execution_mode::training);
  model.reset_mode(sgd_context, execution_mode::training);
  dc.reset_mode(sgd_context);

  // Run callbacks.
  do_train_begin_cbs(model);

  // Start iterating
  bool is_start_of_epoch = true;
  sgd_context.start_timer();
  while (!term(sgd_context)) {

    if (is_start_of_epoch) {
      // Initialize epoch
      model.reset_mode(sgd_context, execution_mode::training);
      model.reset_epoch_statistics(execution_mode::training);
      dc.reset_mode(sgd_context);
      do_epoch_begin_cbs(model);
      is_start_of_epoch = false;
    }

    // Train a mini batch. Returns "true" if the data_coordinator
    // detects the end of an epoch.
    if (train_mini_batch(kfac_context, model, dc)) {
      // Finalize epoch
      sgd_context.inc_epoch();
      model.reconcile_weight_values();
      do_epoch_end_cbs(model);

      // Evaluate on validation set
      //
      // FIXME (trb 05/04/2021): Upon further refactor, this should
      // move out of the main training cycle and become part of an
      // "evaluation policy" or something of that nature, ideally with
      // its own context that we needn't know about.
      if (dc.is_execution_mode_valid(execution_mode::validation)) {
        const execution_mode eval_mode = execution_mode::validation;
        sgd_execution_context eval_context(
          eval_mode,
          dc.get_mini_batch_size(eval_mode));
        // FIXME (trb 05/05/2021): This hacks around a bad assumption
        // in the data store.
        // Note (tym 6/7/21): Copied from sgd_training_algorithm.cpp.
        size_t num_validation_epochs = 1UL;
        if (sgd_context.get_epoch() > 1UL) {
          eval_context.inc_epoch();
          ++num_validation_epochs;
        }
        sgd_training_algorithm eval_algo(
          this->get_name()+"_eval",
          make_unique<epoch_termination_criteria>(num_validation_epochs));
        eval_algo.apply(eval_context, model, dc, eval_mode);

        // FIXME (trb 06/07/21): The early stopping callback is part
        // of the evaluation callbacks but it's meant to affect
        // training. This fixes a bug in which the training context
        // was meant to stop but was never properly told.
        sgd_context.set_early_stop(eval_context.get_early_stop());

      }

      // Trigger new epoch stuff next iteration (if there is one).
      is_start_of_epoch = true;
    }
  }
  sgd_context.stop_timer();

  // Reset the model back to the training execution context prior to
  // end of training callbacks
  model.reset_mode(sgd_context, execution_mode::training);
  do_train_end_cbs(model);
}

// =============================================
// Mini-batch step
// =============================================

// Returns "true" if the data_coordinator detects the end of an epoch.
bool KFAC::train_mini_batch(
  ExeContextType& kfac_context,
  model& model,
  data_coordinator& dc)
{
  auto& sgd_context = kfac_context.get_sgd_execution_context();

  model.reset_mode(sgd_context, execution_mode::training);
  dc.reset_mode(sgd_context);
  do_batch_begin_cbs(model);

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
        sgd_context.get_current_mini_batch_size());

      // Backward prop step
      model.get_objective_function()->differentiate();
      model.backward_prop();
      model.get_objective_function()->compute_weight_regularization();

      // Finish evaluation.
      model.get_objective_function()->finish_evaluation(
        execution_mode::training,
        sgd_context.get_current_mini_batch_size());
      model.evaluate_metrics(execution_mode::training,
                             sgd_context.get_current_mini_batch_size());

      // Update step
      model.update_weights();
      model.update_layers();
#if defined(LBANN_HAVE_OMP_TASKLOOP)
    }
  }
#endif

  kfac_context.inc_step();
  sgd_context.inc_step();
  do_batch_end_cbs(model);
  return finished;
}

// =============================================
// Callbacks
// =============================================

void KFAC::do_train_begin_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_begin(&model);
  }
}

void KFAC::do_train_end_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_end(&model);
  }
}

void KFAC::do_epoch_begin_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_begin(&model);
  }
}

void KFAC::do_epoch_end_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_end(&model);
  }
}

void KFAC::do_batch_begin_cbs(model& model)
{
  sgd_execution_context& c =
    static_cast<sgd_execution_context&>(model.get_execution_context());
  for (const auto& cb : model.get_callbacks()) {
    if (c.get_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_begin(&model);
    }
  }
}

void KFAC::do_batch_end_cbs(model& model)
{
  sgd_execution_context& c =
    static_cast<sgd_execution_context&>(model.get_execution_context());
  for (const auto& cb : model.get_callbacks()) {
    if (c.get_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_end(&model);
    }
  }
}

std::string KFAC::get_type() const { return "KFAC"; }

kfac::ExecutionContext* KFAC::do_get_new_execution_context() const
{
  return new kfac::ExecutionContext(0UL);
}

} // namespace lbann

template <>
std::unique_ptr<lbann::KFAC> lbann::make<lbann::KFAC>(
  google::protobuf::Message const& msg_in)
{
  auto const& params =
    dynamic_cast<lbann_data::TrainingAlgorithm const&>(msg_in);

  lbann_data::KFAC kfac_params;
  LBANN_ASSERT(params.parameters().UnpackTo(&kfac_params));
  auto const& sgd_params = kfac_params.sgd();

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
  return make_unique<lbann::KFAC>(params.name(), std::move(stopping));
}
