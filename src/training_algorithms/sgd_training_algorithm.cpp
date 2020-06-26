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

#include "lbann/training_algorithms/sgd_training_algorithm.hpp"
#include "lbann/models/model.hpp"
#include "lbann/callbacks/callback.hpp"

namespace lbann {

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

void sgd_training_algorithm::apply(execution_context& context,
                                   model& model,
                                   data_coordinator& dc,
                                   execution_mode mode,
                                   termination_criteria const& term_criteria) {
  sgd_execution_context& sgd_context = static_cast<sgd_execution_context&>(context);
  const sgd_termination_criteria& sgd_term = static_cast<const sgd_termination_criteria&>(term_criteria);
  switch(mode) {
  case execution_mode::training:
    train(sgd_context, model, dc, sgd_term.num_epochs, sgd_term.num_steps);
    break;
  case execution_mode::validation:
  case execution_mode::testing:
  case execution_mode::prediction:
    evaluate(sgd_context, model, dc, mode, sgd_term.num_steps);
    break;
  default:
    LBANN_ERROR(std::string{} + "Illegal mode: " + to_string(mode));
  }
}

void sgd_training_algorithm::train(sgd_execution_context& c,
                                   model& model,
                                   data_coordinator& dc,
                                   size_t num_epochs,
                                   size_t num_batches) {

  // Initialize epoch
  model.reset_mode(c, execution_mode::training);
  dc.reset_mode(c);

  do_train_begin_cbs(model);
  for (size_t epoch = c.get_epoch(); epoch < num_epochs; ++epoch) {
    if (c.get_terminate_training()) { break; }

    // Initialize epoch
    model.reset_mode(c, execution_mode::training);
    model.reset_epoch_statistics(execution_mode::training);
    dc.reset_mode(c);
    do_epoch_begin_cbs(model);

    // Training iterations
    if (num_batches > 0) {
      for (size_t i = 0; i < num_batches; i++) { train_mini_batch(c, model, dc); }
    } else {
      while (!train_mini_batch(c, model, dc)) {}
    }

    // Finalize epoch
    c.inc_epoch();
    model.reconcile_weight_values();
    do_epoch_end_cbs(model);

    // Evaluate on validation set
    auto key = c.get_trainer().check_and_build_execution_context(c, model, execution_mode::validation);
    auto& evaluation_context = static_cast<sgd_execution_context&>(c.get_trainer().get_execution_context(key));
    // Check to make sure that the model has a valid execution mode
    // before trying to do inference
    if (dc.is_execution_mode_valid(execution_mode::validation)) {
      evaluate(evaluation_context, model, dc, execution_mode::validation);
    }
  }
  do_train_end_cbs(model);
}

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

bool sgd_training_algorithm::train_mini_batch(sgd_execution_context& c,
                                              model& model,
                                              data_coordinator& dc) {
  model.reset_mode(c, execution_mode::training);
  dc.reset_mode(c);
  do_batch_begin_cbs(model, execution_mode::training);

  bool finished;

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
  model.get_objective_function()->start_evaluation(execution_mode::training,
                                                    c.get_current_mini_batch_size());

  // Backward prop step
  model.get_objective_function()->differentiate();
  model.backward_prop();
  model.get_objective_function()->compute_weight_regularization();

  // Finish evaluation.
  model.get_objective_function()->finish_evaluation(execution_mode::training,
                                                     c.get_current_mini_batch_size());
  model.evaluate_metrics(execution_mode::training,
                          c.get_current_mini_batch_size());

  // Update step
  model.update_weights();
  /*finished = */model.update_layers();
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
                                      size_t num_batches) {
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
  if (!dc.is_execution_mode_valid(mode)) return;
  if (mode != execution_mode::validation
      && mode != execution_mode::testing) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid execution mode for evaluation";
    throw lbann_exception(err.str());
  }

  // Evaluate on all mini-batches
  do_evaluate_begin_cbs(model, mode);
  if (num_batches > 0) {
    for (size_t i = 0; i < num_batches; i++) { evaluate_mini_batch(c, model, dc, mode); }
  } else {
    while (!evaluate_mini_batch(c, model, dc, mode)) {}
  }
  c.inc_epoch();
  do_evaluate_end_cbs(model, mode);
}

bool sgd_training_algorithm::evaluate_mini_batch(sgd_execution_context& c,
                                                 model& model,
                                                 data_coordinator& dc,
                                                 execution_mode mode) {
  model.reset_mode(c, mode);
  dc.reset_mode(c);
  do_batch_begin_cbs(model, mode);
  dc.fetch_data(mode);
  model.forward_prop(mode);
  // check if the data coordinator has finished the epoch and kickoff
  // background I/O
  const bool finished = dc.epoch_complete(mode);

  model.get_objective_function()->start_evaluation(mode, c.get_current_mini_batch_size());
  model.get_objective_function()->finish_evaluation(mode, c.get_current_mini_batch_size());
  model.evaluate_metrics(mode, c.get_current_mini_batch_size());
  model.update_layers();
  c.inc_step();
  do_batch_end_cbs(model, mode);
  return finished;
}

////////////////////////////////////////////////////////////
// Callbacks
////////////////////////////////////////////////////////////

void sgd_training_algorithm::do_train_begin_cbs(model& model) {
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_begin(&model);
  }
}

void sgd_training_algorithm::do_train_end_cbs(model& model) {
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_end(&model);
  }
}

void sgd_training_algorithm::do_evaluate_begin_cbs(model& model, execution_mode mode) {
  for (const auto& cb : model.get_callbacks()) {
    switch (mode) {
    case execution_mode::validation:
      cb->on_validation_begin(&model); break;
    case execution_mode::testing:
      cb->on_test_begin(&model); break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void sgd_training_algorithm::do_evaluate_end_cbs(model& model, execution_mode mode) {
  for (const auto& cb : model.get_callbacks()) {
    switch (mode) {
    case execution_mode::validation:
      cb->on_validation_end(&model); break;
    case execution_mode::testing:
      cb->on_test_end(&model); break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void sgd_training_algorithm::do_epoch_begin_cbs(model& model) {
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_begin(&model);
  }
}

void sgd_training_algorithm::do_epoch_end_cbs(model& model) {
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_end(&model);
  }
}

void sgd_training_algorithm::do_batch_begin_cbs(model& model, execution_mode mode) {
  sgd_execution_context& c = static_cast<sgd_execution_context&>(model.get_execution_context());

  for (const auto& cb : model.get_callbacks()) {
    switch (mode) {
    case execution_mode::training:
      if (c.get_step() % cb->get_batch_interval() == 0) {
        cb->on_batch_begin(&model);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_batch_evaluate_begin(&model);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void sgd_training_algorithm::do_batch_end_cbs(model& model, execution_mode mode) {
  sgd_execution_context& c = static_cast<sgd_execution_context&>(model.get_execution_context());

  for (const auto& cb : model.get_callbacks()) {
    switch (mode) {
    case execution_mode::training:
      if (c.get_step() % cb->get_batch_interval() == 0) {
        cb->on_batch_end(&model);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_batch_evaluate_end(&model);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

}  // namespace lbann
