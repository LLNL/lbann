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
// //#include "lbann/callbacks/callback_save_training_algorithm.hpp"
// #include "lbann/io/persist.hpp"
// #include "lbann/layers/io/input/generic_input_layer.hpp"
// #include "lbann/layers/transform/dummy.hpp"
// #include "lbann/layers/transform/split.hpp"
// #include "lbann/layers/transform/evaluation.hpp"
// #include "lbann/objective_functions/layer_term.hpp"
// #include "lbann/metrics/layer_metric.hpp"
// #include "lbann/utils/random.hpp"
// #include "lbann/utils/omp_diagnostics.hpp"
// #include "lbann/utils/description.hpp"
// #include <string>
// #include <unistd.h>
// #include <iomanip>
// #include <queue>
// #include <unordered_set>
// #include <lbann.pb.h>

// #include "mpi.h"

namespace lbann {

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

void sgd_training_algorithm::apply(execution_context& context,
                                   model& model,
                                   execution_mode mode,
                                   termination_criteria const& term_criteria) {
  sgd_execution_context& sgd_context = static_cast<sgd_execution_context&>(context);
  const sgd_termination_criteria& sgd_term = static_cast<const sgd_termination_criteria&>(term_criteria);
  switch(mode) {
  case execution_mode::training:
    train(sgd_context, model, sgd_term.num_epochs, sgd_term.num_steps);
    break;
  case execution_mode::validation:
  case execution_mode::testing:
  case execution_mode::prediction:
    evaluate(sgd_context, model, mode, sgd_term.num_steps);
    break;
  default:
    LBANN_ERROR(std::string{} + "Illegal mode: " + _to_string(mode));
  }
}

void sgd_training_algorithm::train(sgd_execution_context& c,
                                   model& model,
                                   El::Int num_epochs,
                                   El::Int num_batches) {
  do_train_begin_cbs(model);
  for (int epoch = c.get_epoch(); epoch < num_epochs; ++epoch) {
    if (c.get_terminate_training()) { break; }

    // Initialize epoch
    model.reset_mode(execution_mode::training);
    do_epoch_begin_cbs(model);

    // Training iterations
    if (num_batches > 0) {
      for (int i = 0; i < num_batches; i++) { train_mini_batch(c, model); }
    } else {
      while (!train_mini_batch(c, model)) {}
    }

    // Finalize epoch
    c.inc_epoch();
    model.reconcile_weight_values();
    do_epoch_end_cbs(model);
    model.reset_epoch_statistics(execution_mode::training);

    // Evaluate on validation set
    auto evaluation_context = make_unique<sgd_execution_context>(static_cast<const sgd_execution_context&>(c));
    evaluation_context.get()->set_execution_mode(execution_mode::validation);
    evaluate(*(evaluation_context.get()), model, execution_mode::validation);

  }
  do_train_end_cbs(model);
}

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

bool sgd_training_algorithm::train_mini_batch(sgd_execution_context& c,
                                              model& model) {
  model.reset_mode(execution_mode::training);
  do_batch_begin_cbs(model, execution_mode::training);

  bool finished;

#if defined(LBANN_HAVE_OMP_TASKLOOP)
  LBANN_OMP_PARALLEL
  {
    #pragma omp single
    {
#endif
  // Forward prop step
  model.clear_gradients();
  model.forward_prop(execution_mode::training);
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
  finished = model.update_layers();
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
                                      execution_mode mode,
                                      El::Int num_batches) {
  // Return early if execution mode is invalid
  if (!model.is_execution_mode_valid(mode)) return;
  if (mode != execution_mode::validation
      && mode != execution_mode::testing) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid execution mode for evaluation";
    throw lbann_exception(err.str());
  }

  // Evaluate on all mini-batches
  model.reset_epoch_statistics(mode);
  model.reset_mode(mode);
  do_evaluate_begin_cbs(model, mode);
  if (num_batches > 0) {
    for (int i = 0; i < num_batches; i++) { evaluate_mini_batch(c, model, mode); }
  } else {
    while (!evaluate_mini_batch(c, model, mode)) {}
  }
  do_evaluate_end_cbs(model, mode);
}

bool sgd_training_algorithm::evaluate_mini_batch(sgd_execution_context& c,
                                                 model& model,
                                                 execution_mode mode) {
  model.reset_mode(mode);
  do_batch_begin_cbs(model, mode);
  model.forward_prop(mode);
  model.get_objective_function()->start_evaluation(mode, c.get_current_mini_batch_size());
  model.get_objective_function()->finish_evaluation(mode, c.get_current_mini_batch_size());
  model.evaluate_metrics(mode, c.get_current_mini_batch_size());
  const bool finished = model.update_layers();
  c.inc_step();
  do_batch_end_cbs(model, mode);
  return finished;
}

#if 0
//this is for data store functionality
void sgd_training_algorithm::collect_indices(execution_mode mode) {
  reset_mode(mode);
  while (true) {
    m_layers[0]->forward_prop();
    bool finished = true;
    finished = m_layers[0]->update() && finished;
    if (finished) {
      break;
    }
  }
  //this may not be necessary, but shouldn't hurt
  reset_epoch_statistics(mode);
}
#endif


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
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
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
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
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
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
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
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
    }
  }
}

#if 0
////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_sgd_training_algorithm_header {
  uint32_t execution_mode;
  uint32_t terminate_training;
  uint64_t current_epoch;
  uint64_t current_step;
  uint64_t current_validation_step;
  uint64_t current_testing_step;
  uint32_t max_mini_batch_size;
  uint32_t current_mini_batch_size;
  uint32_t current_phase;
  uint32_t callback_type;;
};

bool trainer::save_to_checkpoint_shared(persist& p) {
  // write out fields we need to save for trainer
  if (p.get_cb_type() != callback_type::validation) {
    if (m_comm->am_trainer_master()) {
      p.write_uint32(persist_type::train, "execution_mode",     (uint32_t) m_execution_mode);
      p.write_uint32(persist_type::train, "terminate_training", (uint32_t) m_terminate_training);
      p.write_uint64(persist_type::train, "epoch",              (uint64_t) m_epoch);
      p.write_uint64(persist_type::train, "training_step",      (uint64_t) get_step(execution_mode::training));
      p.write_uint64(persist_type::train, "testing_step",       (uint64_t) get_step(execution_mode::testing));
      p.write_uint32(persist_type::train, "max_mini_batch_size",      (uint32_t) m_max_mini_batch_size);
      p.write_uint32(persist_type::train, "current_mini_batch_size",      (uint32_t) m_current_mini_batch_size);
      p.write_uint32(persist_type::train, "current_phase",      (uint32_t) m_current_phase);
      p.write_uint32(persist_type::train, "persist_callback_type",      (uint32_t) p.get_cb_type());
      /// @todo BVE FIXME
      if(p.get_cb_type() == callback_type::batch) {
        p.write_uint64(persist_type::validate, "validation_step",       (uint64_t) get_step(execution_mode::validation));
      }
    }

  }
  else{
    if (m_comm->am_trainer_master()) {
      p.write_uint64(persist_type::validate, "validation_step",       (uint64_t) get_step(execution_mode::validation));
    }
    save_rng_to_checkpoint_shared(p, m_comm);
  }
  return true;
}

bool trainer::load_from_checkpoint_shared(persist& p) {
  // have rank 0 read the file
  // read state from file
  struct lbann_trainee_header header;
  // Assume checkpoint reload from epoch end not step end
  if (m_comm->am_trainer_master()) {
    if (p.get_cb_type() != callback_type::validation) {
      p.read_uint32(persist_type::train, "execution_mode",     &header.execution_mode);
      p.read_uint32(persist_type::train, "terminate_training", &header.terminate_training);
      p.read_uint64(persist_type::train, "current_epoch",      &header.current_epoch);
      p.read_uint64(persist_type::train, "current_step",       &header.current_step);
      if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
        p.read_uint64(persist_type::validate, "current_validation_step",       &header.current_validation_step);
      p.read_uint64(persist_type::train, "current_testing_step",       &header.current_testing_step);
      p.read_uint32(persist_type::train, "max_mini_batch_size",      &header.max_mini_batch_size);
      p.read_uint32(persist_type::train, "current_mini_batch_size",      &header.current_mini_batch_size);
      p.read_uint32(persist_type::train, "current_phase",      &header.current_phase);
      p.read_uint32(persist_type::train, "persist_callback_type",     &header.callback_type);
    } else {
      p.read_uint64(persist_type::validate, "current_validation_step",       &header.current_validation_step);
    }
  }
  load_rng_from_checkpoint_shared(p, m_comm);
  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  m_comm->trainer_broadcast(0, header);
  // set our member params from values read from disk
  if (p.get_cb_type() != callback_type::validation) {
    m_execution_mode     = (execution_mode) header.execution_mode;
    m_terminate_training = (bool)           header.terminate_training;
    m_current_epoch      = (int)            header.current_epoch;
    m_current_step       = (int)            header.current_step;
    if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
      m_current_validation_step = (int)       header.current_validation_step;
    m_current_testing_step = (int)          header.current_testing_step;
    m_max_mini_batch_size = (int)           header.max_mini_batch_size;
    m_current_mini_batch_size = (int)       header.current_mini_batch_size;
    m_current_phase      =                  header.current_phase;
    // set state of persist object to know which type of ckpt we are returning from.
    p.set_cb_type((callback_type) header.callback_type);
  } else {
    m_current_validation_step = (int)       header.current_validation_step;
  }

  for (weights *w : m_weights) {
    w->load_from_checkpoint_shared(p);
  }

  // read in each layer
  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->load_from_checkpoint_shared(p)) {
      return false;
    }
  }
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0){
    for (const auto& m : m_metrics) {
      m->load_from_checkpoint_shared(p);
    }
  }
#ifdef LBANN_HAS_GPU
  El::GPUManager::SynchronizeDevice();
#endif // LBANN_HAS_GPU
  return true;
}

bool trainer::save_to_checkpoint_distributed(persist& p){
  // write out fields we need to save for trainer
  if (p.get_cb_type() != callback_type::validation) {
    p.write_uint32(persist_type::train, "execution_mode",     (uint32_t) m_execution_mode);
    p.write_uint32(persist_type::train, "terminate_training", (uint32_t) m_terminate_training);
    p.write_uint64(persist_type::train, "current_epoch",      (uint64_t) m_current_epoch);
    p.write_uint64(persist_type::train, "current_step",       (uint64_t) m_current_step);
    p.write_uint64(persist_type::train, "current_testing_step",       (uint64_t) m_current_testing_step);
    p.write_uint32(persist_type::train, "max_mini_batch_size",      (uint32_t) m_max_mini_batch_size);
    p.write_uint32(persist_type::train, "current_mini_batch_size",      (uint32_t) m_current_mini_batch_size);
    p.write_uint32(persist_type::train, "current_phase",      (uint32_t) m_current_phase);
    p.write_uint32(persist_type::train, "persist_callback_type",      (uint32_t) p.get_cb_type());
    if(p.get_cb_type() == callback_type::batch)
      p.write_uint64(persist_type::validate, "current_validataion_step",       (uint64_t) m_current_validation_step);

    for (weights *w : m_weights) {
      w->save_to_checkpoint_distributed(p);
    }

    for (size_t l = 0; l < m_layers.size(); l++) {
      if (! m_layers[l]->save_to_checkpoint_distributed(p)) {
        return false;
      }
    }
    if(p.get_cb_type() == callback_type::batch || get_num_iterations_per_epoch(execution_mode::validation) == 0){
       save_rng_to_checkpoint_shared(p, m_comm);
      for (const auto& m : m_metrics) {
        m->save_to_checkpoint_distributed(p);
      }
    }
  }

  else {
    p.write_uint64(persist_type::validate, "current_validataion_step",       (uint64_t) m_current_validation_step);
    save_rng_to_checkpoint_shared(p, m_comm);

    for (size_t l = 0; l < m_layers.size(); l++) {
      if (! m_layers[l]->save_to_checkpoint_distributed(p)) {
        return false;
      }
    }
    for (const auto& m : m_metrics) {
      m->save_to_checkpoint_distributed(p);
    }
  }
  return true;
}

bool trainer::load_from_checkpoint_distributed(persist& p){
  struct lbann_trainer_header header;
  p.read_uint32(persist_type::train, "execution_mode",     &header.execution_mode);
  p.read_uint32(persist_type::train, "terminate_training", &header.terminate_training);
  p.read_uint64(persist_type::train, "current_epoch",      &header.current_epoch);
  p.read_uint64(persist_type::train, "current_step",       &header.current_step);
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
    p.read_uint64(persist_type::validate, "current_validation_step",       &header.current_validation_step);
  p.read_uint64(persist_type::train, "current_testing_step",       &header.current_testing_step);
  p.read_uint32(persist_type::train, "max_mini_batch_size",      &header.max_mini_batch_size);
  p.read_uint32(persist_type::train, "current_mini_batch_size",      &header.current_mini_batch_size);
  p.read_uint32(persist_type::train, "current_phase",      &header.current_phase);
  p.read_uint32(persist_type::train, "persist_callback_type",     &header.callback_type);

  m_execution_mode     = (execution_mode) header.execution_mode;
  m_terminate_training = (bool)           header.terminate_training;
  m_current_epoch      = (int)            header.current_epoch;
  m_current_step       = (int)            header.current_step;
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
    m_current_validation_step = (int)       header.current_validation_step;
  m_current_testing_step = (int)          header.current_testing_step;
  m_max_mini_batch_size = (int)           header.max_mini_batch_size;
  m_current_mini_batch_size = (int)       header.current_mini_batch_size;
  m_current_phase      =                  header.current_phase;

  p.set_cb_type((callback_type) header.callback_type);
  load_rng_from_checkpoint_shared(p, m_comm);

  for (weights *w : m_weights) {
    w->load_from_checkpoint_distributed(p);
  }

  for (size_t l = 0; l < m_layers.size(); l++) {
    if (! m_layers[l]->load_from_checkpoint_distributed(p)) {
      return false;
    }
  }
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0){
    for (const auto& m : m_metrics) {
      m->load_from_checkpoint_distributed(p);
    }
  }
  return true;
}

void trainer::write_proto(lbann_data::Trainer* proto) {
  proto->Clear();
  if (m_comm->am_world_master())
    proto->set_mini_batch_size(m_max_mini_batch_size);
}
#endif
}  // namespace lbann
