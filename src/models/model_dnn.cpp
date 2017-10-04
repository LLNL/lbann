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
//
// lbann_model_dnn .hpp .cpp - Deep Neural Networks models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/model_dnn.hpp"

#include <string>
#include <chrono>
#include <random>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include "mpi.h"

namespace lbann {

////////////////////////////////////////////////////////////////////////////////
// deep_neural_network : main deep neural network class
////////////////////////////////////////////////////////////////////////////////

deep_neural_network::deep_neural_network(int mini_batch_size,
                                         lbann_comm *comm,
                                         objective_functions::objective_function *obj_fn,
                                         optimizer_factory *_optimizer_fac)
  : sequential_model(mini_batch_size, comm, obj_fn, _optimizer_fac) {}

deep_neural_network::~deep_neural_network() {}

void deep_neural_network::summarize_stats(lbann_summary& summarizer) {
  for (size_t l = 0; l < m_layers.size(); ++l) {
    m_layers[l]->summarize_stats(summarizer, get_cur_step());
  }
}

void deep_neural_network::summarize_matrices(lbann_summary& summarizer) {
  for (size_t l = 0; l < m_layers.size(); ++l) {
    m_layers[l]->summarize_matrices(summarizer, get_cur_step());
  }
}

void deep_neural_network::train(int num_epochs) {
  do_train_begin_cbs();

  // Epoch main loop
  for (int epoch = 0; epoch < num_epochs; ++epoch) {

    // Check if training has been terminated
    if (get_terminate_training()) {
      break;
    }

    // due to restart, may not always be at start of epoch
    // use mini batch index in data reader to signify start of epoch
    if (at_epoch_start()) {
      ++m_current_epoch;
      do_epoch_begin_cbs();
    }

    // Set the execution mode to training
    m_execution_mode = execution_mode::training;
    for (size_t l = 0; l < m_layers.size(); ++l) {
      m_layers[l]->set_execution_mode(execution_mode::training);
    }

    // Train on mini-batches until data set is traversed
    // Note: The data reader shuffles the data after each epoch
    m_obj_fn->reset_statistics();
    for (auto&& m : m_metrics) {
      m->reset_metric();
    }
    bool finished_epoch = false;
    while (!finished_epoch) {
      finished_epoch = train_mini_batch();

      // save a checkpoint if needed
      if (need_checkpoint()) {
        checkpointShared();
      }
    }

    // Evaluate model on validation set
    // TODO: do we need validation callbacks here?
    // do_validation_begin_cbs();
    evaluate(execution_mode::validation);
    // do_validation_end_cbs();

    do_epoch_end_cbs();

    // save checkpoint after epoch
    if (need_checkpoint()) {
      checkpointShared();
    }
  }

  do_train_end_cbs();
}

bool deep_neural_network::train_mini_batch() {
  do_batch_begin_cbs();

  // Forward propagation
  do_model_forward_prop_begin_cbs();
  for (size_t l = 0u; l < m_layers.size(); ++l) {
    do_layer_forward_prop_begin_cbs(m_layers[l]);
    m_layers[l]->forward_prop();
    do_layer_forward_prop_end_cbs(m_layers[l]);
  }
  do_model_forward_prop_end_cbs();

  // Record and reset objective function value
  m_obj_fn->record_and_reset_value();

  // Backward propagation
  do_model_backward_prop_begin_cbs();
  for (size_t l = m_layers.size(); l-- > 0u;) {
    do_layer_backward_prop_begin_cbs(m_layers[l]);
    m_layers[l]->back_prop();
    do_layer_backward_prop_end_cbs(m_layers[l]);
  }
  do_model_backward_prop_end_cbs();

  /// Update layers
  for (size_t l = m_layers.size() - 1; l > 0u; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();

  do_batch_end_cbs();
  ++m_current_step; // Update the current step once the entire mini-batch is complete
  return data_set_processed;
}

void deep_neural_network::evaluate(execution_mode mode) {
  if (!is_execution_mode_valid(mode)) { return; }
  switch(mode) {
  case execution_mode::validation:
    do_validation_begin_cbs();
    break;
  case execution_mode::testing:
    do_test_begin_cbs();
    break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate function");
  }

  // Set the execution mode
  m_execution_mode = mode;
  for (size_t l = 0u; l < m_layers.size(); ++l) {
    m_layers[l]->set_execution_mode(mode);
  }

  // Evaluate on mini-batches until data set is traversed
  // Note: The data reader shuffles the data after each epoch
  m_obj_fn->reset_statistics();
  for (auto&& m : m_metrics) {
    m->reset_metric();
  }
  bool finished_epoch = false;
  while (!finished_epoch) {
    finished_epoch = evaluate_mini_batch();
  }

  switch(mode) {
  case execution_mode::validation:
    do_validation_end_cbs();
    break;
  case execution_mode::testing:
    do_test_end_cbs();
    break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate function");
  }

  return;
}

bool deep_neural_network::evaluate_mini_batch() {
  do_batch_evaluate_begin_cbs();

  // forward propagation (mini-batch)
  do_model_evaluate_forward_prop_begin_cbs();
  for (size_t l = 0u; l < m_layers.size(); l++) {
    do_layer_evaluate_forward_prop_begin_cbs(m_layers[l]);
    m_layers[l]->forward_prop();
    do_layer_evaluate_forward_prop_end_cbs(m_layers[l]);
  }
  do_model_evaluate_forward_prop_end_cbs();

  // Record and reset objective function value
  m_obj_fn->record_and_reset_value();

  // Update layers
  // Note: should only affect the input and target layers
  for (size_t l = m_layers.size() - 1; l > 0u; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();
  do_batch_evaluate_end_cbs();
  switch(m_execution_mode) {
  case execution_mode::validation:
    ++m_current_validation_step;
    break;
  case execution_mode::testing:
    ++m_current_testing_step;
    break;
  default:
    throw lbann_exception("Illegal execution mode in evaluate mini-batch function");
  }
  return data_set_processed;
}

}  // namespace lbann
