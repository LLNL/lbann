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
// model_dag .hpp .cpp - Directed acyclic graph neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/model_dag.hpp"
#include "lbann/layers/io/input/input_layer.hpp"

#include <iomanip>
#include <vector>
#include <stack>
#include <unordered_map>

namespace lbann {

dag_model::dag_model(int mini_batch_size,
                     lbann_comm *comm,
                     objective_functions::objective_function *obj_fn,
                     optimizer_factory *optimizer_fac)
  : model(comm, mini_batch_size, obj_fn, optimizer_fac) {}

dag_model::dag_model(const dag_model& other) :
  model(other) {

  // Copy layers from the other model's layer list
  std::unordered_map<const Layer*,const Layer*> old_to_new_layer;
  for (const Layer* old_layer : other.m_layers) {
    Layer* new_layer = old_layer->copy();
    old_to_new_layer[old_layer] = new_layer;
    m_layers.push_back(new_layer);
  }

  // Fix layer pointers
  for (Layer* layer : m_layers) {
    for (const Layer*& parent : layer->get_parent_layers()) {
      const Layer* new_parent = old_to_new_layer[parent];
      if (new_parent != nullptr) {
        parent = new_parent;
      }
    }
    for (const Layer*& child : layer->get_child_layers()) {
      const Layer* new_child = old_to_new_layer[child];
      if (new_child != nullptr) {
        child = new_child;
      }
    }
  }

}

dag_model& dag_model::operator=(const dag_model& other) {
  model::operator=(other);

  // Clear list of layers
  for (Layer* layer : m_layers) {
    delete layer;
  }
  m_layers.clear();

  // Copy layers from the other model's layer list
  std::unordered_map<const Layer*,const Layer*> old_to_new_layer;
  for (const Layer* old_layer : other.m_layers) {
    Layer* new_layer = old_layer->copy();
    old_to_new_layer[old_layer] = new_layer;
    m_layers.push_back(new_layer);
  }

  // Fix layer pointers
  for (Layer* layer : m_layers) {
    for (const Layer*& parent : layer->get_parent_layers()) {
      const Layer* new_parent = old_to_new_layer[parent];
      if (new_parent != nullptr) {
        parent = new_parent;
      }
    }
    for (const Layer*& child : layer->get_child_layers()) {
      const Layer* new_child = old_to_new_layer[child];
      if (new_child != nullptr) {
        child = new_child;
      }
    }
  }

  return *this;
}

dag_model::~dag_model() {
  for (Layer* layer : m_layers) {
    delete layer;
  }
}

int dag_model::add(Layer *new_layer) {
  m_layers.push_back(new_layer);
  return 0;
}

void dag_model::setup() {

  // Sort layers topologically
  topologically_sort_layers();

  // Setup each layer
  for (Layer* layer : m_layers) {
    layer->set_neural_network_model(this);
    layer->setup();
    layer->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << "[" << std::setw(18) << layer->get_name() <<  "] Set up a layer with input " << std::setw(7) << layer->get_num_prev_neurons() << " and " << std::setw(7) << layer->get_num_neurons() << " neurons."  << std::endl;
    }
  }

  // Set up callbacks
  setup_callbacks();
}

void dag_model::topologically_sort_layers() {
  /* Note: This sort must be deterministic so that it produces
   * identical orderings when applied on different MPI processes.
   */

  // Initialize data structures for topological sort
  std::stack<const Layer*> sorted_stack;
  std::stack<const Layer*> search_stack;
  std::unordered_map<const Layer*,bool> is_sorted;
  std::unordered_map<const Layer*,bool> is_visited;
  for (const Layer* layer : m_layers) {
    is_sorted[layer] = false;
    is_visited[layer] = false;
  }

  // Iterate through layers that have not already been sorted
  for (const Layer* source_layer : m_layers) {
    if (is_sorted[source_layer]) {
      continue;
    }

    // Perform depth-first search starting from source layer
    search_stack.push(source_layer);
    while(!search_stack.empty()) {
      const Layer* search_layer = search_stack.top();

      if (is_visited[search_layer]) {
        // Move search layer to sorted stack if we have visited already
        search_stack.pop();
        sorted_stack.push(search_layer);
        is_sorted[search_layer] = true;
      }
      else {
        // Visit search layer by adding children to search stack
        is_visited[search_layer] = true;
        for (const Layer* child_layer : search_layer->get_child_layers()) {
          if (!is_sorted[child_layer]) {
            if (is_visited[child_layer]) {
              throw lbann_exception("model_dag: detected a cycle in network graph");
            }
            search_stack.push(child_layer);
          }
        }
      }

    }

  }

  // Record topologically sorted ordering
  m_layers.clear();
  while (!sorted_stack.empty()) {
    m_layers.push_back(const_cast<Layer*>(sorted_stack.top()));
    sorted_stack.pop();
  }
  
}

void dag_model::summarize_stats(lbann_summary& summarizer) {
  for (Layer* layer : m_layers) {
    layer->summarize_stats(summarizer, get_cur_step());
  }
}

void dag_model::summarize_matrices(lbann_summary& summarizer) {
  for (Layer* layer : m_layers) {
    layer->summarize_matrices(summarizer, get_cur_step());
  }
}

void dag_model::train(int num_epochs) {
  do_train_begin_cbs();

  // Epoch main loop
  for (int epoch = 0; epoch < num_epochs; ++epoch) {

    // Check if training has been terminated
    if (get_terminate_training()) {
      break;
    }

    // Check if we are at the start of an epoch
    for (Layer* layer : m_layers) {
      input_layer* input = dynamic_cast<input_layer*>(layer);
      if (input != nullptr && input->at_new_epoch()) {
        ++m_current_epoch;
        do_epoch_begin_cbs();
        break;
      }
    }

    // Set the execution mode to training
    m_execution_mode = execution_mode::training;
    for (Layer* layer : m_layers) {
      layer->set_execution_mode(execution_mode::training);
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

      // Save a checkpoint if needed
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

bool dag_model::train_mini_batch() {
  do_batch_begin_cbs();

  // Forward propagation
  do_model_forward_prop_begin_cbs();
  for (Layer* layer : m_layers) {
    do_layer_forward_prop_begin_cbs(layer);
    layer->forward_prop();
    do_layer_forward_prop_end_cbs(layer);
  }
  do_model_forward_prop_end_cbs();

  // Record and reset objective function value
  m_obj_fn->record_and_reset_value();

  // Backward propagation
  do_model_backward_prop_begin_cbs();
  for (int l = m_layers.size() - 1; l >= 0; --l) {
    Layer* layer = m_layers[l];
    do_layer_backward_prop_begin_cbs(layer);
    layer->back_prop();
    do_layer_backward_prop_end_cbs(layer);
  }
  do_model_backward_prop_end_cbs();

  // Update layers
  for (int l = m_layers.size() - 1; l > 0; --l) {
    m_layers[l]->update();
  }
  const bool data_set_processed = m_layers[0]->update();

  do_batch_end_cbs();
  ++m_current_step; // Update the current step once the entire mini-batch is complete
  return data_set_processed;
}

void dag_model::evaluate(execution_mode mode) {

  // Return early if execution mode is invalid
  for (Layer* layer : m_layers) {
    input_layer* input = dynamic_cast<input_layer*>(layer);
    if (input != nullptr && !input->is_execution_mode_valid(mode)) {
      return;
    }
  }

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
  for (Layer* layer : m_layers) {
    layer->set_execution_mode(mode);
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

bool dag_model::evaluate_mini_batch() {
  do_batch_evaluate_begin_cbs();

  // forward propagation (mini-batch)
  do_model_evaluate_forward_prop_begin_cbs();
  for (Layer* layer : m_layers) {
    do_layer_evaluate_forward_prop_begin_cbs(layer);
    layer->forward_prop();
    do_layer_evaluate_forward_prop_end_cbs(layer);
  }
  do_model_evaluate_forward_prop_end_cbs();

  // Record and reset objective function value
  m_obj_fn->record_and_reset_value();

  // Update layers
  // Note: should only affect the input and target layers
  for (int l = m_layers.size() - 1; l > 0; --l) {
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

bool dag_model::at_epoch_start() { 
    // use mini batch index in data reader to signify start of epoch
    io_layer *input = (io_layer *) m_layers[0];
    bool flag = input->at_new_epoch();
    return flag;
}

}  // namespace lbann
