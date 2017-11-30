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
// lbann_model .hpp .cpp - Abstract class for neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/model.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/layers/io/input/input_layer.hpp"
#include <string>
#include <unistd.h>
#include <iomanip>
#include <queue>
#include <unordered_set>

#include "mpi.h"

namespace lbann {

////////////////////////////////////////////////////////////
// Constructors and destructor
////////////////////////////////////////////////////////////

model::model(lbann_comm *comm,
             int mini_batch_size,
             objective_function *obj_fn,
             optimizer* default_optimizer)
  : m_objective_function(obj_fn),
    m_execution_mode(execution_mode::training),
    m_terminate_training(false),
    m_current_epoch(0),
    m_current_step(0),
    m_current_validation_step(0),
    m_current_testing_step(0),
    m_max_mini_batch_size(mini_batch_size),
    m_current_mini_batch_size(mini_batch_size),
    m_effective_mini_batch_size(mini_batch_size),
    m_current_phase(0),
    m_comm(comm),
    m_checkpoint_dir(""),
    m_checkpoint_epochs(0),
    m_checkpoint_steps(0),
    m_checkpoint_secs(0.0),
    m_checkpoint_last(get_time()),
    m_default_optimizer(default_optimizer) {}

model::model(const model& other) :
  m_execution_mode(other.m_execution_mode),
  m_terminate_training(other.m_terminate_training),
  m_current_epoch(other.m_current_epoch),
  m_current_step(other.m_current_step),
  m_current_validation_step(other.m_current_validation_step),
  m_current_testing_step(other.m_current_testing_step),
  m_max_mini_batch_size(other.m_max_mini_batch_size),
  m_current_mini_batch_size(other.m_current_mini_batch_size),
  m_effective_mini_batch_size(other.m_effective_mini_batch_size),
  m_current_phase(other.m_current_phase),
  m_comm(other.m_comm),
  m_checkpoint_dir(other.m_checkpoint_dir),
  m_checkpoint_epochs(other.m_checkpoint_epochs),
  m_checkpoint_steps(other.m_checkpoint_steps),
  m_checkpoint_secs(other.m_checkpoint_secs),
  m_checkpoint_last(other.m_checkpoint_last) {

  // Deep copies
  m_objective_function = other.m_objective_function;
  m_metrics            = other.m_metrics;
  m_callbacks          = other.m_callbacks;
  m_layers             = other.m_layers;
  m_weights            = other.m_weights;
  if (m_objective_function != nullptr) {
    m_objective_function = m_objective_function->copy();
  }
  for (auto& m : m_metrics) {
    m = m->copy();
  }
  for (auto& cb : m_callbacks) {
    cb = cb->copy();
  }
  std::unordered_map<Layer *,Layer *> layer_map;
  for (auto& l : m_layers) {
    l = layer_map[l] = l->copy();
    l->set_model(this);
  }
  std::unordered_map<weights *,weights *> weights_map;
  for (auto& w : m_weights) {
    w = weights_map[w] = w->copy();
  }
  remap_pointers(layer_map, weights_map);

}

model& model::operator=(const model& other) {

  // Delete objects
  if (m_objective_function != nullptr) { delete m_objective_function; }
  for (const auto& m : m_metrics)      { delete m; }
  for (const auto& cb : m_callbacks)   { delete cb; }
  for (const auto& l : m_layers)       { delete l; }
  for (const auto& w : m_weights)      { delete w; }

  // Shallow copies
  m_execution_mode = other.m_execution_mode;
  m_terminate_training = other.m_terminate_training;
  m_current_epoch = other.m_current_epoch;
  m_current_step = other.m_current_step;
  m_current_validation_step = other.m_current_validation_step;
  m_current_testing_step = other.m_current_testing_step;
  m_max_mini_batch_size = other.m_max_mini_batch_size;
  m_current_mini_batch_size = other.m_current_mini_batch_size;
  m_effective_mini_batch_size = other.m_effective_mini_batch_size;
  m_current_phase = other.m_current_phase;
  m_comm = other.m_comm;
  m_checkpoint_dir = other.m_checkpoint_dir;
  m_checkpoint_epochs = other.m_checkpoint_epochs;
  m_checkpoint_steps = other.m_checkpoint_steps;
  m_checkpoint_secs = other.m_checkpoint_secs;
  m_checkpoint_last = other.m_checkpoint_last;

  // Deep copies
  m_objective_function = other.m_objective_function;
  m_metrics            = other.m_metrics;
  m_callbacks          = other.m_callbacks;
  m_layers             = other.m_layers;
  m_weights            = other.m_weights;
  if (m_objective_function != nullptr) {
    m_objective_function = m_objective_function->copy();
  }
  for (auto& m : m_metrics) {
    m = m->copy();
  }
  for (auto& cb : m_callbacks) {
    cb = cb->copy();
  }
  std::unordered_map<Layer *,Layer *> layer_map;
  for (auto& l : m_layers) {
    l = layer_map[l] = l->copy();
    l->set_model(this);
  }
  std::unordered_map<weights *,weights *> weights_map;
  for (auto& w : m_weights) {
    w = weights_map[w] = w->copy();
  }
  remap_pointers(layer_map, weights_map);

  return *this;
}

model::~model() {
  if (m_objective_function)           { delete m_objective_function; }
  if (m_default_optimizer != nullptr) { delete m_default_optimizer; }
  for (const auto& l : m_layers)      { delete l; }
  for (const auto& w : m_weights)     { delete w; }
  for (const auto& m : m_metrics)     { delete m; }
  for (const auto& cb : m_callbacks)  { delete cb; }
}

////////////////////////////////////////////////////////////
// Model specification
////////////////////////////////////////////////////////////

void model::add_layer(Layer *l) {
  if (l == nullptr) {
    throw lbann_exception("model: Attempted to add null pointer as a layer.");
  }
  m_layers.push_back(l);
}

void model::add_weights(weights *w) {
  if (w == nullptr) {
    throw lbann_exception("model: Attempted to add null pointer as weights.");
  }
  m_weights.push_back(w);
}

void model::add_callback(lbann_callback *cb) {
  if (cb == nullptr) {
    throw lbann_exception("model: Attempted to add null pointer as a callback.");
  }
  m_callbacks.push_back(cb);
}

void model::add_metric(metric *m) {
  if (m == nullptr) {
    throw lbann_exception("model: Attempted to add null pointer as a metric.");
  }
  m_metrics.push_back(m);
}

void model::set_layers(std::vector<Layer*>& layers) {

  // Delete old layers
  for (const auto& layer : m_layers) {
    delete layer;
  }
  m_layers.clear();

  // Add new layers
  for (const auto& layer : layers) {
    add_layer(layer);
  }

}

void model::replace_weights(std::vector<weights*>& new_weights) {

  // Check that number of weights is valid
  if (new_weights.size() > m_weights.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to replace weights with an invalid number of weights "
        << "(expected at most " << m_weights.size() << ", found " << new_weights.size() << ")";
    throw lbann_exception(err.str());
  }
  
  // Replace weights in list
  std::vector<weights *> old_weights(m_weights.begin(),
                                     m_weights.begin() + new_weights.size());
  std::unordered_map<weights *,weights *> weights_map;
  std::unordered_map<Layer *,Layer *> layer_map;
  for (size_t i = 0; i < new_weights.size(); ++i) {
    m_weights[i] = weights_map[old_weights[i]] = new_weights[i];
  }
  remap_pointers(layer_map, weights_map);

  // Delete old weights
  for (const auto& w : old_weights) {
    delete w;
  }

}

optimizer* model::create_optimizer() const {
  if (m_default_optimizer != nullptr) {
    return m_default_optimizer->copy();
  } else {
    return nullptr;
  }
}

bool model::is_execution_mode_valid(execution_mode mode) const {
  for (const auto& layer : m_layers) {
    const auto *input = dynamic_cast<const input_layer*>(layer);
    if (input != nullptr
        && !input->is_execution_mode_valid(mode)) {
      return false;
    }
  }
  return true;
}

bool model::is_topologically_sorted() const {
  std::unordered_set<const Layer *> previous_layers;
  for (const auto& layer : m_layers) {
    for (const auto& parent : layer->get_parent_layers()) {
      if (previous_layers.count(parent) == 0) {
        return false;
      }
    }
    previous_layers.insert(layer);
  }
  return true;
}

std::string model::print_layer_description(const Layer* layer) const {
  if (layer == nullptr) return std::string();
  std::stringstream os;
  //std::string description = layer->get_description();
  os << std::setw(12) << layer->get_name() << ":[" << std::setw(18)
     << layer->get_type() <<  "] Set up a layer with input " << std::setw(7)
     << layer->get_num_prev_neurons() << " and " << std::setw(7)
     << layer->get_num_neurons() << " neurons.";
  std::string s = layer->get_topo_description();
  if(s != "") {
    os << " (" << s << ")";
  }
  return os.str();
}

void model::remap_pointers(const std::unordered_map<Layer *,Layer *>& layer_map,
                           const std::unordered_map<weights *,weights *>& weights_map) {

  // Fix pointers in objective function
  if (m_objective_function != nullptr) {
    auto layer_pointers = m_objective_function->get_layer_pointers();
    for (auto& layer_pointer : layer_pointers) {
      if (layer_map.count(layer_pointer) > 0) {
        layer_pointer = layer_map.at(layer_pointer);
      }
    }
    m_objective_function->set_layer_pointers(layer_pointers);
    auto weights_pointers = m_objective_function->get_weights_pointers();
    for (auto& weights_pointer : weights_pointers) {
      if (weights_map.count(weights_pointer) > 0) {
        weights_pointer = weights_map.at(weights_pointer);
      }
    }
    m_objective_function->set_weights_pointers(weights_pointers);
  }

  // Fix pointers in metrics
  for (const auto& m : m_metrics) {
    auto layer_pointers = m->get_layer_pointers();
    for (auto& layer_pointer : layer_pointers) {
      if (layer_map.count(layer_pointer) > 0) {
        layer_pointer = layer_map.at(layer_pointer);
      }
    }
    m->set_layer_pointers(layer_pointers);
  }

  // Fix pointers in layers
  for (const auto& l : m_layers) {
    auto layer_pointers = l->get_layer_pointers();
    for (auto& layer_pointer : layer_pointers) {
      if (layer_map.count(layer_pointer) > 0) {
        layer_pointer = layer_map.at(layer_pointer);
      }
    }
    l->set_layer_pointers(layer_pointers);
    auto weights_pointers = l->get_weights();
    for (auto& weights_pointer : weights_pointers) {
      if (weights_map.count(weights_pointer) > 0) {
        weights_pointer = weights_map.at(weights_pointer);
      }
    }
    l->set_weights(weights_pointers);
  }
  
}

////////////////////////////////////////////////////////////
// Setup
////////////////////////////////////////////////////////////

void model::setup() {

  // Setup layers
  setup_layer_topology();
  setup_layer_execution_order();
  setup_layers();

  // Setup weights
  setup_weights();

  // Setup objective function
  m_objective_function->setup(*this);

  // Setup metrics
  for (const auto& m : m_metrics) {
    m->setup(*this);
  }

  // Set up callbacks
  for (const auto& cb : m_callbacks) {
    cb->setup(this);
  }

}

void model::setup_layer_topology() {

  // Perform breadth-first searches to find connected components
  std::queue<const Layer *> layer_queue;
  std::unordered_set<const Layer *> layer_set;
  for (const auto& layer : m_layers) {
    layer_queue.push(layer);
    layer_set.insert(layer);
  }
  while (!layer_queue.empty()) {
    const Layer *layer = layer_queue.front();
    layer_queue.pop();
    std::vector<const Layer *> relatives;
    for (const auto& parent : layer->get_parent_layers()) {
      relatives.push_back(parent);
    }
    for (const auto& child : layer->get_child_layers()) {
      relatives.push_back(child);
    }
    for (const auto& relative : relatives) {
      if (layer_set.count(relative) == 0) {
        m_layers.push_back(const_cast<Layer *>(relative));
        layer_queue.push(relative);
        layer_set.insert(relative);
      }
    }
  }

  // Make sure parent and child relationships are reciprocated
  for (const auto& layer : m_layers) {
    for (const auto& parent : layer->get_parent_layers()) {
      const_cast<Layer *>(parent)->add_child_layer(layer);
    }
    for (const auto& child : layer->get_child_layers()) {
      const_cast<Layer *>(child)->add_parent_layer(layer);
    }
  }

}

void model::setup_layers() {
  for (const auto& layer : m_layers) {
    layer->set_model(this);
    layer->setup();
    layer->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << "[" << std::setw(18) << layer->get_type() <<  "] Set up a layer with input " << std::setw(7) << layer->get_num_prev_neurons() << " and " << std::setw(7) << layer->get_num_neurons() << " neurons."  << std::endl;
    }
  }
}

void model::setup_weights() {

  // List of used and unused weights
  std::unordered_set<weights *> weights_set(m_weights.begin(),
                                            m_weights.end());
  std::set<weights *> unused_weights(m_weights.begin(),
                                     m_weights.end());

  // Find weights used by layers
  for (const auto& layer : m_layers) {
    for (const auto& w : layer->get_weights()) {
      if (weights_set.count(w) == 0) {
        m_weights.push_back(w);
        weights_set.insert(w);
      }
      unused_weights.erase(w);
    }
  }

  // Find weights used by objective function
  for (const auto& w : m_objective_function->get_weights_pointers()) {
    if (weights_set.count(w) == 0) {
      m_weights.push_back(w);
      weights_set.insert(w);
    }
    unused_weights.erase(w);
  }

  // Delete unused weights
  for (const auto& w : unused_weights) {
    m_weights.erase(std::remove(m_weights.begin(), m_weights.end(), w),
                    m_weights.end());
  }

}

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

void model::evaluate(execution_mode mode) {

  // Return early if execution mode is invalid
  if (!is_execution_mode_valid(mode)) return;
  if (mode != execution_mode::validation
      && mode != execution_mode::testing) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid execution mode for evaluation";
    throw lbann_exception(err.str());
  }

  // Evaluate on all mini-batches
  reset_epoch(mode);
  do_evaluate_begin_cbs(mode);
  while (!evaluate_mini_batch(mode)) {}
  do_evaluate_end_cbs(mode);

}

void model::train(int num_epochs) {
  do_train_begin_cbs();
  for (int epoch = 0; epoch < num_epochs; ++epoch) {

    // Stop if training has been terminated
    if (get_terminate_training()) { break; }

    // Setup epoch
    reset_epoch(execution_mode::training);
    ++m_current_epoch;

    // Train on mini-batches
    do_epoch_begin_cbs();
    while (!train_mini_batch()) {}
    do_epoch_end_cbs();

    // Evaluate on validation set
    evaluate(execution_mode::validation);

  }
  do_train_end_cbs();
}

void model::reset_epoch(execution_mode mode) {
  set_execution_mode(mode);
  m_objective_function->clear_history();
  for (const auto& m : m_metrics) {
    m->reset_statistics(mode);
  }
  for (const auto& l : m_layers) {
    l->set_model(this);
  }
}

bool model::evaluate_mini_batch(execution_mode mode) {
  do_batch_begin_cbs(mode);
  forward_prop(mode);
  m_objective_function->evaluate();
  for (const auto& m : m_metrics) {
    m->evaluate(mode);
  }
  const bool finished = update_layers();
  do_batch_end_cbs(mode);
  return finished;
}

bool model::train_mini_batch() {
  do_batch_begin_cbs(execution_mode::training);

  // Forward prop step
  forward_prop(execution_mode::training);
  m_objective_function->evaluate();
  for (const auto& m : m_metrics) {
    m->evaluate(execution_mode::training);
  }

  // Backward prop step
  clear_error_signals();
  m_objective_function->differentiate();
  backward_prop();

  // Update step
  update_weights();
  const bool finished = update_layers();

  do_batch_end_cbs(execution_mode::training);
  ++m_current_step;
  return finished;
}

void model::clear_error_signals() {
  for (const auto& layer : m_layers) {
    layer->clear_error_signal();
  }
}

void model::forward_prop(execution_mode mode) {
  do_model_forward_prop_begin_cbs(mode);
  for (const auto& layer : m_layers) {
    do_layer_forward_prop_begin_cbs(mode, layer);
    layer->forward_prop();
    do_layer_forward_prop_end_cbs(mode, layer);
  }
  do_model_forward_prop_end_cbs(mode);
}

void model::backward_prop() {
  do_model_backward_prop_begin_cbs();
  for (int l = m_layers.size() - 1; l >= 0; --l) {
    Layer *layer = m_layers[l];
    do_layer_backward_prop_begin_cbs(layer);
    layer->back_prop();
    do_layer_backward_prop_end_cbs(layer);
  }
  do_model_backward_prop_end_cbs();
}

void model::update_weights() {
  do_model_optimize_begin_cbs();
  for (const auto& w : m_weights) {
    optimizer* opt = w->get_optimizer();
    if (opt != nullptr) {
      do_weight_optimize_begin_cbs(w);
      opt->step();
      do_weight_optimize_end_cbs(w);
    }
  }
  do_model_optimize_end_cbs();
}

bool model::update_layers() {
  bool finished = true;
  for (int l = m_layers.size() - 1; l >= 0; --l) {
    finished = m_layers[l]->update() && finished;
  }
  return finished;
}

////////////////////////////////////////////////////////////
// Callbacks
////////////////////////////////////////////////////////////

void model::do_train_begin_cbs() {
  for (const auto& cb : m_callbacks) {
    cb->on_train_begin(this);
  }
}

void model::do_train_end_cbs() {
  for (const auto& cb : m_callbacks) {
    cb->on_train_end(this);
  }
}

void model::do_evaluate_begin_cbs(execution_mode mode) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::validation:
      cb->on_validation_begin(this); break;
    case execution_mode::testing:
      cb->on_test_begin(this); break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
    }
  }
}

void model::do_evaluate_end_cbs(execution_mode mode) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::validation:
      cb->on_validation_end(this); break;
    case execution_mode::testing:
      cb->on_test_end(this); break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
    }
  }
}

void model::do_epoch_begin_cbs() {
  for (const auto& cb : m_callbacks) {
    cb->on_epoch_begin(this);
  }
}

void model::do_epoch_end_cbs() {
  for (const auto& cb : m_callbacks) {
    cb->on_epoch_end(this);
  }
}

void model::do_batch_begin_cbs(execution_mode mode) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_cur_step() % cb->get_batch_interval() == 0) {
        cb->on_batch_begin(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_batch_evaluate_begin(this);
      break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
    }
  }
}

void model::do_batch_end_cbs(execution_mode mode) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_cur_step() % cb->get_batch_interval() == 0) {
        cb->on_batch_end(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_batch_evaluate_end(this);
      break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
    }
  }
}

void model::do_model_forward_prop_begin_cbs(execution_mode mode) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_cur_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_begin(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_begin(this);
      break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
    }
  }
}

void model::do_model_forward_prop_end_cbs(execution_mode mode) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_cur_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_end(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_end(this);
      break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
    }
  }
}

void model::do_layer_forward_prop_begin_cbs(execution_mode mode, Layer *l) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_cur_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_begin(this, l);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_begin(this, l);
      break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
    }      
  }
}

void model::do_layer_forward_prop_end_cbs(execution_mode mode, Layer *l) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_cur_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_end(this, l);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_end(this, l);
      break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid execution mode";
      throw lbann_exception(err.str());
    }      
  }
}

void model::do_model_backward_prop_begin_cbs() {
  for (const auto& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_begin(this);
    }
  }
}

void model::do_model_backward_prop_end_cbs() {
  for (const auto& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_end(this);
    }
  }
}

void model::do_layer_backward_prop_begin_cbs(Layer *l) {
  for (const auto& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_begin(this, l);
    }
  }
}

void model::do_layer_backward_prop_end_cbs(Layer *l) {
  for (const auto& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_end(this, l);
    }
  }
}

void model::do_model_optimize_begin_cbs() {
  for (const auto& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_begin(this);
    }
  }
}

void model::do_model_optimize_end_cbs() {
  for (const auto& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_end(this);
    }
  }
}

void model::do_weight_optimize_begin_cbs(weights *w) {
  for (const auto& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_begin(this, w);
    }
  }
}

void model::do_weight_optimize_end_cbs(weights *w) {
  for (const auto& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_end(this, w);
    }
  }
}

////////////////////////////////////////////////////////////
// Summarizer
////////////////////////////////////////////////////////////

void model::summarize_stats(lbann_summary& summarizer) {
  for (const auto& layer : m_layers) {
    layer->summarize_stats(summarizer, get_cur_step());
  }
  summarizer.reduce_scalar("objective",
                           m_objective_function->get_history_mean_value(),
                           get_cur_step());
  summarizer.reduce_scalar(
    "objective_value_time",
    m_objective_function->get_value_time(),
    get_cur_step());
  summarizer.reduce_scalar(
    "objective_gradient_time",
    m_objective_function->get_gradient_time(),
    get_cur_step());
  m_objective_function->reset_counters();
}

void model::summarize_matrices(lbann_summary& summarizer) {
  for (const auto& layer : m_layers) {
    layer->summarize_matrices(summarizer, get_cur_step());
  }
}

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////


/** \brief Returns true if a checkpoint should be taken, false otherwise */
bool model::need_checkpoint() {
  /* TODO: since we're using clocks, this requires a bcast for each call,
   * we could use number of samples processed to make a local decision */

  // if none of our checkpoint conditions are set, assume we're not checkpointing
  if (m_checkpoint_epochs == 0 &&
      m_checkpoint_steps  == 0 &&
      m_checkpoint_secs   == 0.0) {
    return false;
  }

  // assume that we won't checkpoint
  int flag = 0;

  // if at start of epoch and evenly divide
  if (flag == 0 && m_checkpoint_epochs > 0) {
    if (at_epoch_start()) {
      flag = (int) (m_current_epoch % m_checkpoint_epochs == 0);
    }
  }

  // if our current step is evenly divisable by checkpoint steps,
  // take a checkpoint
  if (flag == 0 && m_checkpoint_steps > 0) {
    flag = (int) (m_current_step % m_checkpoint_steps == 0);
  }

  // check the clock if time-based checkpoint is enabled
  if (flag == 0 && m_checkpoint_secs != 0.0) {
    // have rank 0 determine whether we should checkpoint
    // to avoid issues with clock skew, we rely on rank 0 to make decision
    if (m_comm->am_world_master()) {
      // get the current time
      double current = MPI_Wtime();

      // compute time next checkpoint is due
      double next = m_checkpoint_last + m_checkpoint_secs;

      // determine whether it's time for a checkpoint
      flag = (current >= next);
    }

    // get flag from rank 0
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return (bool)flag;
}

/** \brief Writes a "latest" file which records epoch number and sample offset for the most recent checkpoint */
static bool write_latest(const char *dir, const char *name, int epoch, int train) {
  // define filename
  char filename[1024];
  sprintf(filename, "%s/%s", dir, name);

  // open the file for writing
  int fd = openwrite(filename);
  if (fd != -1) {
    write_uint32(fd, "epoch", (uint32_t)epoch);
    write_uint32(fd, "train", (uint32_t)train);

    // close our file
    closewrite(fd, filename);
  }

  return true;
}

/** \brief Reads the "latest" file and returns the epoch number and sample offset for most recent checkpoint */
static bool read_latest(const char *dir, const char *name, int *epochLast, int *trainLast) {
  // assume we don't have a file, we'll return -1 in that case
  *epochLast = -1;
  *trainLast = -1;

  // define filename
  char filename[1024];
  sprintf(filename, "%s/%s", dir, name);

  // open the file for reading
  int fd = openread(filename);
  if (fd != -1) {
    // read epoch from file
    uint32_t epoch;
    read_uint32(fd, "epoch", &epoch);
    *epochLast = (int) epoch;

    // read epoch from file
    uint32_t train;
    read_uint32(fd, "train", &train);
    *trainLast = train;

    // close our file
    closeread(fd, filename);
  }

  return true;
}

struct lbann_checkpoint {
  int epoch; // current epoch number
  int step;  // current offset into list of training example indices array
  float learning_rate; // current learning rate
};

//bool model::checkpointShared(TrainingParams& trainParams)
bool model::checkpointShared() {
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0) {
    return false;
  }

  // time how long this takes
  El::Timer timer;

  // get checkpoint directory
  const char *dir = m_checkpoint_dir.c_str();

  // read current epoch and step counters from model
  int epoch = m_current_epoch;
  int step  = m_current_step;

  // let user know we're saving a checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    timer.Start();
    printf("Checkpoint: epoch %d step %d ...\n", epoch, step);
    fflush(stdout);
  }

  // create top level directory
  //const char* dir = trainParams.ParameterDir.c_str();
  makedir(dir);

  // create subdirectory for this epoch
  char epochdir[1024];
  snprintf(epochdir, sizeof(epochdir), "%s/shared.epoch.%d.step.%d", dir, epoch, step);

  // start our checkpoint
  persist p;
  p.open_checkpoint(epochdir);

  // call virtual function to checkpoint model state
  this->save_to_checkpoint_shared(p);

  // close our checkpoint
  p.close_checkpoint();

  uint64_t bytes_count = p.get_bytes();

  // write epoch number to current file, we do this at the end so as to only update
  // this file when we know we have a new valid checkpoint
  if (m_comm->am_world_master()) {
    write_latest(dir, "shared.last", epoch, step);
  }

  // stop timer and report cost
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    double secs = timer.Stop();
    double bw = 0.0;
    if (secs > 0.0) {
      bw = ((double) bytes_count) / (secs * 1024.0 * 1024.0);
    }
    printf("Checkpoint complete: Epoch=%d Step=%d (%f secs, %llu bytes, %f MB/sec)\n",
           epoch, step, secs, (unsigned long long) bytes_count, bw
          );
    fflush(stdout);
  }

  // saved a checkpoint, update our last checkpoint time
  m_checkpoint_last = MPI_Wtime();

  return true;
}

bool model::restartShared() {
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0) {
    return false;
  }

  // get top level directory
  const char *dir = m_checkpoint_dir.c_str();

  // read epoch number from current file
  int epoch, step;
  if (m_comm->am_world_master()) {
    read_latest(dir, "shared.last", &epoch, &step);
  }
  MPI_Bcast(&epoch, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&step,  1, MPI_INT, 0, MPI_COMM_WORLD);

  // if we couldn't find the latest epoch, just return
  if (epoch < 0) {
    return false;
  }

  // time how long this takes
  El::Timer timer;

  // let user know we're restarting from a checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    timer.Start();
    printf("Restart: epoch %d ...\n", epoch);
    fflush(stdout);
  }

  // get subdirectory for this epoch
  char epochdir[1024];
  sprintf(epochdir, "%s/shared.epoch.%d.step.%d", dir, epoch, step);

  // open our checkpoint
  persist p;
  p.open_restart(epochdir);

  // call virtual function to restore model from checkpoint
  this->load_from_checkpoint_shared(p);

  // close our checkpoint
  p.close_restart();

  uint64_t bytes_count = p.get_bytes();

  // let user know we've completed reading our restart
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    double secs = timer.Stop();
    double bw = 0.0;
    if (secs > 0.0) {
      bw = ((double) bytes_count) / (secs * 1024.0 * 1024.0);
    }
    printf("Restart complete: Epoch=%d Step=%d (%f secs, %llu bytes, %f MB/sec)\n",
           epoch, step, secs, (unsigned long long) bytes_count, bw
          );
    fflush(stdout);
  }

  return true;
}
/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_model_header {
  uint32_t execution_mode;
  uint32_t terminate_training;
  uint64_t current_epoch;
  uint64_t current_step;
  uint32_t current_phase;
};

bool model::save_to_checkpoint_shared(persist& p) {
  // write out fields we need to save for model
  if (p.get_rank() == 0) {
    p.write_uint32(persist_type::train, "execution_mode",     (uint32_t) m_execution_mode);
    p.write_uint32(persist_type::train, "terminate_training", (uint32_t) m_terminate_training);
    p.write_uint64(persist_type::train, "current_epoch",      (uint64_t) m_current_epoch);
    p.write_uint64(persist_type::train, "current_step",       (uint64_t) m_current_step);
    p.write_uint32(persist_type::train, "current_phase",      (uint32_t) m_current_phase);
  }
  //for (weights *w : m_weights) {
  //  w->saveToCheckpointShared(p);
  //}
  return true;
}

bool model::load_from_checkpoint_shared(persist& p) {
  // have rank 0 read the file
  // read state from file
  struct lbann_model_header header;
  if (p.get_rank() == 0) {
    p.read_uint32(persist_type::train, "execution_mode",     &header.execution_mode);
    p.read_uint32(persist_type::train, "terminate_training", &header.terminate_training);
    p.read_uint64(persist_type::train, "current_epoch",      &header.current_epoch);
    p.read_uint64(persist_type::train, "current_step",       &header.current_step);
    p.read_uint32(persist_type::train, "current_phase",      &header.current_phase);
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

  // set our member params from values read from disk
  m_execution_mode     = (execution_mode) header.execution_mode;
  m_terminate_training = (bool)           header.terminate_training;
  m_current_epoch      = (int)            header.current_epoch;
  m_current_step       = (int)            header.current_step;
  m_current_phase      =                  header.current_phase;
  //for (weights *w : m_weights) {
  //  w->loadFromCheckpointShared(p);
  //}
  return true;
}

 // 0

}  // namespace lbann
