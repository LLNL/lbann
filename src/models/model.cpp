////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/models/model.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/callback_save_model.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/layers/transform/dummy.hpp"
#include "lbann/layers/transform/split.hpp"
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/objective_functions/layer_term.hpp"
#include "lbann/metrics/layer_metric.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include <string>
#include <unistd.h>
#include <iomanip>
#include <queue>
#include <unordered_set>
#include <lbann.pb.h>

#include "mpi.h"

namespace lbann {

// =============================================
// Life cycle functions
// =============================================

model::model(lbann_comm* comm,
             El::Int mini_batch_size,
             objective_function* obj_fn,
             optimizer* default_optimizer)
  : m_comm(comm),
    m_current_mini_batch_size(mini_batch_size),
    m_max_mini_batch_size(mini_batch_size),
    m_effective_mini_batch_size(mini_batch_size),
    m_default_optimizer(default_optimizer),
    m_objective_function(obj_fn) {

  // Default model name
  static El::Int num_models = 0;
  m_name = "model" + std::to_string(num_models);
  num_models++;

}

model::model(const model& other) :
  m_comm(other.m_comm),
  m_name(other.m_name),
  m_execution_mode(other.m_execution_mode),
  m_epoch(other.m_epoch),
  m_step(other.m_step),
  m_terminate_training(other.m_terminate_training),
  m_current_mini_batch_size(other.m_current_mini_batch_size),
  m_max_mini_batch_size(other.m_max_mini_batch_size),
  m_effective_mini_batch_size(other.m_effective_mini_batch_size),
  m_background_io_allowed(other.m_background_io_allowed) {

  // Deep copies
  m_default_optimizer = (other.m_default_optimizer ?
                         other.m_default_optimizer->copy() : nullptr);
  m_objective_function = (other.m_objective_function ?
                          other.m_objective_function->copy() : nullptr);
  m_metrics = other.m_metrics;
  m_callbacks = other.m_callbacks;
  for (auto& m : m_metrics) {
    m = m->copy();
  }
  for (auto& cb : m_callbacks) {
    cb = cb->copy();
  }

  // Copy layers
  std::unordered_map<Layer*,Layer*> layer_map;
  m_layers.reserve(other.m_layers.size());
  for (const auto& ptr : other.m_layers) {
    if (ptr == nullptr) { LBANN_ERROR("unexpected null pointer"); }
    auto* old_layer = ptr.get();
    auto* new_layer = old_layer->copy();
    new_layer->set_model(this);
    m_layers.emplace_back(new_layer);
    layer_map[old_layer] = new_layer;
  }

  // Copy weights
  m_weights = other.m_weights;
  std::unordered_map<weights*,weights*> weights_map;
  for (auto& w : m_weights) {
    auto&& w_copy = w->copy();
    weights_map[w] = w_copy;
    w = w_copy;
  }

  // Fix pointers
  remap_pointers(layer_map, weights_map);

}

model& model::operator=(const model& other) {

  // Delete objects
  if (m_objective_function != nullptr) { delete m_objective_function; }
  for (const auto& m : m_metrics)      { delete m; }
  for (const auto& cb : m_callbacks)   { delete cb; }
  for (const auto& w : m_weights)      { delete w; }

  // Shallow copies
  m_comm = other.m_comm;
  m_name = other.m_name;
  m_execution_mode = other.m_execution_mode;
  m_epoch = other.m_epoch;
  m_step = other.m_step;
  m_terminate_training = other.m_terminate_training;
  m_current_mini_batch_size = other.m_current_mini_batch_size;
  m_max_mini_batch_size = other.m_max_mini_batch_size;
  m_effective_mini_batch_size = other.m_effective_mini_batch_size;
  m_background_io_allowed = other.m_background_io_allowed;

  // Deep copies
  m_objective_function = other.m_objective_function;
  m_metrics            = other.m_metrics;
  m_callbacks          = other.m_callbacks;
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
  std::unordered_map<Layer*,Layer*> layer_map;
  m_layers.clear();
  m_layers.reserve(other.m_layers.size());
  for (const auto& ptr : other.m_layers) {
    if (ptr == nullptr) { LBANN_ERROR("unexpected null pointer"); }
    auto* old_layer = ptr.get();
    auto* new_layer = old_layer->copy();
    new_layer->set_model(this);
    m_layers.emplace_back(new_layer);
    layer_map[old_layer] = new_layer;
  }
  std::unordered_map<weights*,weights*> weights_map;
  for (auto& w : m_weights) {
    w = weights_map[w] = w->copy();
  }
  remap_pointers(layer_map, weights_map);

  return *this;
}

model::~model() {
  if (m_objective_function != nullptr) { delete m_objective_function; }
  if (m_default_optimizer != nullptr)  { delete m_default_optimizer; }
  for (const auto& w : m_weights)      { delete w; }
  for (const auto& m : m_metrics)      { delete m; }
  for (const auto& cb : m_callbacks)   { delete cb; }
}

// =============================================
// Access functions
// =============================================

void model::set_name(std::string name) {
  if (name.empty()) {
    std::ostringstream err;
    err << "attempted to rename model \"" << get_name() << "\" "
        << "with empty string";
    LBANN_ERROR(err.str());
  }
  m_name = std::move(name);
}

description model::get_description() const {

  // Construct description object
  description desc(get_name());
  desc.add("Type", get_type());

  // Layer topology
  description layer_topology_desc("Layer topology:");
  for (El::Int k = 0; k < get_num_layers(); ++k) {
    const auto& l = get_layer(k);
    std::stringstream ss;
    ss << l.get_name() << " (" << l.get_type() << "): {";
    const auto& parents = l.get_parent_layers();
    const auto& children = l.get_child_layers();
    for (size_t i = 0; i < parents.size(); ++i) {
      ss << (i > 0 ? ", " : "");
      if (parents[i] == nullptr) {
        ss << "unknown layer";
      } else {
        ss << parents[i]->get_name() << " (";
        const auto& dims = l.get_input_dims(i);
        for (size_t j = 0; j < dims.size(); ++j) {
          ss << (j > 0 ? "x" : "") << dims[j];
        }
        ss << ")";
      }
    }
    ss << "} -> {";
    for (size_t i = 0; i < children.size(); ++i) {
      ss << (i > 0 ? ", " : "");
      if (children[i] == nullptr) {
        ss << "unknown layer";
      } else {
        ss << children[i]->get_name() << " (";
        const auto& dims = l.get_output_dims(i);
        for (size_t j = 0; j < dims.size(); ++j) {
          ss << (j > 0 ? "x" : "") << dims[j];
        }
        ss << ")";
      }
    }
    ss << "}";
    layer_topology_desc.add(ss.str());
  }
  desc.add(std::string{});
  desc.add(layer_topology_desc);

  // Layer details
  description layer_details_desc("Layer details:");
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    layer_details_desc.add(get_layer(i).get_description());
  }
  desc.add(std::string{});
  desc.add(layer_details_desc);

  // Weights
  description weights_desc("Weights:");
  for (const auto* w : m_weights) {
    if (w == nullptr) {
      weights_desc.add("unknown weights");
    } else {
      weights_desc.add(w->get_description());
    }
  }
  desc.add(std::string{});
  desc.add(weights_desc);

  /// @todo Descriptions for objective function, metrics, callbacks

  // Result
  return desc;

}

El::Int model::get_num_layers() const noexcept {
  return m_layers.size();
}
Layer& model::get_layer(El::Int pos) {
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<Layer&>(static_cast<const model&>(*this).get_layer(pos));
}
const Layer& model::get_layer(El::Int pos) const {
  std::stringstream err;
  if (pos < 0 || pos >= get_num_layers()) {
    err << "could not access layer in model \"" << get_name() << "\" "
        << "(requested index " << pos << ", "
        << "but there are " << get_num_layers() << " layers)";
    LBANN_ERROR(err.str());
  } else if (m_layers[pos] == nullptr) {
    err << "model \"" << get_name() << "\" "
        << "has a null pointer in its layer list";
    LBANN_ERROR(err.str());
  }
  return *m_layers[pos];
}
std::vector<Layer*> model::get_layers() {
  std::vector<Layer*> layer_list;
  layer_list.reserve(m_layers.size());
  for (const auto& ptr : m_layers) {
    layer_list.push_back(ptr.get());
  }
  return layer_list;
}
const std::vector<Layer*> model::get_layers() const {
  std::vector<Layer*> layer_list;
  layer_list.reserve(m_layers.size());
  for (const auto& ptr : m_layers) {
    layer_list.push_back(ptr.get());
  }
  return layer_list;
}

std::vector<weights*> model::get_weights() {
  std::vector<weights*> weights_list;
  for (const auto& w : m_weights) {
    weights_list.push_back(w);
  }
  return weights_list;
}

const std::vector<weights*> model::get_weights() const {
  std::vector<weights*> weights_list;
  for (const auto& w : m_weights) {
    weights_list.push_back(w);
  }
  return weights_list;
}

void model::set_execution_mode(execution_mode mode) {
  m_execution_mode = mode;
}

execution_mode model::get_execution_mode() const noexcept {
  return m_execution_mode;
}

El::Int model::get_step() const noexcept {
  return get_step(get_execution_mode());
}

El::Int model::get_step(execution_mode mode) const noexcept {
  if (m_step.count(mode) > 0) {
    return m_step.at(mode);
  } else {
    return 0;
  }
}

int model::get_num_iterations_per_epoch(execution_mode mode) const {
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    const auto* input = dynamic_cast<const generic_input_layer*>(&get_layer(i));
    if (input != nullptr) {
      return input->get_num_iterations_per_epoch(mode);
    }
  }
  return 0;
}

// =============================================
// Model specification
// =============================================

void model::add_layer(std::unique_ptr<Layer> l) {
  std::stringstream err;

  // Check for null pointer
  if (l == nullptr) {
    err << "attempted to add a null pointer as a layer to "
        << "model \"" << get_name() << "\"";
    LBANN_ERROR(err.str());
  }

  // Check that the new layer name is unique
  // Note: Adding layers is O(n^2), but this is unlikely to be a
  // bottleneck. If it is, consider maintaining a hash table
  // containing all layer names (and properly updating it during
  // copies and pointer remaps).
  const auto& name = l->get_name();
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    if (get_layer(i).get_name() == name) {
      err << "attempted to add layer \"" << name << "\" to "
          << "model \"" << get_name() << "\", "
          << "but the model already contains a layer with that name";
      LBANN_ERROR(err.str());
    }
  }

  // Add layer to model
  m_layers.emplace_back(std::move(l));
  m_layers.back()->set_model(this);

}

void model::add_weights(weights* w) {
  std::stringstream err;

  // Check for null pointer
  if (w == nullptr) {
    err << "attempted to add a null pointer as weights to "
        << "model \"" << get_name() << "\"";
    LBANN_ERROR(err.str());
  }

  // Check that the new weights name is unique
  // Note: Adding weights is O(n^2), but this is unlikely to be a
  // bottleneck. If it is, consider maintaining a hash table
  // containing all weights names (and properly updating it during
  // copies and pointer remaps).
  const auto& name = w->get_name();
  for (const auto& w2 : m_weights) {
    if (w2->get_name() == name) {
      err << "attempted to add weights \"" << name << "\" to "
          << "model \"" << get_name() << "\", "
          << "but the model already contains weights with that name";
      LBANN_ERROR(err.str());
    }
  }

  // Add weights to model
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
  std::unordered_map<weights*,weights*> weights_map;
  std::unordered_map<Layer*,Layer*> layer_map;
  for (size_t i = 0; i < new_weights.size(); ++i) {
    m_weights[i] = weights_map[old_weights[i]] = new_weights[i];
  }
  remap_pointers(layer_map, weights_map);

  // Delete old weights
  for (const auto& w : old_weights) {
    delete w;
  }

}

void model::copy_trained_weights_from(std::vector<weights*>& new_weights) {
  if (new_weights.empty()) {
    if(m_comm->am_world_master()) std::cout << "No trained weights to copy " << std::endl;
    return;
  }
  for(size_t i = 0; i < new_weights.size(); ++i) {
     for (size_t j = 0; j < m_weights.size(); ++j) {
       //copy only trained weights (that is unfrozen layer)
       if(m_weights[j]->get_name() == new_weights[i]->get_name() && !new_weights[i]->is_frozen()) {
         #ifdef LBANN_DEBUG
         if(m_comm->am_world_master()) std::cout << " Replacing " << m_weights[j]->get_name() << " with " << new_weights[i]->get_name() << std::endl;
         #endif
         m_weights[j]->set_values(new_weights[i]->get_values());
       }
     }
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
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    const auto* input = dynamic_cast<const generic_input_layer*>(&get_layer(i));
    if (input != nullptr && !input->is_execution_mode_valid(mode)) {
      return false;
    }
  }
  return true;
}

void model::reorder_layers(const std::vector<El::Int>& gather_indices) {
  std::stringstream err;

  // Check that gather indices are in valid range
  const auto& num_layers = get_num_layers();
  if (std::any_of(gather_indices.begin(), gather_indices.end(),
                  [num_layers](El::Int index) {
                    return index < 0 || index >= num_layers;
                  })) {
    err << "attempted to reorder layer list for "
        << "model \"" << get_name() << "\" "
        << "with invalid gather index";
    LBANN_ERROR(err.str());
  }

  // Reorder layers
  std::vector<std::unique_ptr<Layer>> reordered_layers(gather_indices.size());
  for (size_t i = 0; i < gather_indices.size(); ++i) {
    reordered_layers[i] = std::move(m_layers[gather_indices[i]]);
  }
  m_layers = std::move(reordered_layers);

  // Check that layer list has no null pointers
  for (const auto& l : m_layers) {
    if (l == nullptr) {
      err << "found a null pointer in the layer list for "
          << "model \"" << get_name() << "\" after reordering";
      LBANN_ERROR(err.str());
    }
  }

}

void model::remap_pointers(const std::unordered_map<Layer*,Layer*>& layer_map,
                           const std::unordered_map<weights*,weights*>& weights_map) {

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
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    auto layer_pointers = l.get_layer_pointers();
    auto weights_pointers = l.get_weights();
    for (auto& ptr : layer_pointers) {
      if (layer_map.count(ptr) > 0) {
        ptr = layer_map.at(ptr);
      }
    }
    for (auto& ptr : weights_pointers) {
      if (weights_map.count(ptr) > 0) {
        ptr = weights_map.at(ptr);
      }
    }
    l.set_layer_pointers(layer_pointers);
    l.set_weights(weights_pointers);
  }

}

// =============================================
// Setup
// =============================================

void model::setup(std::shared_ptr<thread_pool> io_thread_pool) {
  // Setup I/O threads - set up before setting up the layers (input
  // layer depends on having a properly initialized thread pool)
  m_io_thread_pool = std::move(io_thread_pool);

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
  std::stringstream err;

  // Check that layer list is valid
  // Note: Throws an exception if the layer list contains two layers
  // with the same name or if a layer has a pointer to a layer in a
  // different model.
  std::unordered_set<Layer*> layer_set;
  std::unordered_set<std::string> layer_names;
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    if (layer_names.count(l.get_name()) > 0) {
      err << "model \"" << get_name() << "\" "
          << "has multiple layers named \"" << l.get_name() << "\"";
      LBANN_ERROR(err.str());
    }
    layer_set.insert(&l);
    layer_names.insert(l.get_name());
  }
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    for (const auto& ptr : l.get_layer_pointers()) {
      if (ptr != nullptr && layer_set.count(ptr) == 0) {
        err << "layer \"" << l.get_name() << "\" "
            << "(in model \"" << get_name() << "\") "
            << "has a pointer to layer " << ptr->get_name() << "\" ";
        if (ptr->get_model() == nullptr) {
          err << "(not in a model)";
        } else {
          err << "(in model \"" << ptr->get_model()->get_name() << "\")";
        }
        LBANN_ERROR(err.str());
      }
    }
  }

  // Make sure parent/child relationships are reciprocated
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    for (auto* parent : l.get_parent_layers()) {
      const_cast<Layer*>(parent)->add_child_layer(&l);
    }
    for (auto* child : l.get_child_layers()) {
      const_cast<Layer*>(child)->add_parent_layer(&l);
    }
  }

  // Add utility layers
  add_evaluation_layers(layer_set, layer_names);
  add_dummy_layers(layer_names);
  add_split_layers(layer_names);

}

void model::setup_layer_execution_order() {

  // Find input layers
  std::vector<El::Int> input_layers, other_layers;
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    if (dynamic_cast<generic_input_layer*>(&get_layer(i)) != nullptr) {
      input_layers.push_back(i);
    } else {
      other_layers.push_back(i);
    }
  }

  // Reorder layers so input layers are executed first
  std::vector<El::Int> gather_indices;
  gather_indices.insert(gather_indices.end(),
                        input_layers.begin(), input_layers.end());
  gather_indices.insert(gather_indices.end(),
                        other_layers.begin(), other_layers.end());
  reorder_layers(gather_indices);

}

void model::setup_layers() {
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    l.set_model(this);
    l.setup();
    l.check_setup();
  }
}

void model::setup_weights() {

  // List of used and unused weights
  std::unordered_set<weights*> weights_set(m_weights.begin(),
                                           m_weights.end());
  std::set<weights*> unused_weights(m_weights.begin(),
                                    m_weights.end());

  // Find weights used by layers
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    for (const auto& w : get_layer(i).get_weights()) {
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
  for (auto&& w : unused_weights) {
    m_weights.erase(std::remove(m_weights.begin(), m_weights.end(), w),
                    m_weights.end());
  }

  // Setup weights
  for (auto* w : m_weights) { w->setup(); }

}

void model::add_evaluation_layers(std::unordered_set<Layer*>& layer_set,
                                  std::unordered_set<std::string>& layer_names) {
  std::stringstream err;

  // Add evaluation layers corresponding to objective function layer terms
  for (auto* t : m_objective_function->get_terms()) {
    auto* term = dynamic_cast<layer_term*>(t);
    if (term != nullptr) {
      auto& l = term->get_layer();
      if (layer_set.count(&l) == 0) {
        err << "model \"" << get_name() << "\" "
            << "has an objective function layer term corresponding to "
            << "layer \"" << l.get_name() << "\", "
            << "which isn't in the model's list of layers";
        LBANN_ERROR(err.str());
      }
      if (dynamic_cast<abstract_evaluation_layer*>(&l) == nullptr) {

        // Create evaluation layer
        std::unique_ptr<Layer> eval(abstract_evaluation_layer::construct(
                                      l.get_comm(),
                                      l.get_data_layout(),
                                      l.get_device_allocation()));

        // Set evaluation layer name
        El::Int name_index = 1;
        std::string name = l.get_name() + "_eval";
        while (layer_names.count(name) > 0) {
          name_index++;
          name = l.get_name() + "_eval" + std::to_string(name_index);
        }
        eval->set_name(name);

        // Update workspace objects
        layer_set.insert(eval.get());
        layer_names.insert(eval->get_name());

        // Add evaluation layer to model
        l.add_child_layer(eval.get());
        eval->add_parent_layer(&l);
        term->set_layer(*eval);
        add_layer(std::move(eval));

      }
    }
  }

  // Add evaluation layers corresponding to layer metrics
  for (auto* m : m_metrics) {
    auto* met = dynamic_cast<layer_metric*>(m);
    if (met != nullptr) {
      auto& l = met->get_layer();
      if (layer_set.count(&l) == 0) {
        err << "layer metric \"" << met->name() << "\" "
            << "corresponds to layer \"" << l.get_name() << "\", "
            << "which is not in model \"" << get_name() << "\"";
        LBANN_ERROR(err.str());
      }
      if (dynamic_cast<abstract_evaluation_layer*>(&l) == nullptr) {

        // Create evaluation layer
        std::unique_ptr<Layer> eval(abstract_evaluation_layer::construct(
                                      l.get_comm(),
                                      l.get_data_layout(),
                                      l.get_device_allocation()));

        // Set evaluation layer name
        El::Int name_index = 1;
        std::string name = l.get_name() + "_eval";
        while (layer_names.count(name) > 0) {
          name_index++;
          name = l.get_name() + "_eval" + std::to_string(name_index);
        }
        eval->set_name(name);

        // Update workspace objects
        layer_set.insert(eval.get());
        layer_names.insert(eval->get_name());

        // Add evaluation layer to model
        l.add_child_layer(eval.get());
        eval->add_parent_layer(&l);
        met->set_layer(*eval);
        add_layer(std::move(eval));

      }
    }
  }

}

void model::add_dummy_layers(std::unordered_set<std::string>& layer_names) {
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    while (l.get_num_children() < l.get_expected_num_child_layers()) {

      // Create dummy layer
      std::unique_ptr<Layer> dummy;
      using args_tuple = std::tuple<data_layout,El::Device>;
      args_tuple args(l.get_data_layout(), l.get_device_allocation());
      if (args == args_tuple(data_layout::DATA_PARALLEL, El::Device::CPU)) {
        dummy.reset(new dummy_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(m_comm));
      }
      if (args == args_tuple(data_layout::MODEL_PARALLEL, El::Device::CPU)) {
        dummy.reset(new dummy_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>(m_comm));
      }
#ifdef LBANN_HAS_GPU
      if (args == args_tuple(data_layout::DATA_PARALLEL, El::Device::GPU)) {
        dummy.reset(new dummy_layer<data_layout::DATA_PARALLEL, El::Device::GPU>(m_comm));
      }
      if (args == args_tuple(data_layout::MODEL_PARALLEL, El::Device::GPU)) {
        dummy.reset(new dummy_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>(m_comm));
      }
#endif // LBANN_HAS_GPU
      if (dummy == nullptr) {
        std::stringstream err;
        err << "could not construct dummy layer corresponding to "
            << "layer \"" << l.get_name() << "\" "
            << "in model \"" << get_name() << "\"";
        LBANN_ERROR(err.str());
      }

      // Set dummy layer name
      El::Int name_index = 1;
      std::string name = l.get_name() + "_dummy";
      while (layer_names.count(name) > 0) {
        name_index++;
        name = l.get_name() + "_dummy" + std::to_string(name_index);
      }
      dummy->set_name(name);
      layer_names.insert(name);

      // Add dummy layer to model
      l.add_child_layer(dummy.get());
      dummy->add_parent_layer(&l);
      add_layer(std::move(dummy));

    }
  }
}

void model::add_split_layers(std::unordered_set<std::string>& layer_names) {
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);

    // Add split layer if layer expects one child but has multiple
    auto& children = l.get_child_layers();
    if (l.get_expected_num_child_layers() == 1 && children.size() != 1) {

      // Create split layer
      std::unique_ptr<Layer> split;
      using args_tuple = std::tuple<data_layout,El::Device>;
      args_tuple args(l.get_data_layout(), l.get_device_allocation());
      if (args == args_tuple(data_layout::DATA_PARALLEL, El::Device::CPU)) {
        split.reset(new split_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(m_comm));
      }
      if (args == args_tuple(data_layout::MODEL_PARALLEL, El::Device::CPU)) {
        split.reset(new split_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>(m_comm));
      }
#ifdef LBANN_HAS_GPU
      if (args == args_tuple(data_layout::DATA_PARALLEL, El::Device::GPU)) {
        split.reset(new split_layer<data_layout::DATA_PARALLEL, El::Device::GPU>(m_comm));
      }
      if (args == args_tuple(data_layout::MODEL_PARALLEL, El::Device::GPU)) {
        split.reset(new split_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>(m_comm));
      }
#endif // LBANN_HAS_GPU
      if (split == nullptr) {
        std::stringstream err;
        err << "could not construct split layer corresponding to "
            << "layer \"" << l.get_name() << "\" "
            << "in model \"" << get_name() << "\"";
        LBANN_ERROR(err.str());
      }

      // Set split layer name
      El::Int name_index = 1;
      std::string name = l.get_name() + "_split";
      while (layer_names.count(name) > 0) {
        name_index++;
        name = l.get_name() + "_split" + std::to_string(name_index);
      }
      split->set_name(name);
      layer_names.insert(name);

      // Setup relationships between split layer and child layers
      for (auto&& const_child : children) {
        auto* child = const_cast<Layer*>(const_child);
        split->add_child_layer(child);
        auto& child_parents = child->get_parent_layers();
        std::replace(child_parents.begin(), child_parents.end(),
                     &l, split.get());
      }

      // Setup relationship between current layer and split layer
      children.clear();
      l.add_child_layer(split.get());
      split->add_parent_layer(&l);

      // Add split layer to layer list
      add_layer(std::move(split));

    }

  }
}

// =============================================
// Execution
// =============================================

void model::evaluate(execution_mode mode, int num_batches) {

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
  reset_epoch_statistics(mode);
  reset_mode_and_model(mode);
  do_evaluate_begin_cbs(mode);
  if (num_batches > 0) {
    for (int i = 0; i < num_batches; i++) { evaluate_mini_batch(mode); }
  } else {
    while (!evaluate_mini_batch(mode)) {}
  }
  do_evaluate_end_cbs(mode);
}

void model::train(int num_epochs, int num_batches) {
  do_train_begin_cbs();
  for (int epoch = m_epoch; epoch < num_epochs; ++epoch) {
    if (get_terminate_training()) { break; }

    // Initialize epoch
    reset_mode_and_model(execution_mode::training);
    do_epoch_begin_cbs();

    // Training iterations
    if (num_batches > 0) {
      for (int i = 0; i < num_batches; i++) { train_mini_batch(); }
    } else {
      while (!train_mini_batch()) {}
    }

    // Finalize epoch
    ++m_epoch;
    reconcile_weight_values();
    do_epoch_end_cbs();
    reset_epoch_statistics(execution_mode::training);

    // Evaluate on validation set
    evaluate(execution_mode::validation);

  }
  do_train_end_cbs();
}


void model::collect_background_data_fetch(execution_mode mode) {
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto *input = dynamic_cast<generic_input_layer*>(&get_layer(i));
    if (input != nullptr) {
      input->collect_background_data_fetch(mode);
    }
  }
}

void model::make_data_store_preloaded(execution_mode mode) {
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto *input = dynamic_cast<generic_input_layer*>(&get_layer(i));
    if (input != nullptr) {
      auto *data_store = input->get_data_reader(mode)->get_data_store_ptr();
      if(data_store != nullptr && !data_store->is_preloaded()) {
        input->get_data_reader(mode)->get_data_store_ptr()->set_preload();
        input->get_data_reader(mode)->get_data_store_ptr()->set_explicit_loading(false);
      }
    }
  }
}

void model::mark_data_store_explicitly_loading(execution_mode mode) {
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto *input = dynamic_cast<generic_input_layer*>(&get_layer(i));
    if (input != nullptr) {
      auto *data_store = input->get_data_reader(mode)->get_data_store_ptr();
      if(data_store != nullptr && !data_store->is_preloaded()) {
        input->get_data_reader(mode)->get_data_store_ptr()->set_explicit_loading(true);
      }
    }
  }
}

// At the start of the epoch, set the execution mode and make sure
// that each layer points to this model
void model::reset_mode_and_model(execution_mode mode) {
  set_execution_mode(mode);
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    get_layer(i).set_model(this);
  }
}

// At the end of the epoch, clean up the objective function and metrics
void model::reset_epoch_statistics(execution_mode mode) {
  m_objective_function->reset_statistics(mode);
  for (const auto& m : m_metrics) {
    m->reset_statistics(mode);
  }
}

bool model::evaluate_mini_batch(execution_mode mode) {
  reset_mode_and_model(mode);
  do_batch_begin_cbs(mode);
  forward_prop(mode);
  m_objective_function->start_evaluation(mode, get_current_mini_batch_size());
  m_objective_function->finish_evaluation(mode, get_current_mini_batch_size());
  for (const auto& m : m_metrics) {
    m->evaluate(mode, get_current_mini_batch_size());
  }
  const bool finished = update_layers();

  // Increment mini-batch step
  /// @todo Move after the callbacks
  if (m_step.count(mode) < 1) { m_step[mode] = 0; }
  ++m_step[mode];

  do_batch_end_cbs(mode);
  return finished;
}

bool model::train_mini_batch() {
  constexpr execution_mode mode = execution_mode::training;
  reset_mode_and_model(mode);
  do_batch_begin_cbs(mode);


  bool finished;

#if defined(LBANN_HAVE_OMP_TASKLOOP)
  LBANN_OMP_PARALLEL
  {
    #pragma omp single
    {
#endif
  // Forward prop step
  clear_gradients();
  forward_prop(mode);
  // Result is not needed until the end of the mini-batch.
  m_objective_function->start_evaluation(mode, get_current_mini_batch_size());

  // Backward prop step
  m_objective_function->differentiate();
  backward_prop();
  m_objective_function->compute_weight_regularization();

  // Finish evaluation.
  m_objective_function->finish_evaluation(mode, get_current_mini_batch_size());
  for (const auto& m : m_metrics) {
    m->evaluate(mode, get_current_mini_batch_size());
  }

  // Update step
  update_weights();
  finished = update_layers();
#if defined(LBANN_HAVE_OMP_TASKLOOP)
    }
  }
#endif

  // Increment mini-batch step
  /// @todo Move after the callbacks
  if (m_step.count(mode) < 1) { m_step[mode] = 0; }
  ++m_step[mode];

  do_batch_end_cbs(execution_mode::training);
  return finished;
}

void model::clear_gradients() {
  for (const auto& w : m_weights) {
    optimizer* opt = w->get_optimizer();
    if (opt != nullptr) { opt->clear_gradient(); }
  }
}

void model::forward_prop(execution_mode mode) {
  do_model_forward_prop_begin_cbs(mode);
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    do_layer_forward_prop_begin_cbs(mode, &l);
    l.forward_prop();
    do_layer_forward_prop_end_cbs(mode, &l);
  }
  do_model_forward_prop_end_cbs(mode);
}

void model::backward_prop() {
  do_model_backward_prop_begin_cbs();
  for (El::Int i = get_num_layers()-1; i >= 0; --i) {

    // Perform backward prop step on current layer
    auto& l = get_layer(i);
    do_layer_backward_prop_begin_cbs(&l);
    l.back_prop();
    do_layer_backward_prop_end_cbs(&l);

    // Terminate early if all gradients have been computed
    bool all_gradients_computed = true;
    for (auto&& w : m_weights) {
      auto&& opt = w->get_optimizer();
      if (opt != nullptr && opt->get_num_gradient_sources() != 0) {
        all_gradients_computed = false;
        break;
      }
    }
    if (all_gradients_computed) { break; }

  }
  do_model_backward_prop_end_cbs();
}

void model::update_weights() {
  do_model_optimize_begin_cbs();
  for (El::Int i = m_weights.size()-1; i >= 0; --i) {
    auto& w = *m_weights[i];
    optimizer* opt = w.get_optimizer();
    if (opt != nullptr) {
      do_weight_optimize_begin_cbs(&w);
      opt->step();
      do_weight_optimize_end_cbs(&w);
    }
  }
  do_model_optimize_end_cbs();
}

bool model::update_layers() {
  bool finished = true;
  for (El::Int i = get_num_layers()-1; i >= 0; --i) {
    finished = get_layer(i).update() && finished;
  }
  return finished;
}

void model::reconcile_weight_values() {
  std::vector<Al::request> reqs(m_weights.size());
  for (El::Int i = m_weights.size()-1; i >= 0; --i) {
    m_weights[i]->reconcile_values(reqs[i]);
  }
  for (auto& req : reqs) { m_comm->wait(req); }
}

// =============================================
// Callbacks
// =============================================

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
      if (get_step() % cb->get_batch_interval() == 0) {
        cb->on_batch_begin(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_batch_evaluate_begin(this);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void model::do_batch_end_cbs(execution_mode mode) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_step() % cb->get_batch_interval() == 0) {
        cb->on_batch_end(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_batch_evaluate_end(this);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void model::do_model_forward_prop_begin_cbs(execution_mode mode) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_begin(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_begin(this);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void model::do_model_forward_prop_end_cbs(execution_mode mode) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_end(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_end(this);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

/** @todo Consistent behavior between train, validation, and test
 *  modes
 */
void model::do_layer_forward_prop_begin_cbs(execution_mode mode, Layer *l) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_begin(this, l);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_begin(this, l);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

/** @todo Consistent behavior between train, validation, and test
 *  modes
 */
void model::do_layer_forward_prop_end_cbs(execution_mode mode, Layer *l) {
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_end(this, l);
      }
      break;
    case execution_mode::validation:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_end(this, l);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void model::do_model_backward_prop_begin_cbs() {
  for (const auto& cb : m_callbacks) {
    if (get_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_begin(this);
    }
  }
}

void model::do_model_backward_prop_end_cbs() {
  for (const auto& cb : m_callbacks) {
    if (get_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_end(this);
    }
  }
}

void model::do_layer_backward_prop_begin_cbs(Layer *l) {
  for (const auto& cb : m_callbacks) {
    if (get_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_begin(this, l);
    }
  }
}

void model::do_layer_backward_prop_end_cbs(Layer *l) {
  for (const auto& cb : m_callbacks) {
    if (get_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_end(this, l);
    }
  }
}

void model::do_model_optimize_begin_cbs() {
  for (const auto& cb : m_callbacks) {
    if (get_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_begin(this);
    }
  }
}

void model::do_model_optimize_end_cbs() {
  for (const auto& cb : m_callbacks) {
    if (get_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_end(this);
    }
  }
}

void model::do_weight_optimize_begin_cbs(weights *w) {
  for (const auto& cb : m_callbacks) {
    if (get_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_begin(this, w);
    }
  }
}

void model::do_weight_optimize_end_cbs(weights *w) {
  for (const auto& cb : m_callbacks) {
    if (get_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_end(this, w);
    }
  }
}

// =============================================
// Summarizer
// =============================================

void model::summarize_stats(lbann_summary& summarizer) {
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    get_layer(i).summarize_stats(summarizer, get_step(execution_mode::training));
  }
  summarizer.reduce_scalar("objective",
                           m_objective_function->get_mean_value(m_execution_mode),
                           get_step(execution_mode::training));
  summarizer.reduce_scalar(
    "objective_evaluation_time",
    m_objective_function->get_evaluation_time(),
    get_step(execution_mode::training));
  summarizer.reduce_scalar(
    "objective_differentiation_time",
    m_objective_function->get_differentiation_time(),
    get_step(execution_mode::training));
  m_objective_function->reset_counters();
  double total_metric_time = 0.0;
  for (auto&& m : m_metrics) {
    total_metric_time += m->get_evaluate_time();
    m->reset_counters();
  }
  summarizer.reduce_scalar(
    "metric_evaluation_time",
    total_metric_time,
    get_step(execution_mode::training));
}

void model::summarize_matrices(lbann_summary& summarizer) {
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    get_layer(i).summarize_matrices(summarizer, get_step(execution_mode::training));
  }
}

// =============================================
// Checkpointing
// =============================================

/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_model_header {
  uint32_t execution_mode;
  uint32_t terminate_training;
  uint64_t epoch;
  uint64_t training_step;
  uint64_t validation_step;
  uint64_t testing_step;
  uint32_t max_mini_batch_size;
  uint32_t current_mini_batch_size;
  uint32_t callback_type;;
};

bool model::save_to_checkpoint_shared(persist& p) {
  // write out fields we need to save for model
  if (p.get_cb_type() != callback_type::validation) {
    if (m_comm->am_trainer_master()) {
      p.write_uint32(persist_type::train, "execution_mode",     (uint32_t) m_execution_mode);
      p.write_uint32(persist_type::train, "terminate_training", (uint32_t) m_terminate_training);
      p.write_uint64(persist_type::train, "epoch",              (uint64_t) m_epoch);
      p.write_uint64(persist_type::train, "training_step",      (uint64_t) get_step(execution_mode::training));
      p.write_uint64(persist_type::train, "testing_step",       (uint64_t) get_step(execution_mode::testing));
      p.write_uint32(persist_type::train, "max_mini_batch_size",      (uint32_t) m_max_mini_batch_size);
      p.write_uint32(persist_type::train, "current_mini_batch_size",      (uint32_t) m_current_mini_batch_size);
      p.write_uint32(persist_type::train, "persist_callback_type",      (uint32_t) p.get_cb_type());
      if(p.get_cb_type() == callback_type::batch)
        p.write_uint64(persist_type::validate, "validation_step",       (uint64_t) get_step(execution_mode::validation));
    }

    for (weights *w : m_weights) {
      w->save_to_checkpoint_shared(p);
    }

    for (El::Int i = 0; i < get_num_layers(); ++i) {
      if (!get_layer(i).save_to_checkpoint_shared(p)) {
        return false;
      }
    }
    if(p.get_cb_type() == callback_type::batch || get_num_iterations_per_epoch(execution_mode::validation) == 0){
      save_rng_to_checkpoint_shared(p, m_comm);
      for (const auto& m : m_metrics) {
        m->save_to_checkpoint_shared(p);
      }
    }
  }
  else{
    if (m_comm->am_trainer_master()) {
      p.write_uint64(persist_type::validate, "validation_step",       (uint64_t) get_step(execution_mode::validation));
    }
    save_rng_to_checkpoint_shared(p, m_comm);
    for (weights *w : m_weights) {
      w->save_to_checkpoint_shared(p);
    }
    for (El::Int i = 0; i < get_num_layers(); ++i) {
      if (!get_layer(i).save_to_checkpoint_shared(p)) {
        return false;
      }
    }
    for (const auto& m : m_metrics) {
      m->save_to_checkpoint_shared(p);
    }
  }
  return true;
}

bool model::load_from_checkpoint_shared(persist& p) {
  // have rank 0 read the file
  // read state from file
  struct lbann_model_header header;
  // Assume checkpoint reload from epoch end not step end
  if (m_comm->am_trainer_master()) {
    if (p.get_cb_type() != callback_type::validation) {
      p.read_uint32(persist_type::train, "execution_mode",     &header.execution_mode);
      p.read_uint32(persist_type::train, "terminate_training", &header.terminate_training);
      p.read_uint64(persist_type::train, "epoch",              &header.epoch);
      p.read_uint64(persist_type::train, "training_step",       &header.training_step);
      if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
        p.read_uint64(persist_type::validate, "validation_step",       &header.validation_step);
      p.read_uint64(persist_type::train, "testing_step",       &header.testing_step);
      p.read_uint32(persist_type::train, "max_mini_batch_size",      &header.max_mini_batch_size);
      p.read_uint32(persist_type::train, "current_mini_batch_size",      &header.current_mini_batch_size);
      p.read_uint32(persist_type::train, "persist_callback_type",     &header.callback_type);
    } else {
      p.read_uint64(persist_type::validate, "validation_step",       &header.validation_step);
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
    m_epoch              = (int)            header.epoch;
    m_step[execution_mode::training] = (int) header.training_step;
    if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
      m_step[execution_mode::validation] = (int) header.validation_step;
    m_step[execution_mode::testing] = (int) header.testing_step;
    m_max_mini_batch_size = (int)           header.max_mini_batch_size;
    m_current_mini_batch_size = (int)       header.current_mini_batch_size;
    // set state of persist object to know which type of ckpt we are returning from.
    p.set_cb_type((callback_type) header.callback_type);
  } else {
    m_step[execution_mode::validation] = (int) header.validation_step;
  }

  for (weights *w : m_weights) {
    w->load_from_checkpoint_shared(p);
  }

  // read in each layer
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    if (!get_layer(i).load_from_checkpoint_shared(p)) {
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

bool model::save_to_checkpoint_distributed(persist& p){
  // write out fields we need to save for model
  if (p.get_cb_type() != callback_type::validation) {
    p.write_uint32(persist_type::train, "execution_mode",     (uint32_t) m_execution_mode);
    p.write_uint32(persist_type::train, "terminate_training", (uint32_t) m_terminate_training);
    p.write_uint64(persist_type::train, "epoch",              (uint64_t) m_epoch);
    p.write_uint64(persist_type::train, "training_step",      (uint64_t) get_step(execution_mode::training));
    p.write_uint64(persist_type::train, "testing_step",       (uint64_t) get_step(execution_mode::testing));
    p.write_uint32(persist_type::train, "max_mini_batch_size",      (uint32_t) m_max_mini_batch_size);
    p.write_uint32(persist_type::train, "current_mini_batch_size",      (uint32_t) m_current_mini_batch_size);
    p.write_uint32(persist_type::train, "persist_callback_type",      (uint32_t) p.get_cb_type());
    if(p.get_cb_type() == callback_type::batch)
      p.write_uint64(persist_type::validate, "validataion_step",       (uint64_t) get_step(execution_mode::validation));

    for (weights *w : m_weights) {
      w->save_to_checkpoint_distributed(p);
    }

    for (El::Int i = 0; i < get_num_layers(); ++i) {
      if (!get_layer(i).save_to_checkpoint_distributed(p)) {
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
    p.write_uint64(persist_type::validate, "validataion_step",       (uint64_t) get_step(execution_mode::validation));
    save_rng_to_checkpoint_shared(p, m_comm);

    for (El::Int i = 0; i < get_num_layers(); ++i) {
      if (!get_layer(i).save_to_checkpoint_distributed(p)) {
        return false;
      }
    }
    for (const auto& m : m_metrics) {
      m->save_to_checkpoint_distributed(p);
    }
  }
  return true;
}

bool model::load_from_checkpoint_distributed(persist& p){
  struct lbann_model_header header;
  p.read_uint32(persist_type::train, "execution_mode",     &header.execution_mode);
  p.read_uint32(persist_type::train, "terminate_training", &header.terminate_training);
  p.read_uint64(persist_type::train, "epoch",              &header.epoch);
  p.read_uint64(persist_type::train, "training_step",      &header.training_step);
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
    p.read_uint64(persist_type::validate, "validation_step",       &header.validation_step);
  p.read_uint64(persist_type::train, "testing_step",               &header.testing_step);
  p.read_uint32(persist_type::train, "max_mini_batch_size",      &header.max_mini_batch_size);
  p.read_uint32(persist_type::train, "current_mini_batch_size",      &header.current_mini_batch_size);
  p.read_uint32(persist_type::train, "persist_callback_type",     &header.callback_type);

  m_execution_mode     = (execution_mode) header.execution_mode;
  m_terminate_training = (bool)           header.terminate_training;
  m_epoch              = (int)            header.epoch;
  m_step[execution_mode::training] = (int) header.training_step;
  if(get_num_iterations_per_epoch(execution_mode::validation) != 0)
    m_step[execution_mode::validation] = (int) header.validation_step;
  m_step[execution_mode::testing] = (int) header.testing_step;
  m_max_mini_batch_size = (int)           header.max_mini_batch_size;
  m_current_mini_batch_size = (int)       header.current_mini_batch_size;

  p.set_cb_type((callback_type) header.callback_type);
  load_rng_from_checkpoint_shared(p, m_comm);

  for (weights *w : m_weights) {
    w->load_from_checkpoint_distributed(p);
  }

  for (El::Int i = 0; i < get_num_layers(); ++i) {
    if (!get_layer(i).load_from_checkpoint_distributed(p)) {
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

void model::write_proto(lbann_data::Model* proto) {
  proto->Clear();
  if (m_comm->am_world_master())
    proto->set_mini_batch_size(m_max_mini_batch_size);
}


bool model::save_weights(persist& p) {
  // write out fields we need to save a model's weights
  for (weights *w : m_weights) {
    w->save_to_checkpoint_shared(p);
  }
  return true;
}

bool model::reload_weights(const std::string latest, const std::vector<std::string>& weight_list) {
  // load weights that appear in weight list.
  for(weights *w : m_weights) {
    w->load_from_save(latest,weight_list);
  }
  return true;
}

bool model::save_model() {
  for (auto* c : m_callbacks) {
    auto *cb = dynamic_cast<lbann_callback_save_model*>(c);
    if(cb != nullptr) {
      return cb->save_model(this);
    }
  }
  if(m_comm->am_trainer_master()) {
    LBANN_WARNING("save_model was called, but the callback_save_model was not loaded");
  }
  return false;
}

}  // namespace lbann
