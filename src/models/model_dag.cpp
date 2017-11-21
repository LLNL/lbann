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
#include "lbann/layers/io/target/target_layer.hpp"

#include <iomanip>
#include <vector>
#include <stack>
#include <unordered_map>

namespace lbann {

dag_model::dag_model(lbann_comm *comm,
                     int mini_batch_size,
                     objective_function *obj_fn,
                     optimizer* default_optimizer)
  : model(comm, mini_batch_size, obj_fn, default_optimizer) {}

void dag_model::setup() {

  // Make sure parent and child relationships are reciprocated
  for (Layer* layer : m_layers) {
    for (const Layer *parent_layer : layer->get_parent_layers()) {
      const_cast<Layer*>(parent_layer)->add_child_layer(layer);
    }
    for (const Layer *child_layer : layer->get_child_layers()) {
      const_cast<Layer*>(child_layer)->add_parent_layer(layer);
    }
  }

  // Sort layers topologically
  topologically_sort_layers();

  // Setup each layer
  for (Layer* layer : m_layers) {
    layer->set_neural_network_model(this);
    layer->setup();
    layer->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << "[" << std::setw(18) << layer->get_type() <<  "] Set up a layer with input " << std::setw(7) << layer->get_num_prev_neurons() << " and " << std::setw(7) << layer->get_num_neurons() << " neurons."  << std::endl;
    }
  }

  // Setup objective function
  m_objective_function->setup(*this);

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
    search_stack.push(layer);
  }

  // Perform depth-first searches until DAG has been traversed
  while(!search_stack.empty()) {
    const Layer* layer = search_stack.top();
    search_stack.pop();
    if (!is_sorted[layer]) {
      if (is_visited[layer]) {
        // Move search layer to sorted stack if we have visited already
        sorted_stack.push(layer);
        is_sorted[layer] = true;
      } else {
        // Visit search layer by adding children to search stack
        search_stack.push(layer);
        is_visited[layer] = true;
        for (const Layer* child_layer : layer->get_child_layers()) {
          if (is_visited[child_layer] && !is_sorted[child_layer]) {
            throw lbann_exception("model_dag: detected a cycle in network graph");
          }
          search_stack.push(child_layer);
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

}  // namespace lbann
