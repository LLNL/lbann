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
// directed_acyclic_graph .hpp .cpp - Directed acyclic graph neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/directed_acyclic_graph.hpp"
#include <stack>
#include <unordered_map>

namespace lbann {

directed_acyclic_graph_model::directed_acyclic_graph_model(lbann_comm *comm,
                                                           int mini_batch_size,
                                                           objective_function *obj_fn,
                                                           optimizer* default_optimizer)
  : model(comm, mini_batch_size, obj_fn, default_optimizer) {}

void directed_acyclic_graph_model::setup_layer_execution_order() {
  /* Note: This topological sort must be deterministic so that it
   * produces identical orderings when applied on different MPI
   * processes.
   */

  // Check if execution order is already valid
  if (is_topologically_sorted()) {
    return;
  }

  // Initialize data structures for topological sort
  std::stack<const Layer*> sorted_stack;
  std::stack<const Layer*> search_stack;
  std::unordered_map<const Layer*,bool> is_sorted;
  std::unordered_map<const Layer*,bool> is_visited;
  for (const auto& layer : m_layers) {
    is_sorted[layer] = false;
    is_visited[layer] = false;
    search_stack.push(layer);
  }

  // Perform depth-first searches until DAG has been traversed
  while (!search_stack.empty()) {
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
        for (const auto& child_layer : layer->get_child_layers()) {
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
