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
// recurrent .hpp .cpp - Recurrent neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/recurrent.hpp"
#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/layers/io/target/target_layer.hpp"
#include "lbann/layers/transform/constant.hpp"
#include "lbann/layers/activations/id.hpp"

namespace lbann {

recurrent_model::recurrent_model(lbann_comm *comm,
                                 int mini_batch_size,
                                 objective_function *obj_fn,
                                 optimizer *default_optimizer,
                                 int unroll_depth)
  : model(comm, mini_batch_size, obj_fn, default_optimizer) {
  m_unroll_depth = std::max(unroll_depth, 1);
}

void recurrent_model::setup_layer_topology() {
  model::setup_layer_topology();

  // Setup layer execution order
  setup_layer_execution_order();

  // We expect starting with input layer and ending with target layer
  /// @todo Design a more general interface
  if (m_layers.size() < 4) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "expected the first layer to be an input layer, "
        << "the second to be a slice layer, "
        << "the second to last to be a concatenation layer "
        << "and the last to be a target layer";
    throw lbann_exception(err.str());
  }
  auto input         = *(m_layers.begin());
  auto input_slice   = *(m_layers.begin() + 1);
  auto target_concat = *(m_layers.end() - 2);
  auto target        = *(m_layers.end() - 1);
  if (dynamic_cast<input_layer *>(m_layers.front()) == nullptr
      || input_slice->get_type() != "slice"
      || target_concat->get_type() != "concatenation"
      || dynamic_cast<target_layer *>(m_layers.back()) == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "expected the first layer to be an input layer, "
        << "the second to be a slice layer, "
        << "the second to last to be a concatenation layer "
        << "and the last to be a target layer";
    throw lbann_exception(err.str());
  }

  // Initialize pointers for input and target layer
  m_previous_roll_layer.clear();
  m_next_roll_layer.clear();
  m_previous_roll_layer[input] = input;
  m_previous_roll_layer[input_slice] = input_slice;
  m_previous_roll_layer[target_concat] = target_concat;
  m_previous_roll_layer[target] = target;
  m_next_roll_layer[input] = input;
  m_next_roll_layer[input_slice] = input_slice;
  m_next_roll_layer[target_concat] = target_concat;
  m_next_roll_layer[target] = target;
  std::vector<Layer *> input_children, target_parents;
  for (const auto& child : input_slice->get_child_layers()) {
    input_children.push_back(const_cast<Layer *>(child));
  }
  for (const auto& parent : target_concat->get_parent_layers()) {
    target_parents.push_back(const_cast<Layer *>(parent));
  }

  // Unroll network to desired depth
  std::vector<Layer *> previous_roll_layers(m_layers.begin() + 2,
                                            m_layers.end() - 2);
  const int roll_size = previous_roll_layers.size();
  for (int roll = 1; roll < m_unroll_depth; ++roll) {
    
    // Construct current roll by copying layers from previous roll
    std::vector<Layer *> current_roll_layers;
    for (const auto& previous_layer : previous_roll_layers) {
      Layer *current_layer = previous_layer->copy();
      current_roll_layers.push_back(current_layer);
      m_previous_roll_layer[current_layer] = previous_layer;
      m_next_roll_layer[previous_layer] = current_layer;
    }

    // Fix pointers in current roll
    for (const auto& current_layer : current_roll_layers) {
      auto layer_pointers = current_layer->get_layer_pointers();
      for (auto& layer_pointer : layer_pointers) {
        layer_pointer = m_next_roll_layer[layer_pointer];
      }
      current_layer->set_layer_pointers(layer_pointers);
    }

    // Fix pointers in input and target layers
    for (auto& child : input_children) {
      child = m_next_roll_layer[child];
      input_slice->add_child_layer(child);
    }
    for (auto& parent : target_parents) {
      parent = m_next_roll_layer[parent];
      target_concat->add_parent_layer(parent);
    }

    // Add current roll layers to model
    m_layers.insert(m_layers.end() - 2,
                    current_roll_layers.begin(),
                    current_roll_layers.end());
    previous_roll_layers = current_roll_layers;

  }

  // Rename layers
  for (int roll = 0; roll < m_unroll_depth; ++roll) {
    const std::string name_suffix = "_" + std::to_string(roll);
    const int roll_start = 2 + roll * roll_size;
    const int roll_end = 2 + (roll + 1) * roll_size;
    for (int i = roll_start; i < roll_end; ++i) {
      m_layers[i]->set_name(m_layers[i]->get_name() + name_suffix);
    }
  }

  // Fix pointers between adjacent rolls
  std::unordered_map<Layer *,bool> is_visited;
  std::vector<Layer *> placeholder_parents, placeholder_children;
  for (const auto& layer : m_layers) {

    // Fix pointers to parent layers
    std::vector<Layer *> parents;
    for (const auto& parent : layer->get_parent_layers()) {
      parents.push_back(const_cast<Layer *>(parent));
    }
    for (auto& parent : parents) {
      if (!is_visited[parent]) {
        if (m_previous_roll_layer[parent] != nullptr) {
          parent = m_previous_roll_layer[parent];
        } else {
          if (parent->get_num_neurons() <= 0) {
            std::stringstream err;
            err << __FILE__ << " " << __LINE__ << " :: "
                << layer->get_name() << " has ambiguous neuron "
                << "dimensions since it depends on "
                << parent->get_name() << ", which does not have "
                << "specified neuron dimensions. Consider inserting a "
                << "reshape layer after " << parent->get_name()
                << "to explicitly specify neuron dimensions.";
            throw lbann_exception(err.str());
          }
          Layer *placeholder;
          switch (parent->get_data_layout()) {
          case data_layout::DATA_PARALLEL:
            placeholder = new constant_layer<data_layout::DATA_PARALLEL>
                            (get_comm(), DataType(0), parent->get_neuron_dims());
            break;
          case data_layout::MODEL_PARALLEL:
            placeholder = new constant_layer<data_layout::MODEL_PARALLEL>
                            (get_comm(), DataType(0), parent->get_neuron_dims());
            break;
          default:
            std::stringstream err;
            err << __FILE__ << " " << __LINE__ << " :: " << "invalid data layout";
            throw lbann_exception(err.str());
          }
          placeholder->set_name(parent->get_name() + "_input_placeholder");
          placeholder->add_child_layer(layer);
          placeholder_parents.push_back(placeholder);
          parent = placeholder;
        }
      }
    }
    layer->clear_parent_layers();
    for (const auto& parent : parents) {
      layer->add_parent_layer(parent);
    }

    // Visit layer
    is_visited[layer] = true;

    // Fix pointers to parent layers
    std::vector<Layer *> children;
    for (const auto& child : layer->get_child_layers()) {
      children.push_back(const_cast<Layer *>(child));
    }
    for (auto& child : children) {
      if (is_visited[child]) {
        if (m_next_roll_layer[child] != nullptr) {
          child = m_next_roll_layer[child];
        } else {
          Layer *placeholder;
          switch (child->get_data_layout()) {
          case data_layout::DATA_PARALLEL:
            placeholder = new id_layer<data_layout::DATA_PARALLEL>(get_comm());
            break;
          case data_layout::MODEL_PARALLEL:
            placeholder = new id_layer<data_layout::MODEL_PARALLEL>(get_comm());
            break;
          default:
            std::stringstream err;
            err << __FILE__ << " " << __LINE__ << " :: " << "invalid data layout";
            throw lbann_exception(err.str());
          }
          placeholder->set_name(child->get_name() + "_output_placeholder");
          placeholder->add_parent_layer(layer);
          placeholder_children.push_back(placeholder);
          child = placeholder;
        }
      }
    }
    layer->clear_child_layers();
    for (const auto& child : children) {
      layer->add_child_layer(child);
    }

  }

  // Add placeholder layers to model
  m_layers.insert(m_layers.begin() + 2,
                  placeholder_parents.begin(),
                  placeholder_parents.end());
  m_layers.insert(m_layers.end() - 2,
                  placeholder_children.begin(),
                  placeholder_children.end());

}

void recurrent_model::setup_layer_execution_order() {
  model::setup_layer_execution_order();

  // Get layer graph
  std::set<int> nodes;
  std::map<int,std::set<int>> edges;
  construct_layer_graph(nodes, edges);
  const auto& edges_transpose = graph::transpose(nodes, edges);

  // Return immediately if no recurrence is detected
  if (!graph::is_cyclic(nodes, edges)) {
    return;
  }

  // Get topologically sorted condensation graph
  std::set<int> condensation_nodes;
  std::map<int,std::set<int>> components, condensation_edges;
  graph::condensation(nodes, edges,
                      components,
                      condensation_nodes, condensation_edges);

  // Sort nodes in each strongly connected component (SCC)
  std::vector<int> order;
  for (const auto& component : condensation_nodes) {
    const auto& component_nodes = components[component];

    // Find a node that requires external input
    // Note: Execution order is ambiguous if there are more than one
    // such nodes.
    int input_node = -1;
    for (const auto& node : component_nodes) {
      const auto& parents = graph::get_neighbors(node, edges_transpose);
      for (const auto& parent : parents) {
        if (!component_nodes.count(parent)) {
          if (input_node != node && input_node >= 0) {
            std::stringstream err;
            err << __FILE__ << " " << __LINE__ << " :: "
                << "recurrent network has ambiguous execution order "
                << "(each strongly connected component in the layer "
                << "graph can have at most one layer that requires "
                << "external input)";
            throw lbann_exception(err.str());
          }
          input_node = node;
        }
      }
    }

    // Induce subgraph on SCC and break cycles containing input node
    // Note: Execution order is ambiguous if this subgraph is cyclic.
    auto&& component_edges = graph::induce_subgraph(component_nodes,
                                                    edges);
    for (const auto& node
           : graph::depth_first_search(input_node, component_edges)) {
      component_edges[node].erase(input_node);
    }
    if (graph::is_cyclic(component_nodes, component_edges)) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "recurrent network has ambiguous execution order "
          << "(each strongly connected component in the layer graph "
          << "must be a DAG after removing edges toward input nodes)";
      throw lbann_exception(err.str());
    }

    // Sort SCC nodes topologically
    const auto& component_order = graph::topological_sort(component_nodes,
                                                          component_edges);
    order.insert(order.end(), component_order.begin(), component_order.end());

  }

  // Reorder layers
  permute_layers(order);
  
}

void recurrent_model::setup_layers() {
  for (const auto& layer : m_layers) {
    if (m_previous_roll_layer[layer] != nullptr) {
      layer->set_weights(m_previous_roll_layer[layer]->get_weights());
    }
    layer->set_model(this);
    layer->setup();
    layer->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << print_layer_description(layer) << std::endl;
    }
  }
}

}  // namespace lbann
