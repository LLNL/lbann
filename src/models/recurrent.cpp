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

#include <algorithm>
#include "lbann/models/recurrent.hpp"
#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/layers/io/target/generic_target_layer.hpp"
#include "lbann/layers/transform/slice.hpp"
#include "lbann/layers/transform/split.hpp"
#include "lbann/layers/transform/concatenation.hpp"
#include "lbann/layers/transform/constant.hpp"
#include "lbann/layers/transform/dummy.hpp"

namespace lbann {

namespace {

/** Setup input layer to match unrolled network.
 *  The first layer is assumed to be an input layer. A slice layer
 *  and a split layer are inserted after the input layer.
 */
void unroll_input_layer(int unroll_depth,
                        std::vector<Layer*>& layers,
                        std::unordered_map<const Layer*,Layer*>& prev_step_layer,
                        std::unordered_map<const Layer*,Layer*>& next_step_layer) {
    std::stringstream err;

  // We expect first layer to be an input layer
  auto&& input = dynamic_cast<generic_input_layer*>(layers.front());
  if (input == nullptr) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "expected the first layer to be an input layer";
    throw lbann_exception(err.str());
  }
  
  // Determine slice points
  const auto& input_dims = input->get_neuron_dims();
  const auto& slice_size = input_dims.front() / unroll_depth;
  std::vector<int> slice_points;
  for (int step = 0; step <= unroll_depth; ++step) {
    slice_points.push_back(slice_size * step);
  }

  // Construct slice and split layer
  Layer* slice = nullptr;
  Layer* split = nullptr;
  auto&& comm = input->get_comm();
  cudnn::cudnn_manager* cudnn = nullptr;
  switch (input->get_data_layout()) {
  case data_layout::DATA_PARALLEL:
    slice = new slice_layer<data_layout::DATA_PARALLEL>(comm, 0, slice_points, cudnn);
    split = new split_layer<data_layout::DATA_PARALLEL>(comm, cudnn);
    break;
  case data_layout::MODEL_PARALLEL:
    slice = new slice_layer<data_layout::MODEL_PARALLEL>(comm, 0, slice_points, cudnn);
    split = new split_layer<data_layout::MODEL_PARALLEL>(comm, cudnn);
    break;
  default:
    err << __FILE__ << " " << __LINE__ << " :: " << "invalid data layout";
    throw lbann_exception(err.str());
  }
  slice->set_name(input->get_name() + "_slice");
  split->set_name(input->get_name() + "_split");
  layers.insert(layers.begin() + 1, slice);
  layers.insert(layers.begin() + 2, split);

  // Setup relationships between split layer and child layers
  for (auto&& child : input->get_child_layers()) {
    split->add_child_layer(child);
    auto& child_parents = const_cast<Layer*>(child)->get_parent_layers();
    std::replace(child_parents.begin(), child_parents.end(),
                 static_cast<Layer*>(input), split);
  }
  input->clear_child_layers();

  // Setup relationship between input layer, slice layer, and split layer
  input->add_child_layer(slice);
  slice->add_parent_layer(input);
  slice->add_child_layer(split);
  split->add_parent_layer(slice);

  // Input layer and slice layer are not unrolled any further
  prev_step_layer[input] = input;
  prev_step_layer[slice] = slice;
  next_step_layer[input] = input;
  next_step_layer[slice] = slice;

}

/** Setup target layer to match unrolled network.
 *  The last layer is assumed to be a target layer. A concatenation
 *  layer is inserted before the target layer.
 */
void unroll_target_layer(int unroll_depth,
                         std::vector<Layer*>& layers,
                         std::unordered_map<const Layer*,Layer*>& prev_step_layer,
                         std::unordered_map<const Layer*,Layer*>& next_step_layer) {
  std::stringstream err;

  // We expect last layer to be a target layer
  auto&& target = dynamic_cast<generic_target_layer*>(layers.back());
  if (target == nullptr) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "expected the last layer to be a target layer";
    throw lbann_exception(err.str());
  }

  // Construct concatenation layer
  Layer* concat = nullptr;
  auto&& comm = target->get_comm();
  switch (target->get_data_layout()) {
  case data_layout::DATA_PARALLEL:
    concat = new concatenation_layer<data_layout::DATA_PARALLEL>(comm, 0, nullptr);
    break;
  case data_layout::MODEL_PARALLEL:
    concat = new concatenation_layer<data_layout::MODEL_PARALLEL>(comm, 0, nullptr);
    break;
  default:
    err << __FILE__ << " " << __LINE__ << " :: " << "invalid data layout";
    throw lbann_exception(err.str());
  }
  concat->set_name(target->get_name() + "_concat");
  layers.insert(layers.end() - 1, concat);

  // Setup relationships between concatenation layer and parent layers
  for (auto&& parent : target->get_parent_layers()) {
    concat->add_parent_layer(parent);
    auto& parent_children = const_cast<Layer*>(parent)->get_child_layers();
    std::replace(parent_children.begin(), parent_children.end(),
                 static_cast<Layer*>(target), concat);
  }
  target->clear_parent_layers();

  // Setup relationship between target layer and concatenation layer
  concat->add_child_layer(target);
  target->add_parent_layer(concat);

  // Target layer and concatenation layer are not unrolled any further
  prev_step_layer[target] = target;
  prev_step_layer[concat] = concat;
  next_step_layer[target] = target;
  next_step_layer[concat] = concat;

}

/** Duplicate layer network to achieve desired recurrence depth.
 *  The layers within each recurrence step have the same topology as
 *  the original network.
 */
void add_unrolled_layers(int unroll_depth,
                         std::vector<Layer*>& layers,
                         std::unordered_map<const Layer*,Layer*>& prev_step_layer,
                         std::unordered_map<const Layer*,Layer*>& next_step_layer) {

  // Unroll network to desired depth
  std::vector<Layer*> previous_step(layers.begin() + 2, layers.end() - 2);
  const int num_step_layers = previous_step.size();
  for (int step = 1; step < unroll_depth; ++step) {
    
    // Construct current step by copying layers from previous step
    std::vector<Layer*> current_step;
    for (const auto& previous_layer : previous_step) {
      auto&& current_layer = previous_layer->copy();
      current_step.push_back(current_layer);
      prev_step_layer[current_layer] = previous_layer;
      next_step_layer[previous_layer] = current_layer;
    }

    // Fix pointers within current step
    for (const auto& current_layer : current_step) {
      auto layer_pointers = current_layer->get_layer_pointers();
      for (auto& layer_pointer : layer_pointers) {
        layer_pointer = next_step_layer[layer_pointer];
      }
      current_layer->set_layer_pointers(layer_pointers);
    }

    // Add current step layers to model
    layers.insert(layers.end() - 2, current_step.begin(), current_step.end());
    previous_step = current_step;

  }

  // Rename layers
  for (int step = 0; step < unroll_depth; ++step) {
    const std::string name_suffix = "_step" + std::to_string(step);
    const int step_start = 2 + step * num_step_layers;
    const int step_end = 2 + (step + 1) * num_step_layers;
    for (int i = step_start; i < step_end; ++i) {
      layers[i]->set_name(layers[i]->get_name() + name_suffix);
    }
  }
  
}

/** Add placeholder layers for first and last recurrence step.
 *  If a layer in the first recurrence step expects input from an
 *  earlier recurrence step, we insert a zero-valued constant
 *  layer. If a layer in the last recurrence step expects to output to
 *  a later recurrence step, we insert a dummy layer.
 */
void add_placeholder_layers(std::vector<Layer*>& layers,
                            std::unordered_map<const Layer*,Layer*>& prev_step_layer,
                            std::unordered_map<const Layer*,Layer*>& next_step_layer) {

  std::unordered_map<const Layer*,bool> is_visited;
  std::vector<Layer*> input_placeholders, output_placeholders;
  for (const auto& l : layers) {

    // Create constant layers as input placeholders
    for (auto&& parent : l->get_parent_layers()) {
      if (!is_visited[parent] && prev_step_layer[parent] == nullptr) {
        if (parent->get_num_neurons() <= 0) {
          std::stringstream err;
          err << __FILE__ << " " << __LINE__ << " :: "
              << l->get_name() << " has ambiguous neuron "
              << "dimensions since it depends on "
              << parent->get_name() << ", which does not have "
              << "specified neuron dimensions. Consider inserting a "
              << "reshape layer after " << parent->get_name() << " "
              << "to explicitly specify neuron dimensions.";
          throw lbann_exception(err.str());
        }
        Layer* placeholder = nullptr;
        auto&& comm = parent->get_comm();
        switch (parent->get_data_layout()) {
        case data_layout::DATA_PARALLEL:
          placeholder = new constant_layer<data_layout::DATA_PARALLEL>(
                              comm,
                              DataType(0),
                              parent->get_neuron_dims()
                            );
          break;
        case data_layout::MODEL_PARALLEL:
          placeholder = new constant_layer<data_layout::MODEL_PARALLEL>(
                              comm,
                              DataType(0),
                              parent->get_neuron_dims()
                            );
          break;
        default:
          std::stringstream err;
          err << __FILE__ << " " << __LINE__ << " :: " << "invalid data layout";
          throw lbann_exception(err.str());
        }
        placeholder->set_name(parent->get_name() + "_input_placeholder");
        placeholder->add_child_layer(l);
        input_placeholders.push_back(placeholder);
        prev_step_layer[parent] = placeholder;
        next_step_layer[placeholder] = const_cast<Layer*>(parent);
      }
    }

    // Visit layer
    is_visited[l] = true;
    
    // Create dummy layers as output placeholders
    for (auto&& child : l->get_child_layers()) {
      if (is_visited[child] && next_step_layer[child] == nullptr) {
        Layer* placeholder = nullptr;
        auto&& comm = child->get_comm();
        switch (child->get_data_layout()) {
        case data_layout::DATA_PARALLEL:
          placeholder = new dummy_layer<data_layout::DATA_PARALLEL>(comm);
          break;
        case data_layout::MODEL_PARALLEL:
          placeholder = new dummy_layer<data_layout::MODEL_PARALLEL>(comm);
          break;
        default:
          std::stringstream err;
          err << __FILE__ << " " << __LINE__ << " :: " << "invalid data layout";
          throw lbann_exception(err.str());
        }
        placeholder->set_name(child->get_name() + "_output_placeholder");
        placeholder->add_parent_layer(l);
        output_placeholders.push_back(placeholder);
        next_step_layer[child] = placeholder;
        prev_step_layer[placeholder] = const_cast<Layer*>(child);
      }
    }
    
  }

  // Add placeholder layers to model
  layers.insert(layers.begin() + 2,
                input_placeholders.begin(),
                input_placeholders.end());
  layers.insert(layers.end() - 2,
                output_placeholders.begin(),
                output_placeholders.end());

}

/** Setup pointers between recurrence steps.
 *  If a layer's parent appears after the layer itself, change the
 *  parent to the corresponding layer in the previous recurrence
 *  step. Similarly, if a layer's child appears before the layer
 *  itself, change the child to the corresponding layer in the next
 *  recurrence step.
 */
void setup_unrolled_layer_pointers(std::vector<Layer*>& layers,
                                   const std::unordered_map<const Layer*,Layer*>& prev_step_layer,
                                   const std::unordered_map<const Layer*,Layer*>& next_step_layer) {

  std::unordered_map<const Layer*,bool> is_visited;
  for (auto&& l : layers) {
    for (auto& parent : l->get_parent_layers()) {
      if (!is_visited[parent]) {
        parent = prev_step_layer.at(parent);
      }
    }
    is_visited[l] = true;
    for (auto& child : l->get_child_layers()) {
      if (is_visited[child]) {
        child = next_step_layer.at(child);
      }
    }
  }
  
}

} // namespace
  
recurrent_model::recurrent_model(lbann_comm *comm,
                                 int mini_batch_size,
                                 objective_function *obj_fn,
                                 optimizer *default_optimizer,
                                 int unroll_depth)
  : directed_acyclic_graph_model(comm,
                                 mini_batch_size,
                                 obj_fn,
                                 default_optimizer) {
  m_unroll_depth = std::max(unroll_depth, 1);
}

void recurrent_model::setup_layer_topology() {
  
  // Make sure parent/child relationships are reciprocated
  for (const auto& l : m_layers) {
    for (const auto& parent : l->get_parent_layers()) {
      const_cast<Layer*>(parent)->add_child_layer(l);
    }
    for (const auto& child : l->get_child_layers()) {
      const_cast<Layer*>(child)->add_parent_layer(l);
    }
  }

  // Unroll layers
  if (m_layers.size() < 2) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "expected the first layer to be an input layer "
        << "and the last to be a target layer";
    throw lbann_exception(err.str());

  }
  m_previous_step_layer.clear();
  m_next_step_layer.clear();
  unroll_input_layer(m_unroll_depth, m_layers, m_previous_step_layer, m_next_step_layer);
  unroll_target_layer(m_unroll_depth, m_layers, m_previous_step_layer, m_next_step_layer);
  add_unrolled_layers(m_unroll_depth, m_layers, m_previous_step_layer, m_next_step_layer);
  add_placeholder_layers(m_layers, m_previous_step_layer, m_next_step_layer);
  setup_unrolled_layer_pointers(m_layers, m_previous_step_layer, m_next_step_layer);

  // Make sure unrolled topology is a DAG
  directed_acyclic_graph_model::setup_layer_topology();
  
}

void recurrent_model::setup_layers() {
  for (size_t i=0; i<m_layers.size(); ++i) {
    auto&& l = m_layers[i];

    // Set slice points for the inserted slice layer
    if (i == 1) {
      std::vector<int> slice_points;
      const auto& input_dims = m_layers[0]->get_neuron_dims();
      const auto& slice_size = input_dims[0] / m_unroll_depth;
      for (int step = 0; step <= m_unroll_depth; ++step) {
        slice_points.push_back(slice_size * step);
      }
      auto&& slice_dp = dynamic_cast<slice_layer<data_layout::DATA_PARALLEL>*>(l);
      auto&& slice_mp = dynamic_cast<slice_layer<data_layout::MODEL_PARALLEL>*>(l);
      if (slice_dp != nullptr) {
        slice_dp->get_slice_points() = slice_points;
      }
      if (slice_mp != nullptr) {
        slice_mp->get_slice_points() = slice_points;
      }
      if (slice_dp == nullptr && slice_mp == nullptr) {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
            << "expected the second layer to be a slice layer, "
            << "but " << l->get_name() << " is " << l->get_type();
        throw lbann_exception(err.str());
      }
    }
  
    // Corresponding weights in different steps share weights
    auto&& prev_step_layer = m_previous_step_layer[l];
    if (prev_step_layer != nullptr) {
      auto&& w = prev_step_layer->get_weights();
      if (!w.empty()) {
        l->set_weights(w);
      }
    }

    // Setup layer
    l->set_model(this);
    l->setup();
    l->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << print_layer_description(l) << std::endl;
    }

  }
}
  
}  // namespace lbann
