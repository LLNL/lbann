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
// siamese .hpp .cpp - Siamese neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/siamese.hpp"

namespace lbann {

siamese_model::siamese_model(lbann_comm *comm,
                             int mini_batch_size,
                             objective_function *obj_fn,
                             optimizer *default_optimizer,
                             int num_heads)
  : dag_model(comm, mini_batch_size, obj_fn, default_optimizer),
    m_num_heads(num_heads) {}

void siamese_model::setup() {

  // Sort layers topologically
  topologically_sort_layers();

  // Determine layers in master head
  int heads_start = m_layers.size();
  int heads_end = -1;
  for (size_t i = 0; i < m_layers.size(); ++i) {
    if (m_layers[i]->is_fan_out_layer()) {
      heads_start = i + 1;
      break;
    }
  }
  for (int i = m_layers.size() - 1; i >= 0; --i) {
    if (m_layers[i]->is_fan_in_layer()) {
      heads_end = i;
      break;
    }
  }
  if (heads_start > heads_end) {
    throw lbann_exception("siamese_model: siamese models must have a fan-out layer before a fan-in layer");
  }
  std::vector<Layer*> master_head(m_layers.begin() + heads_start,
                                  m_layers.begin() + heads_end);
  const int master_head_end = heads_end;

  // Construct map from follower layers to master layers
  std::unordered_map<Layer*,Layer*> follower_to_master_layer;
  for (Layer* master_layer : m_layers) {
    follower_to_master_layer[master_layer] = master_layer;
  }

  // Duplicate heads
  for (int i = 1; i < m_num_heads; ++i) {
    std::string name_suffix = "_" + std::to_string(i);

    // Construct map from master layers to follower layers
    std::unordered_map<Layer*,Layer*> master_to_follower_layer;
    for (Layer* master_layer : m_layers) {
      master_to_follower_layer[master_layer] = master_layer;
    }

    // Duplicate layers in master head
    std::vector<Layer*> follower_head;
    for (Layer* master_layer : master_head) {
      
      // Create copy of master layer
      Layer* follower_layer = master_layer->copy();
      follower_layer->set_name(follower_layer->get_name() + name_suffix);
      follower_head.push_back(follower_layer);

      // Add new layer to layer maps
      master_to_follower_layer[master_layer] = follower_layer;
      follower_to_master_layer[follower_layer] = master_layer;

      // Add layer to model
      m_layers.insert(m_layers.begin() + heads_end, follower_layer);
      heads_end++;

    }

    // Fix pointers in follower head
    for (Layer* follower_layer : follower_head) {
      std::vector<Layer*> master_layer_pointers = follower_layer->get_layer_pointers();
      std::vector<Layer*> follower_layer_pointers;
      for (Layer* master_layer_pointer : master_layer_pointers) {
        Layer* follower_layer_pointer = master_to_follower_layer[master_layer_pointer];
        follower_layer_pointers.push_back(follower_layer_pointer);
      }
      follower_layer->set_layer_pointers(follower_layer_pointers);
    }

    // Fix pointers at start and end of head
    m_layers[heads_start-1]->add_child_layer(follower_head.front());
    m_layers[heads_end]->add_parent_layer(follower_head.back());

  }

  // Setup layers before heads
  for (int i = 0; i < heads_start; ++i) {
    Layer* layer = m_layers[i];
    layer->setup();
    layer->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << print_layer_description(layer) << std::endl;
    }
  }

  // Setup master head
  for (int i = heads_start; i < master_head_end; ++i) {
    Layer* layer = m_layers[i];
    layer->setup();
    layer->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << print_layer_description(layer) << std::endl;
    }
  }

  // Setup follower heads
  for (int i = master_head_end; i < heads_end; ++i) {
    Layer* layer = m_layers[i];
    Layer* master_layer = follower_to_master_layer[layer];
    layer->set_weights(master_layer->get_weights());
    layer->setup();
    layer->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << print_layer_description(layer) << std::endl;
    }
  }

  // Setup layers after heads
  for (size_t i = heads_end; i < m_layers.size(); ++i) {
    Layer* layer = m_layers[i];
    layer->setup();
    layer->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << print_layer_description(layer) << std::endl;
    }
  }

  // Setup objective function
  m_objective_function->setup(*this);

  // Set up callbacks
  setup_callbacks();

}

}  // namespace lbann
