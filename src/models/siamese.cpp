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
#include "lbann/layers/io/io_layer.hpp"

namespace lbann {

siamese_model::siamese_model(lbann_comm *comm,
                             int mini_batch_size,
                             objective_function *obj_fn,
                             optimizer *default_optimizer,
                             int num_heads)
  : directed_acyclic_graph_model(comm, mini_batch_size, obj_fn, default_optimizer),
    m_num_heads(num_heads) {}

void siamese_model::freeze_layers_under_frozen_surface() {
  // Assuming m_layers is topologically sorted
  for (size_t i = m_layers.size(); i-- > 0u; ) {
    const auto layer = m_layers[i];
    if (layer->is_frozen()) {
      for (const auto& parent : layer->get_parent_layers()) {
        if (dynamic_cast<const io_layer*>(parent) == nullptr) {
          const_cast<Layer *>(parent)->freeze();
        }
      }
    }
  }
}

void siamese_model::setup_layer_topology() {

  /** @todo Handle case where heads have already been initialized. */
  m_follower_to_master_layer.clear();

  // Initialize network with master head
  directed_acyclic_graph_model::setup_layer_topology();
  setup_layer_execution_order();

  // Determine layers in master head
  int heads_start = m_layers.size();
  int heads_end = -1;
  for (size_t i = 0; i < m_layers.size(); ++i) {
    if (m_layers[i]->get_expected_num_child_layers() < 0) {
      heads_start = i + 1;
      break;
    }
  }
  for (int i = m_layers.size() - 1; i >= 0; --i) {
    if (m_layers[i]->get_expected_num_parent_layers() < 0) {
      heads_end = i;
      break;
    }
  }
  if (heads_start > heads_end) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "siamese models must have a fan-out layer before a fan-in layer "
        << "(found fan-out layer at position " << heads_start - 1
        << " and fan-in layer at position " << heads_end << ")";
    throw lbann_exception(err.str());
  }
  std::vector<Layer*> master_head(m_layers.begin() + heads_start,
                                  m_layers.begin() + heads_end);

  // Construct map from follower layers to master layers
  for (const auto& layer : m_layers) {
    m_follower_to_master_layer[layer] = layer;
  }

  // Duplicate master head
  for (int i = 1; i < m_num_heads; ++i) {
    std::string name_suffix = "_head" + std::to_string(i);

    // Construct map from master layers to follower layers
    std::unordered_map<Layer*,Layer*> master_to_follower_layer;
    for (const auto& master_layer : m_layers) {
      master_to_follower_layer[master_layer] = master_layer;
    }

    // Construct follower head by copying layers in master head
    std::vector<Layer*> follower_head;
    for (const auto& master_layer : master_head) {
      Layer* follower_layer = master_layer->copy();
      follower_layer->set_name(follower_layer->get_name() + name_suffix);
      follower_head.push_back(follower_layer);
      master_to_follower_layer[master_layer] = follower_layer;
      m_follower_to_master_layer[follower_layer] = master_layer;
    }

    // Fix pointers in follower head
    for (const auto& follower_layer : follower_head) {
      auto layer_pointers = follower_layer->get_layer_pointers();
      for (auto& layer_pointer : layer_pointers) {
        layer_pointer = master_to_follower_layer[layer_pointer];
      }
      follower_layer->set_layer_pointers(layer_pointers);
    }

    // Fix pointers at start and end of head
    m_layers[heads_start-1]->add_child_layer(follower_head.front());
    m_layers[heads_end]->add_parent_layer(follower_head.back());

    // Add follower head to model
    m_layers.insert(m_layers.begin() + heads_end,
                    follower_head.begin(),
                    follower_head.end());
    heads_end += follower_head.size();

  }

  // Rename layers in master head
  for (const auto& layer : master_head) {
    layer->set_name(layer->get_name() + "_head0");
  }

  // Make sure all parent/child relationships are reciprocated
  directed_acyclic_graph_model::setup_layer_topology();

  freeze_layers_under_frozen_surface();
}

void siamese_model::setup_layers() {
  for (const auto& layer : m_layers) {
    layer->set_weights(m_follower_to_master_layer[layer]->get_weights());
    layer->set_model(this);
    layer->setup();
    layer->check_setup();
    if (m_comm->am_world_master()) {
      std::cout << print_layer_description(layer) << std::endl;
    }
  }
}

}  // namespace lbann
