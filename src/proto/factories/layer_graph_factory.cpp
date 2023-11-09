////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/factories.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/layers/learning/fully_connected.hpp"

#include "lbann/proto/model.pb.h"
#include "lbann/proto/trainer.pb.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace lbann {
namespace proto {

namespace {

/** Setup parent/child relationships between layers. */
void setup_parents_and_children(
  lbann_comm* comm,
  const std::vector<OwningLayerPtr>& layers,
  const std::unordered_map<std::string, ViewingLayerPtr>& names_to_layers,
  const lbann_data::Model& proto_model)
{
  for (int i = 0; i < proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);
    for (int pid = 0; pid < proto_layer.parents_size(); ++pid) {
      auto const& parent = proto_layer.parents(pid);
      if (names_to_layers.count(parent) == 0) {
        LBANN_ERROR("Could not find parent layer \"",
                    parent,
                    "\" for layer \"",
                    layers[i]->get_name(),
                    "\"");
      }
      layers[i]->add_parent_layer(names_to_layers.at(parent));
    }
    for (int cid = 0; cid < proto_layer.children_size(); ++cid) {
      auto const& child = proto_layer.children(cid);
      if (names_to_layers.count(child) == 0) {
        LBANN_ERROR("Could not find child layer \"",
                    child,
                    "\" for layer \"",
                    layers[i]->get_name(),
                    "\"");
      }
      layers[i]->add_child_layer(names_to_layers.at(child));
    }
  }
}

void setup_hints(
  const std::vector<OwningLayerPtr>& layers,
  const std::unordered_map<std::string, ViewingLayerPtr>& names_to_layers,
  const lbann_data::Model& proto_model)
{
  for (int i = 0; i < proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);
    const auto& hint = proto_layer.hint_layer();
    if (!hint.empty()) {
      if (names_to_layers.count(hint) == 0)
        LBANN_ERROR("Could not find hint layer \"",
                    hint,
                    "\" "
                    "for layer \"",
                    layers[i]->get_name(),
                    "\"");
      layers[i]->set_hint_layer(names_to_layers.at(hint));
    }
  }
}

} // namespace

std::vector<OwningLayerPtr>
construct_layer_graph(lbann_comm* comm,
                      const lbann_data::Trainer& proto_trainer,
                      const lbann_data::Model& proto_model)
{

  // List of layers
  std::vector<OwningLayerPtr> layers;
  layers.reserve(proto_model.layer_size());

  // Map from names to layer pointers
  std::unordered_map<std::string, ViewingLayerPtr> names_to_layers;

  // Create each layer in prototext
  for (int i = 0; i < proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);

    // Check that layer name is valid
    std::string name = proto_layer.name();
    if (!name.empty()) {
      // FIXME (trb 04/15/22): I don't think this should still be an
      // issue since parents/children lists are now "repeated"
      // protobuf fields and not space-separated strings. OTOH, we
      // shouldn't allow names like "foo\vbar" or "fu\n\name".
      if (name.find_first_of(" \n\r\t\v\f") != std::string::npos)
        LBANN_ERROR("Layer name \"",
                    name,
                    "\" is invalid since it contains whitespace.");
      if (names_to_layers.count(name) != 0)
        LBANN_ERROR("Layer name \"", name, "\" is not unique.");
    }

    // Get parameters from prototext
    const auto model_disable_gpus = proto_model.disable_cuda();

    const auto& layout_str = proto_layer.data_layout();
    data_layout layout =
      (layout_str.empty() ? data_layout::DATA_PARALLEL
                          : data_layout_from_string(layout_str));

    El::Device device = El::Device::CPU;
#ifdef LBANN_HAS_GPU
    // Input layers must be on CPU
    if (!proto_layer.has_input() && !model_disable_gpus) {
      const auto& device_str = proto_layer.device_allocation();
      device =
        (device_str.empty() ? El::Device::GPU : device_from_string(device_str));
    }
#else
    (void)model_disable_gpus;
#endif // LBANN_HAS_GPU

    auto proto_datatype = resolve_default_datatype(proto_layer.datatype());

    // Construct layer
    OwningLayerPtr l;
#define TEMPLATE_INSTANTIATION(TensorDataType, T_layout, T_device)             \
  do {                                                                         \
    if (proto_datatype == TypeToProtoDataType<TensorDataType>::value &&        \
        layout == T_layout && device == T_device) {                            \
      l = construct_layer<TensorDataType, T_layout, T_device>(comm,            \
                                                              proto_layer);    \
    }                                                                          \
  } while (0)

#define PROTO_DEVICE(T, Device)                                                \
  TEMPLATE_INSTANTIATION(T, data_layout::DATA_PARALLEL, Device);               \
  TEMPLATE_INSTANTIATION(T, data_layout::MODEL_PARALLEL, Device)

#include "lbann/macros/instantiate_device.hpp"

#undef TEMPLATE_INSTANTIATION

    // Check that layer has been constructed
    if (l == nullptr) {
      LBANN_ERROR("Could not construct layer ", name);
    }

    // Set up parallel strategy.
    ParallelStrategy& ps = l->get_parallel_strategy();
    ps.sample_groups = proto_layer.parallel_strategy().sample_groups();
    ps.sample_splits = proto_layer.parallel_strategy().sample_splits();
    ps.height_groups = proto_layer.parallel_strategy().height_groups();
    ps.height_splits = proto_layer.parallel_strategy().height_splits();
    ps.width_groups = proto_layer.parallel_strategy().width_groups();
    ps.width_splits = proto_layer.parallel_strategy().width_splits();
    ps.channel_groups = proto_layer.parallel_strategy().channel_groups();
    ps.channel_splits = proto_layer.parallel_strategy().channel_splits();
    ps.filter_groups = proto_layer.parallel_strategy().filter_groups();
    ps.filter_splits = proto_layer.parallel_strategy().filter_splits();
    ps.replications = proto_layer.parallel_strategy().replications();
    ps.depth_groups = proto_layer.parallel_strategy().depth_groups();
    ps.depth_splits = proto_layer.parallel_strategy().depth_splits();

    // Initialize layer name and check it is unique
    if (!name.empty()) {
      l->set_name(name);
    }
    name = l->get_name();
    if (names_to_layers.count(name) != 0)
      LBANN_ERROR("layer name \"", name, "\" is not unique");
    names_to_layers[name] = l;

    if (proto_layer.freeze()) {
#ifdef LBANN_DEBUG
      if (comm->am_world_master()) {
        std::cout << "freezing " << l->get_name() << std::endl;
      }
#endif
      l->freeze();
    }
    // Add layer to list
    layers.emplace_back(std::move(l));
  }

  // Setup pointers between layers
  setup_parents_and_children(comm, layers, names_to_layers, proto_model);
  setup_hints(layers, names_to_layers, proto_model);

  // Return layer list
  return layers;
}

} // namespace proto
} // namespace lbann
