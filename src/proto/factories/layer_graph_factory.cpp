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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/proto/factories.hpp"

namespace lbann {
namespace proto {

namespace {

/** Setup parent/child relationships between layers. */
void setup_parents_and_children(lbann_comm* comm,
                                std::vector<Layer*>& layers,
                                std::unordered_map<std::string, Layer*>& names_to_layers,
                                const lbann_data::Model& proto_model) {
  std::stringstream err;
  for (int i=0; i<proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);
    const auto& parents = parse_list<std::string>(proto_layer.parents());
    const auto& children = parse_list<std::string>(proto_layer.children());
    for (const auto& parent : parents) {
      if (names_to_layers.count(parent) == 0) {
        err << "could not find parent layer " << parent << " "
            << "for layer " << layers[i]->get_name();
        LBANN_ERROR(err.str());
      }
      layers[i]->add_parent_layer(names_to_layers[parent]);
    }
    for (const auto& child : children) {
      if (names_to_layers.count(child) == 0) {
        err << "could not find child layer " << child << " "
            << "for layer " << layers[i]->get_name();
        LBANN_ERROR(err.str());
      }
      layers[i]->add_child_layer(names_to_layers[child]);
    }
  }
}

/** Setup paired input layers for target layers. */
void setup_target_pointers(lbann_comm* comm,
                           std::vector<Layer*>& layers,
                           std::unordered_map<std::string, Layer*>& names_to_layers,
                           const lbann_data::Model& proto_model) {
  std::stringstream err;
  for (int i=0; i<proto_model.layer_size(); ++i) {
    generic_target_layer* target = dynamic_cast<generic_target_layer*>(layers[i]);
    if (target != nullptr) {
      generic_input_layer* input = nullptr;
      const auto& input_name = proto_model.layer(i).target().paired_input_layer();
      if (!input_name.empty()) {
        input = dynamic_cast<generic_input_layer*>(names_to_layers[input_name]);
      } else {
        for (auto&& other : layers) {
          input = dynamic_cast<generic_input_layer*>(other);
          if (input != nullptr) { break; }
        }
      }
      if (input == nullptr) {
        err << "could not find input layer " << input_name << " "
            << "to pair with target layer " << target->get_name();
        LBANN_ERROR(err.str());
      }
      if (input->is_for_regression() != target->is_for_regression()) {
        err << "target layer " << target->get_name() << " "
            << "and its paired input layer " << input->get_name()
            << "are not consistent regarding regression/classification";
        LBANN_ERROR(err.str());
      }
      target->set_paired_input_layer(input);
    }
  }
}

/** Setup original layers for reconstruction layers. */
void setup_reconstruction_pointers(lbann_comm* comm,
                                   std::vector<Layer*>& layers,
                                   std::unordered_map<std::string, Layer*>& names_to_layers,
                                   const lbann_data::Model& proto_model) {
  std::stringstream err;
  for (int i=0; i<proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);
    Layer* l = layers[i];
    if (proto_layer.has_reconstruction()) {
      Layer* original = nullptr;
      const auto& original_name = proto_layer.reconstruction().original_layer();
      if (!original_name.empty()) {
        original = names_to_layers[original_name];
      } else {
        for (auto&& other : layers) {
          original = dynamic_cast<generic_input_layer*>(other);
          if (original != nullptr) { break; }
        }
      }
      if (original == nullptr) {
        err << "could not find original layer " << original_name << " "
            << "for reconstruction layer " << l->get_name();
        LBANN_ERROR(err.str());
      }
      auto&& recon_dp_cpu = dynamic_cast<reconstruction_layer<data_layout::DATA_PARALLEL, El::Device::CPU>*>(l);
      auto&& recon_mp_cpu = dynamic_cast<reconstruction_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>*>(l);
#ifdef LBANN_HAS_GPU
      auto&& recon_dp_gpu = dynamic_cast<reconstruction_layer<data_layout::DATA_PARALLEL, El::Device::GPU>*>(l);
      auto&& recon_mp_gpu = dynamic_cast<reconstruction_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>*>(l);
#endif // LBANN_HAS_GPU
      if (recon_dp_cpu != nullptr) { recon_dp_cpu->set_original_layer(original); }
      if (recon_mp_cpu != nullptr) { recon_mp_cpu->set_original_layer(original); }
#ifdef LBANN_HAS_GPU
      if (recon_dp_gpu != nullptr) { recon_dp_gpu->set_original_layer(original); }
      if (recon_mp_gpu != nullptr) { recon_mp_gpu->set_original_layer(original); }
#endif // LBANN_HAS_GPU
    }
  }
}

/** Setup paired pooling layers for unpooling layers. */
void setup_unpooling_pointers(lbann_comm* comm,
                              std::vector<Layer*>& layers,
                              std::unordered_map<std::string, Layer*>& names_to_layers,
                              const lbann_data::Model& proto_model) {
  std::stringstream err;
  for (int i=0; i<proto_model.layer_size(); ++i) {
    {
      unpooling_layer<data_layout::DATA_PARALLEL, El::Device::CPU>* unpool
        = dynamic_cast<unpooling_layer<data_layout::DATA_PARALLEL, El::Device::CPU>*>(layers[i]);
      if (unpool != nullptr) {
        const auto& pool_name = proto_model.layer(i).unpooling().pooling_layer();
        pooling_layer<data_layout::DATA_PARALLEL, El::Device::CPU>* pool
          = dynamic_cast<pooling_layer<data_layout::DATA_PARALLEL, El::Device::CPU>*>(names_to_layers[pool_name]);
        if (pool == nullptr) {
          err << "could not find pooling layer " << pool_name << " "
              << "to pair with unpooling layer " << unpool->get_name();
          LBANN_ERROR(err.str());
        }
        unpool->set_pooling_layer(pool);
      }
    }
#ifdef LBANN_HAS_GPU
    {
      unpooling_layer<data_layout::DATA_PARALLEL, El::Device::GPU>* unpool
        = dynamic_cast<unpooling_layer<data_layout::DATA_PARALLEL, El::Device::GPU>*>(layers[i]);
      if (unpool != nullptr) {
        const auto& pool_name = proto_model.layer(i).unpooling().pooling_layer();
        pooling_layer<data_layout::DATA_PARALLEL, El::Device::GPU>* pool
          = dynamic_cast<pooling_layer<data_layout::DATA_PARALLEL, El::Device::GPU>*>(names_to_layers[pool_name]);
        if (pool == nullptr) {
          err << "could not find pooling layer " << pool_name << " "
              << "to pair with unpooling layer " << unpool->get_name();
          LBANN_ERROR(err.str());
        }
        unpool->set_pooling_layer(pool);
      }
    }
#endif // LBANN_HAS_GPU
  }
}

} // namespace

std::vector<Layer*> construct_layer_graph(lbann_comm* comm,
                                          std::map<execution_mode, generic_data_reader *>& data_readers,
                                          cudnn::cudnn_manager* cudnn,
                                          const lbann_data::Model& proto_model) {
  std::stringstream err;

  // List of layers
  std::vector<Layer*> layers;

  // Map from names to layer pointers
  std::unordered_map<std::string, Layer*> names_to_layers;

  // Create each layer in prototext
  for (int i=0; i<proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);

    // Check that layer name is valid
    auto name = proto_layer.name();
    const auto& parsed_name = parse_list<std::string>(name);
    if (!name.empty()) {
      if (parsed_name.empty() || parsed_name.front() != name) {
        err << "weights name \"" << name << "\" is invalid since it "
            << "contains whitespace";
        LBANN_ERROR(err.str());
      }
      if (names_to_layers.count(name) != 0) {
        err << "layer name \"" << name << "\" is not unique";
        LBANN_ERROR(err.str());
      }
    }

    // Get parameters from prototext
    const auto& layout_str = proto_layer.data_layout();
    data_layout layout = data_layout::invalid;
    if (layout_str.empty())             { layout = data_layout::DATA_PARALLEL; }
    if (layout_str == "data_parallel")  { layout = data_layout::DATA_PARALLEL; }
    if (layout_str == "model_parallel") { layout = data_layout::MODEL_PARALLEL; }
    const auto& num_parallel_readers = proto_model.num_parallel_readers();

    const auto& device_allocation_str = proto_layer.device_allocation();
    El::Device default_device_allocation = El::Device::CPU;
#ifdef LBANN_HAS_GPU
    if (cudnn != nullptr) { default_device_allocation = El::Device::GPU; }
#endif // LBANN_HAS_GPU
    El::Device device_allocation = default_device_allocation;
    if (device_allocation_str.empty()
        && (proto_layer.has_input() || proto_layer.has_target()
            || proto_layer.has_softmax())) {
      // Input and Target layers are not allowed on the GPUs force the
      // default to be the CPU
      device_allocation = El::Device::CPU;
    }
    if (device_allocation_str == "cpu") { device_allocation = El::Device::CPU; }
#ifdef LBANN_HAS_GPU
    if (device_allocation_str == "gpu") { device_allocation = El::Device::GPU; }
#endif // LBANN_HAS_GPU

    // Construct layer
    Layer* l = nullptr;
    switch (layout) {
    case data_layout::DATA_PARALLEL:
      switch (device_allocation) {
      case El::Device::CPU:
        l = construct_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
              comm,
              data_readers,
              num_parallel_readers,
              nullptr,
              proto_layer
            );
        break;
#ifdef LBANN_HAS_GPU
      case El::Device::GPU:
        l = construct_layer<data_layout::DATA_PARALLEL, El::Device::GPU>(
              comm,
              data_readers,
              num_parallel_readers,
              cudnn,
              proto_layer
            );
        break;
#endif // LBANN_HAS_GPU
      default:
        err << "layer " << name << " has an invalid device allocation "
            << "(" << device_allocation_str << ")";
        LBANN_ERROR(err.str());
      }
      break;
    case data_layout::MODEL_PARALLEL:
      switch (device_allocation) {
      case El::Device::CPU:
        l = construct_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>(
              comm,
              data_readers,
              num_parallel_readers,
              nullptr,
              proto_layer
            );
        break;
#ifdef LBANN_HAS_GPU
      case El::Device::GPU:
        l = construct_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>(
              comm,
              data_readers,
              num_parallel_readers,
              cudnn,
              proto_layer
            );
        break;
#endif // LBANN_HAS_GPU
      default:
        err << "layer " << name << " has an invalid device allocation "
            << "(" << device_allocation_str << ")";
        LBANN_ERROR(err.str());
      }
      break;
    case data_layout::invalid:
    default:
      err << "layer " << name << " has an invalid data layout "
          << "(" << layout_str << ")";
      LBANN_ERROR(err.str());
    }

    // Check that layer has been constructed
    if (l == nullptr) {
      err << "could not construct layer " << name;
      LBANN_ERROR(err.str());
    }

    // Initialize layer name and check it is unique
    if (!name.empty()) {
      l->set_name(name);
    }
    name = l->get_name();
    if (names_to_layers.count(name) != 0) {
      err << "layer name \"" << name << "\" is not unique";
      LBANN_ERROR(err.str());
    }
    names_to_layers[name] = l;

    if (proto_layer.freeze()) {
      if (comm->am_world_master()) {
        std::cout << "freezing " << l->get_name() << std::endl;
      }
      l->freeze();
    }
    // Add layer to list
    layers.push_back(l);

  }

  // Setup pointers between layers
  setup_parents_and_children(comm, layers, names_to_layers, proto_model);
  setup_target_pointers(comm, layers, names_to_layers, proto_model);
  setup_reconstruction_pointers(comm, layers, names_to_layers, proto_model);
  setup_unpooling_pointers(comm, layers, names_to_layers, proto_model);

  // Return layer list
  return layers;

}

} // namespace proto
} // namespace lbann
