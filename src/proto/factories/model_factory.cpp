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
#include "lbann/objective_functions/layer_term.hpp"

namespace lbann {
namespace proto {

namespace {

/** Instantiate a model based on prototext. */
model* instantiate_model(lbann_comm* comm,
                         objective_function* obj,
                         const lbann_data::Optimizer& proto_opt,
                         const lbann_data::Model& proto_model) {
  std::stringstream err;

  // Default optimizer
  auto&& opt = construct_optimizer(comm, proto_opt);

  // Construct model
  const auto& type = proto_model.name();
  const auto& mini_batch_size = proto_model.mini_batch_size();
  if (type == "sequential_model" || type == "") {
    return new sequential_model(comm, mini_batch_size, obj, opt);
  }
  if (type == "directed_acyclic_graph_model") {
    return new directed_acyclic_graph_model(comm, mini_batch_size, obj, opt);
  }
  if (type == "siamese_model") {
    const auto& params = proto_model.siamese();
    return new siamese_model(comm,
                             mini_batch_size,
                             obj,
                             opt,
                             params.num_heads());
  }

  // Throw error if model type is not supported
  err << "unknown model type (" << type << ")";
  LBANN_ERROR(err.str());
  return nullptr;

}

void assign_layers_to_objective_function(std::vector<Layer*>& layer_list,
                                         objective_function* obj,
                                         const lbann_data::ObjectiveFunction& proto_obj) {
  std::stringstream err;

  // Construct map from layer names to layers
  std::unordered_map<std::string, Layer*> names_to_layers;
  for (auto&& l : layer_list) {
    const auto& name = l->get_name();
    if (names_to_layers.count(name) > 0) {
      err << "layer name \"" << name << "\" is not unique";
      LBANN_ERROR(err.str());
    }
    names_to_layers[name] = l;
  }

  // Assign layers to layer terms in objective function
  auto&& obj_terms = obj->get_terms();
  int num_layer_terms = 0;
  for (size_t i = 0; i < obj_terms.size(); ++i) {
    auto&& term = dynamic_cast<layer_term*>(obj_terms[i]);
    if (term != nullptr) {
      ++num_layer_terms;
      if (num_layer_terms > proto_obj.layer_term_size()) { continue; }
      const auto& params = proto_obj.layer_term(num_layer_terms-1);
      auto* l = names_to_layers[params.layer()];
      if (l == nullptr) {
        err << "attempted to set objective function layer term "
            << "to correspond to layer \"" << params.layer() << "\", "
            << "but no such layer exists";
        LBANN_ERROR(err.str());
      }
      term->set_layer(*l);
    }
  }

  // Check that layer terms in objective function match prototext
  if (num_layer_terms != proto_obj.layer_term_size()) {
    err << "recieved " << num_layer_terms << " "
        << "objective function layer terms, "
        << "but there are " << proto_obj.layer_term_size() << " "
        << "in the prototext";
    LBANN_ERROR(err.str());
  }
  
}

void assign_layers_to_metrics(std::vector<Layer*>& layer_list,
                              std::vector<metric*>& metric_list,
                              const lbann_data::Model& proto_model) {

  // Construct map from layer names to layers
  std::unordered_map<std::string, Layer*> names_to_layers;
  for (auto&& l : layer_list) {
    const auto& name = l->get_name();
    if (names_to_layers.count(name) > 0) {
      std::stringstream err;
      err << "layer name \"" << name << "\" is not unique";
      LBANN_ERROR(err.str());
    }
    names_to_layers[name] = l;
  }

  // Assign layers to layer metrics
  for (int i=0; i<proto_model.metric_size(); ++i) {
    auto&& m = dynamic_cast<layer_metric*>(metric_list[i]);
    if (m != nullptr) {
      const auto& params = proto_model.metric(i).layer_metric();
      auto* l = names_to_layers[params.layer()];
      if (l == nullptr) {
        std::stringstream err;
        err << "attempted to set layer metric \"" << m->name() << "\" "
            << "to correspond to layer \"" << params.layer() << "\", "
            << "but no such layer exists";
        LBANN_ERROR(err.str());
      }
      m->set_layer(*l);
    }
  }
  
}

/** Setup pointers from layers to weights. */
void assign_weights_to_layers(std::vector<Layer*>& layer_list,
                              std::vector<weights*>& weights_list,
                              const lbann_data::Model& proto_model) {
  std::stringstream err;

  // Construct map from weights names to weights
  std::unordered_map<std::string, weights*> names_to_weights;
  for (auto&& w : weights_list) {
    const auto& name = w->get_name();
    if (names_to_weights.count(name) > 0) {
      err << "weights name \"" << name << "\" is not unique";
      LBANN_ERROR(err.str());
    }
    names_to_weights[name] = w;
  }

  // Find weights assigned to each layer
  for (int i=0; i<proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);
    auto& layer_weights = layer_list[i]->get_weights();
    const bool is_frozen = layer_list[i]->is_frozen();
    for (auto&& name : parse_list<std::string>(proto_layer.weights())) {
      auto&& w = names_to_weights[name];
      if (w == nullptr) {
        err << "could not find weights named \"" << name << "\", "
            << "which are expected by layer " << layer_list[i]->get_name();
        LBANN_ERROR(err.str());
      }
      if (is_frozen) {
        w->freeze();
      } else if (w->is_frozen()) {
        w->unfreeze();
      }
      layer_weights.push_back(w);
    }
  }  

}

} // namespace

model* construct_model(lbann_comm* comm,
                       const std::map<execution_mode, generic_data_reader*>& data_readers,
                       const lbann_data::Optimizer& proto_opt,
                       const lbann_data::Model& proto_model) {

  // Construct layer graph
  auto&& layer_list = construct_layer_graph(comm,
                                            data_readers,
                                            proto_model);

  // Construct objective function
  const auto& proto_obj = proto_model.objective_function();
  auto&& obj = construct_objective_function(proto_obj);
  assign_layers_to_objective_function(layer_list, obj, proto_obj);

  // Construct weights
  std::vector<weights*> weights_list;
  for (int i=0; i<proto_model.weights_size(); i++) {
    weights_list.push_back(construct_weights(comm,
                                             proto_opt,
                                             proto_model.weights(i)));
  }
  assign_weights_to_layers(layer_list, weights_list, proto_model);

  // Construct metrics
  std::vector<metric*> metric_list;
  for (int i=0; i<proto_model.metric_size(); ++i) {
    metric_list.push_back(construct_metric(comm, proto_model.metric(i)));
  }
  assign_layers_to_metrics(layer_list, metric_list, proto_model);

  // Construct callbacks
  std::vector<lbann_callback*> callback_list;
  auto&& summarizer = construct_summarizer(comm, proto_model);
  for (int i=0; i<proto_model.callback_size(); i++) {
    callback_list.push_back(construct_callback(comm,
                                               proto_model.callback(i),
                                               data_readers,
                                               layer_list,
                                               weights_list,
                                               summarizer));
  }

  // Instantiate model
  auto&& m = instantiate_model(comm, obj, proto_opt, proto_model);
  for (auto&& l   : layer_list   ) { m->add_layer(l);     }
  for (auto&& w   : weights_list ) { m->add_weights(w);   }
  for (auto&& met : metric_list  ) { m->add_metric(met);  }
  for (auto&& cb  : callback_list) { m->add_callback(cb); }
  return m;

}

} // namespace proto
} // namespace lbann
