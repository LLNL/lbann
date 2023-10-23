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

#include "lbann/models/model.hpp"

#include "lbann/base.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/callbacks/save_model.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/layers/io/input_layer.hpp"
#include "lbann/layers/transform/dummy.hpp"
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/layers/transform/split.hpp"
#include "lbann/metrics/layer_metric.hpp"
#include "lbann/objective_functions/layer_term.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/amp.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/graph.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/onnx_utils.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/summary_impl.hpp"

#include "lbann/proto/model.pb.h"
#include "lbann/proto/optimizers.pb.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

namespace lbann {

// =============================================
// Life cycle functions
// =============================================

model::model(lbann_comm* comm,
             std::unique_ptr<objective_function> obj_fn,
             std::unique_ptr<lbann_data::Optimizer> default_optimizer_msg)
  : m_execution_context(nullptr),
    m_comm(comm),
    m_default_optimizer_msg(std::move(default_optimizer_msg))
{

  m_objective_function = std::move(obj_fn);
  // Default model name
  static El::Int num_models = 0;
  m_name = "model" + std::to_string(num_models);
  num_models++;
}

model::model() : model(&utils::get_current_comm(), nullptr, nullptr) {}

model::model(const model& other)
  : m_execution_context(other.m_execution_context),
    m_comm(other.m_comm),
    m_name(other.m_name),
    m_model_is_setup(false)
{

  // Deep copies
  m_default_optimizer_msg =
    (other.m_default_optimizer_msg
       ? std::make_unique<lbann_data::Optimizer>(*other.m_default_optimizer_msg)
       : nullptr);
  m_objective_function =
    (other.m_objective_function
       ? std::make_unique<objective_function>(*other.m_objective_function)
       : nullptr);
  for (const auto& m : other.m_metrics) {
    m_metrics.emplace_back(m ? m->copy() : nullptr);
  }
  for (const auto& cb : other.m_callbacks) {
    m_callbacks.emplace_back(cb ? cb->copy() : nullptr);
  }

  // Copy layers
  std::unordered_map<Layer*, ViewingLayerPtr> layer_map;
  m_layers.reserve(other.m_layers.size());
  for (const auto& other_layer : other.m_layers) {
    if (other_layer == nullptr) {
      LBANN_ERROR("model \"",
                  other.get_name(),
                  "\" ",
                  "has a null pointer in its list of layers");
    }
    m_layers.emplace_back(other_layer->copy());
    m_layers.back()->set_model(this);
    layer_map[other_layer.get()] = m_layers.back();
  }

  // Copy weights
  std::unordered_map<weights*, ViewingWeightsPtr> weights_map;
  m_weights.reserve(other.m_weights.size());
  for (const auto& other_weights : other.m_weights) {
    if (other_weights == nullptr) {
      LBANN_ERROR("model \"",
                  other.get_name(),
                  "\" ",
                  "has a null pointer in its list of weights");
    }
    m_weights.emplace_back(std::make_shared<data_type_weights<DataType>>(
      dynamic_cast<data_type_weights<DataType>&>(*other_weights)));
    weights_map[other_weights.get()] = m_weights.back();
  }

  // Fix pointers
  remap_pointers(layer_map, weights_map);
}

model& model::operator=(const model& other)
{

  // Delete objects
  // if (m_execution_context  != nullptr) { delete m_execution_context; } ///
  // @todo BVE FIXME what do we do with smart pointers here

  // Shallow copies
  m_comm = other.m_comm;
  m_name = other.m_name;
  m_model_is_setup = false;

  // Deep copies
  m_execution_context = other.m_execution_context;
  m_objective_function =
    (other.m_objective_function
       ? std::make_unique<objective_function>(*other.m_objective_function)
       : nullptr);
  m_metrics.clear();
  for (const auto& m : other.m_metrics) {
    m_metrics.emplace_back(m ? m->copy() : nullptr);
  }
  m_callbacks.clear();
  for (const auto& cb : other.m_callbacks) {
    m_callbacks.emplace_back(cb ? cb->copy() : nullptr);
  }

  // Copy layers
  std::unordered_map<Layer*, ViewingLayerPtr> layer_map;
  m_layers.clear();
  m_layers.reserve(other.m_layers.size());
  for (const auto& other_layer : other.m_layers) {
    if (other_layer == nullptr) {
      LBANN_ERROR("model \"",
                  other.get_name(),
                  "\" ",
                  "has a null pointer in its list of layers");
    }
    m_layers.emplace_back(other_layer->copy());
    m_layers.back()->set_model(this);
    layer_map[other_layer.get()] = m_layers.back();
  }

  // Copy weights
  std::unordered_map<weights*, ViewingWeightsPtr> weights_map;
  m_weights.clear();
  m_weights.reserve(other.m_weights.size());
  for (const auto& other_weights : other.m_weights) {
    if (other_weights == nullptr) {
      LBANN_ERROR("model \"",
                  other.get_name(),
                  "\" ",
                  "has a null pointer in its list of weights");
    }
    m_weights.emplace_back(std::make_unique<data_type_weights<DataType>>(
      dynamic_cast<data_type_weights<DataType>&>(*other_weights)));
    weights_map[other_weights.get()] = m_weights.back();
  }

  // Fix pointers
  remap_pointers(layer_map, weights_map);

  return *this;
}

template <class Archive>
void model::serialize(Archive& ar)
{
  ar(
    // CEREAL_NVP(m_execution_context),
    CEREAL_NVP(m_name),
    // CEREAL_NVP(m_comm),
    CEREAL_NVP(m_layers),
    CEREAL_NVP(m_weights),
    // CEREAL_NVP(m_default_optimizer_msg),
    CEREAL_NVP(m_objective_function),
    CEREAL_NVP(m_metrics),
    // CEREAL_NVP(m_callbacks),
    // CEREAL_NVP(m_model_is_setup),
    CEREAL_NVP(m_max_mini_batch_size),
    // CEREAL_NVP(m_current_mini_batch_size),
    CEREAL_NVP(m_max_mini_batch_size),
    CEREAL_NVP(m_amp_enabled),
    CEREAL_NVP(m_amp_scale_factor),
    CEREAL_NVP(m_amp_growth_factor),
    CEREAL_NVP(m_amp_backoff_factor),
    CEREAL_NVP(m_amp_growth_interval),
    CEREAL_NVP(m_amp_cur_steps));

  ar.serializeDeferments();
  if constexpr (utils::IsInputArchive<Archive>)
    m_model_is_setup = false;
}

// =============================================
// Access functions
// =============================================

void model::set_name(std::string name)
{
  if (name.empty()) {
    LBANN_ERROR("attempted to rename model \"",
                get_name(),
                "\" with empty string");
  }
  m_name = std::move(name);
}

description model::get_description() const
{

  // Construct description object
  description desc(get_name());

  // AMP details.
  if (is_amp_enabled()) {
    description amp_desc("Automatic mixed precision: Enabled");
    desc.add(amp_desc);
  }

  // Layer topology
  description layer_topology_desc("Layer topology:");
  for (El::Int k = 0; k < get_num_layers(); ++k) {
    const auto& l = get_layer(k);
    std::ostringstream oss;
    oss << l.get_name() << " (" << l.get_type() << "): {";
    const auto& parents = l.get_parent_layers();
    const auto& children = l.get_child_layers();
    for (size_t i = 0; i < parents.size(); ++i) {
      oss << (i > 0 ? ", " : "");
      if (parents[i] == nullptr) {
        oss << "unknown layer";
      }
      else {
        oss << parents[i]->get_name() << " (";
        const auto& dims = l.get_input_dims(i);
        for (size_t j = 0; j < dims.size(); ++j) {
          oss << (j > 0 ? "x" : "") << dims[j];
        }
        oss << ")";
      }
    }
    oss << "} -> {";
    for (size_t i = 0; i < children.size(); ++i) {
      oss << (i > 0 ? ", " : "");
      if (children[i] == nullptr) {
        oss << "unknown layer";
      }
      else {
        oss << children[i]->get_name() << " (";
        const auto& dims = l.get_output_dims(i);
        for (size_t j = 0; j < dims.size(); ++j) {
          oss << (j > 0 ? "x" : "") << dims[j];
        }
        oss << ")";
      }
    }
    oss << "}";
    layer_topology_desc.add(oss.str());
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
  for (const auto& w : m_weights) {
    if (w == nullptr) {
      weights_desc.add("unknown weights");
    }
    else {
      weights_desc.add(w->get_description());
    }
  }
  desc.add(std::string{});
  desc.add(weights_desc);

  // Callbacks
  description callback_desc("Callbacks:");
  for (const auto& cb : m_callbacks) {
    callback_desc.add(cb->get_description());
  }
  desc.add(std::string{});
  desc.add(callback_desc);

  /// @todo Descriptions for objective function, metrics

  // Result
  return desc;
}

std::vector<metric*> model::get_metrics()
{
  std::vector<metric*> ptrs;
  for (const auto& ptr : m_metrics) {
    ptrs.push_back(ptr.get());
  }
  return ptrs;
}

std::vector<metric const*> model::get_metrics() const
{
  std::vector<metric const*> ptrs;
  for (const auto& ptr : m_metrics) {
    ptrs.push_back(ptr.get());
  }
  return ptrs;
}

El::Int model::get_num_layers() const noexcept { return m_layers.size(); }
Layer& model::get_layer(El::Int pos)
{
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<Layer&>(static_cast<const model&>(*this).get_layer(pos));
}

Layer const& model::get_layer(El::Int pos) const
{
  if (pos < 0 || pos >= get_num_layers()) {
    LBANN_ERROR("could not access layer in model \"",
                get_name(),
                "\" (requested index ",
                pos,
                ", but there are ",
                get_num_layers(),
                " layers)");
  }
  else if (m_layers[pos] == nullptr) {
    LBANN_ERROR("model \"",
                get_name(),
                "\" has a null pointer in its layer list");
  }
  return *m_layers[pos];
}

std::vector<Layer*> model::get_layers()
{
  std::vector<Layer*> layer_list;
  layer_list.reserve(m_layers.size());
  for (const auto& ptr : m_layers) {
    layer_list.push_back(ptr.get());
  }
  return layer_list;
}

std::vector<Layer const*> model::get_layers() const
{
  std::vector<Layer const*> layer_list;
  layer_list.reserve(m_layers.size());
  for (const auto& ptr : m_layers) {
    layer_list.push_back(ptr.get());
  }
  return layer_list;
}

std::vector<weights*> model::get_weights()
{
  std::vector<weights*> weights_list;
  for (const auto& w : m_weights) {
    weights_list.push_back(w.get());
  }
  return weights_list;
}

std::vector<weights const*> model::get_weights() const
{
  std::vector<weights const*> weights_list;
  for (const auto& w : m_weights) {
    weights_list.push_back(w.get());
  }
  return weights_list;
}

std::vector<ViewingWeightsPtr> model::get_weights_pointers() const
{
  std::vector<ViewingWeightsPtr> ptrs;
  for (const auto& w : m_weights) {
    ptrs.emplace_back(w);
  }
  return ptrs;
}

#ifdef LBANN_HAS_ONNX
void model::serialize_to_onnx(onnx::ModelProto& mp)
{
  mp.set_ir_version(7);
  auto* opset = mp.add_opset_import();
  // The empty string ("") domain indicates the operators defined
  // as part of the ONNX specification; other domains correspond
  // to operator sets of other vendors (e.g., they can be used to
  // provide vendor-specific extensions to ONNX)
  opset->set_domain("");
  opset->set_version(14);

  mp.set_producer_name("LBANN");
  mp.set_producer_version(LBANN_MAKE_STR(LBANN_VERSION));
  mp.set_domain("lbann/LLNL/com.github");
  mp.set_model_version(1);
  mp.set_doc_string("Livermore Big Artificial Neural Network");

  // graph info
  auto* gp = mp.mutable_graph();
  gp->set_name(this->get_name());

  for (auto const* weights : this->get_weights()) {
    weights->fill_onnx_node(*gp);
  }

  auto const layers = this->get_layers();
  for (auto const* layer : layers) {
    layer->fill_onnx_node(*gp);
  }

  gp->set_doc_string(this->get_name());
}
#endif // LBANN_HAS_ONNX

// =============================================
// Model specification
// =============================================

void model::add_layer(OwningLayerPtr&& ptr)
{

  // Check for null pointer
  if (ptr == nullptr) {
    LBANN_ERROR("attempted to add a null pointer as layer to ",
                "model \"",
                get_name(),
                "\"");
  }

  // Check that the new layer name is unique
  // Note: Adding layers is O(n^2), but this is unlikely to be a
  // bottleneck. If it is, consider maintaining a hash table
  // containing all layer names (and properly updating it during
  // copies and pointer remaps).
  const auto& name = ptr->get_name();
  for (const auto& l : m_layers) {
    if (l->get_name() == name) {
      LBANN_ERROR("attempted to add layer \"",
                  name,
                  "\" to ",
                  "model \"",
                  get_name(),
                  "\", ",
                  "but the model already contains a layer with that name");
    }
  }

  // Add layer to model
  m_layers.emplace_back(std::move(ptr));
  m_layers.back()->set_model(this);
}

void model::add_weights(OwningWeightsPtr&& ptr)
{

  // Check for null pointer
  if (ptr == nullptr) {
    LBANN_ERROR("attempted to add a null pointer as weights to ",
                "model \"",
                get_name(),
                "\"");
  }

  // Check that the new weights name is unique
  // Note: Adding weights is O(n^2), but this is unlikely to be a
  // bottleneck. If it is, consider maintaining a hash table
  // containing all weights names (and properly updating it during
  // copies and pointer remaps).
  const auto& name = ptr->get_name();
  for (const auto& w : m_weights) {
    if (w->get_name() == name) {
      LBANN_ERROR("attempted to add weights \"",
                  name,
                  "\" to ",
                  "model \"",
                  get_name(),
                  "\", ",
                  "but the model already contains weights with that name");
    }
  }

  // Add weights to model
  m_weights.emplace_back(std::move(ptr));
}

void model::remove_weights(std::string const& removable_weight_name)
{
  auto const removable_weight_iterator =
    std::find_if(cbegin(m_weights),
                 cend(m_weights),
                 [&removable_weight_name](auto const& w) {
                   return w->get_name() == removable_weight_name;
                 });
  if (removable_weight_iterator != cend(m_weights))
    m_weights.erase(removable_weight_iterator);
  else
    LBANN_ERROR("Attempted to remove weight",
                " \"",
                removable_weight_name,
                "\", ",
                "but no such weight exists");
}

void model::add_callback(std::shared_ptr<callback_base> cb)
{
  if (cb == nullptr) {
    LBANN_ERROR("attempted to add a null pointer as callback to ",
                "model \"",
                get_name(),
                "\"");
  }
  m_callbacks.emplace_back(std::move(cb));
}

void model::add_metric(std::unique_ptr<metric> m)
{
  if (m == nullptr) {
    LBANN_ERROR("attempted to add a null pointer as a metric to ",
                "model \"",
                get_name(),
                "\"");
  }
  m_metrics.emplace_back(std::move(m));
}

void model::copy_trained_weights_from(std::vector<weights*>& new_weights)
{
  if (new_weights.empty()) {
    if (m_comm->am_world_master())
      std::cout << "No trained weights to copy " << std::endl;
    return;
  }
  for (size_t i = 0; i < new_weights.size(); ++i) {
    for (size_t j = 0; j < m_weights.size(); ++j) {
      // copy only trained weights (that is unfrozen layer)
      if (m_weights[j]->get_name() == new_weights[i]->get_name() &&
          !new_weights[i]->is_frozen()) {
#ifdef LBANN_DEBUG
        if (m_comm->am_world_master())
          std::cout << " Replacing " << m_weights[j]->get_name() << " with "
                    << new_weights[i]->get_name() << std::endl;
#endif
        dynamic_cast<data_type_weights<DataType>&>(*m_weights[j].get())
          .set_values(
            dynamic_cast<data_type_weights<DataType> const&>(*new_weights[i])
              .get_values_sharded());
      }
    }
  }
}

void model::swap_layers(model& other) { std::swap(m_layers, other.m_layers); }

void model::swap_weights(model& other)
{
  std::swap(m_weights, other.m_weights);
}

void model::swap_metrics(model& other)
{
  std::swap(m_metrics, other.m_metrics);
}

void model::swap_objective_function(model& other)
{
  std::swap(m_objective_function, other.m_objective_function);
}

void model::reorder_layers(const std::vector<El::Int>& gather_indices)
{
  // Check that gather indices are in valid range
  const auto& num_layers = get_num_layers();
  if (std::any_of(gather_indices.begin(),
                  gather_indices.end(),
                  [num_layers](El::Int index) {
                    return index < 0 || index >= num_layers;
                  })) {
    LBANN_ERROR("Attempted to reorder layer list for model \"",
                get_name(),
                "\" with invalid gather index");
  }

  // Reorder layers
  std::vector<OwningLayerPtr> reordered_layers(gather_indices.size());
  for (size_t i = 0; i < gather_indices.size(); ++i) {
    reordered_layers[i] = std::move(m_layers[gather_indices[i]]);
  }
  m_layers = std::move(reordered_layers);

  // Check that layer list has no null pointers
  for (const auto& l : m_layers) {
    if (l == nullptr) {
      LBANN_ERROR("Found a null pointer in the layer list for model \"",
                  get_name(),
                  "\" after reordering");
    }
  }
}

void model::remap_pointers(
  const std::unordered_map<Layer*, ViewingLayerPtr>& layer_map,
  const std::unordered_map<weights*, ViewingWeightsPtr>& weights_map)
{

  // Fix pointers in objective function
  if (m_objective_function != nullptr) {
    auto layer_pointers = m_objective_function->get_layer_pointers();
    for (auto& ptr : layer_pointers) {
      auto* raw_ptr = ptr.lock().get();
      if (layer_map.count(raw_ptr) > 0) {
        ptr = layer_map.at(raw_ptr);
      }
    }
    m_objective_function->set_layer_pointers(layer_pointers);
    auto weights_pointers = m_objective_function->get_weights_pointers();
    for (auto& ptr : weights_pointers) {
      auto* raw_ptr = ptr.lock().get();
      if (weights_map.count(raw_ptr) > 0) {
        ptr = weights_map.at(raw_ptr);
      }
    }
    m_objective_function->set_weights_pointers(weights_pointers);
  }

  // Fix pointers in metrics
  for (const auto& m : m_metrics) {
    auto layer_pointers = m->get_layer_pointers();
    for (auto& ptr : layer_pointers) {
      auto* raw_ptr = ptr.lock().get();
      if (layer_map.count(raw_ptr) > 0) {
        ptr = layer_map.at(raw_ptr);
      }
    }
    m->set_layer_pointers(layer_pointers);
  }

  // Fix pointers in layers
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    auto layer_pointers = l.get_layer_pointers();
    auto weights_pointers = l.get_weights_pointers();
    for (auto& ptr : layer_pointers) {
      auto* raw_ptr = ptr.lock().get();
      if (layer_map.count(raw_ptr) > 0) {
        ptr = layer_map.at(raw_ptr);
      }
    }
    for (auto& ptr : weights_pointers) {
      auto* raw_ptr = ptr.lock().get();
      if (weights_map.count(raw_ptr) > 0) {
        ptr = weights_map.at(raw_ptr);
      }
    }
    l.set_layer_pointers(layer_pointers);
    l.set_weights_pointers(weights_pointers);
  }
}

// =============================================
// Setup
// =============================================

void model::setup(size_t max_mini_batch_size,
                  const std::vector<El::Grid*>& grids_,
                  bool force)
{

  // Bail out if the model is already setup
  if (m_model_is_setup && !force) {
    return;
  }

  for (const auto& cb : m_callbacks) {
    cb->on_setup_begin(this);
  }

  check_subgraph_parallelism();

  // Setup layers

  setup_layer_topology();
  setup_layer_execution_order();
  setup_layer_grid_tags(grids_);

  if (this->is_subgraph_parallelism_enabled()) {
    for (auto& layer : this->get_layers())
      layer->set_num_spliting_groups(grids_.size() - 1);

    setup_subgrid_layers_run_condition();
    setup_subcommunicators(grids_);
  }

  // Because it is used as a default for layers with no inputs
  // initialize the current mini-batch size before setting up the
  // layers
  m_current_mini_batch_size = max_mini_batch_size;
  setup_layers(max_mini_batch_size, grids_);

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

  m_max_mini_batch_size = max_mini_batch_size;

#ifdef LBANN_HAS_DISTCONV
  setup_distconv();
#endif

  // Callback hooks at end of setup
  do_setup_end_cbs();

  m_model_is_setup = true;
}

void model::setup_layer_topology()
{

  // Check that layer list is valid
  // Note: Throws an exception if the layer list contains two layers
  // with the same name or if a layer has a pointer to a layer in a
  // different model.
  std::unordered_set<Layer*> layer_set;
  std::unordered_set<std::string> layer_names;
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    if (layer_names.count(l.get_name()) > 0) {
      LBANN_ERROR("model \"",
                  get_name(),
                  "\" "
                  "has multiple layers named \"",
                  l.get_name(),
                  "\"");
    }
    layer_set.insert(&l);
    layer_names.insert(l.get_name());
  }
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);
    for (const auto& ptr : l.get_layer_pointers()) {
      auto* raw_ptr = ptr.lock().get();
      if (raw_ptr != nullptr && layer_set.count(raw_ptr) == 0) {
        auto message = build_string(l.get_type(),
                                    " layer \"",
                                    l.get_name(),
                                    "\" ",
                                    "(in model \"",
                                    get_name(),
                                    "\") has a pointer to ",
                                    raw_ptr->get_type(),
                                    " layer \"",
                                    raw_ptr->get_name(),
                                    "\" ");
        if (raw_ptr->get_model() == nullptr) {
          message += "(not in a model)";
        }
        else {
          message += build_string("(in model \"",
                                  raw_ptr->get_model()->get_name(),
                                  "\")");
        }
        LBANN_ERROR(message);
      }
    }
  }

  // Make sure parent/child relationships are reciprocated
  for (auto& l : m_layers) {
    for (int i = 0; i < l->get_num_parents(); ++i) {
      const_cast<Layer&>(l->get_parent_layer(i)).add_child_layer(l);
    }
    for (int i = 0; i < l->get_num_children(); ++i) {
      const_cast<Layer&>(l->get_child_layer(i)).add_parent_layer(l);
    }
  }

  // Add utility layers
  add_evaluation_layers(layer_set, layer_names);
  add_dummy_layers(layer_names);
  add_split_layers(layer_names);
}

void model::check_subgraph_parallelism()
{
  // Enables sub-graph parallelism if a layer has a sub-graph tag greater than
  // zero
  const auto& layers = this->get_layers();
  for (auto const& l : layers) {
    if (l->get_grid_tag() > 0) {
      this->enable_subgraph_parallelism();
      break;
    }
  }
}

void model::setup_subgrid_layers_run_condition()
{
  const auto& layers = this->get_layers();
  for (auto const& l : layers) {
    // Special case when split/slice and concatenate/sum need sub-graph
    // parallelism execution
    if ((l->get_type() == "split" || l->get_type() == "slice") &&
        (l->get_child_layer(0).get_grid_tag() != l->get_grid_tag())) {
      l->set_subgraph_parallelism_execution();
    }
    if ((l->get_type() == "concatenate" || l->get_type() == "sum") &&
        (l->get_parent_layer(0).get_grid_tag() != l->get_grid_tag())) {
      l->set_subgraph_parallelism_execution();
    }
  }
}

void model::get_subgrids_order(std::vector<int>& ranks_order, int num_branches)
{
  // function to get ranks in order according to the topology
  // more topology aware designs can be defined here
  // currently there is only one design
  int size_grid = ranks_order.size();

  std::vector<int> temp_ranks(ranks_order.begin(), ranks_order.end());
  std::sort(temp_ranks.begin(), temp_ranks.end());
  int rank = 0;

  // parent grid has more ranks than subgrids but less than total number of
  // ranks
  bool cond_parent_have_more_resources =
    (size_grid / (this->get_num_resources_branch_layers() / num_branches)) > 1;

  // No need to order when parent's total resources are less than subgrids
  // Topology aware design for this case is implemented in get_input_resources
  // and merge_resources layer
  if (this->get_subgrid_topology() == true &&
      (this->get_num_resources_non_branch_layers() ==
         this->get_num_resources_branch_layers() ||
       cond_parent_have_more_resources)) {

    int size_branch = size_grid / num_branches;
    for (int i = 0; i < size_branch; ++i) {
      for (int j = 0; j < num_branches; ++j) {
        ranks_order[j * size_branch + i] = temp_ranks[rank];
        rank++;
      }
    }
  }
  else {
    for (El::Int i = 0; i < size_grid; ++i) {
      ranks_order[i] = temp_ranks[rank];
      rank++;
    }
  }
}

void model::setup_subcommunicators(const std::vector<El::Grid*>& fngrids)
{
  // Because of this optimization we cannot have dynamic number of sub-graph in
  // a model For example: In a model, module1 cannnot different number of
  // sub-graphs than module2
  std::string one_index = "1";

  const auto& layers = this->get_layers();
  const El::Int num_layers = layers.size();

  for (El::Int node = 0; node < num_layers; ++node) {
    if ((layers[node]->get_type() == "slice" ||
         layers[node]->get_type() == "split" ||
         layers[node]->get_type() == "concatenate" ||
         layers[node]->get_type() == "sum") &&
        layers[node]->subgraph_parallelism_execution()) {
      if (subCommunicatorsSubgrids.find(one_index) !=
          subCommunicatorsSubgrids.end()) {
        layers[node]->reset_inter_subgrid_vc_comm(
          subCommunicatorsSubgrids[one_index]);
      }
      else {
        subCommunicatorsSubgrids[one_index] = std::make_shared<El::mpi::Comm>();
        const auto& childs = layers[node]->get_child_layers();

        int indexSubgrid = -1;
        for (int child = 0; child < layers[node]->get_num_children(); ++child) {
          if (fngrids[childs[child]->get_grid_tag()]->InGrid())

          {
            indexSubgrid = child;
          }
        }

        const int child_tag = childs[indexSubgrid]->get_grid_tag();
        const int layer_tag = layers[node]->get_grid_tag();

        const int posInSubGrid = fngrids[child_tag]->VCRank();
        const int posInGrid = fngrids[layer_tag]->ViewingRank();
        El::mpi::Split(layers[node]->get_comm()->get_trainer_comm(),
                       posInSubGrid,
                       posInGrid,
                       *subCommunicatorsSubgrids[one_index]);

        layers[node]->reset_inter_subgrid_vc_comm(
          subCommunicatorsSubgrids[one_index]);
      }
    }

    if (layers[node]->get_type() == "cross_grid_sum" ||
        layers[node]->get_type() == "cross_grid_sum_slice") {
      layers[node]->reset_inter_subgrid_vc_comm(
        subCommunicatorsSubgrids[one_index]);
    }
  }
}

void model::setup_layer_execution_order()
{
  // Construct layer graph
  // Note: Each layer depends on its parent layers and its hint layer.
  const auto& layers = this->get_layers();
  const El::Int num_layers = layers.size();
  std::set<El::Int> nodes;
  std::map<El::Int, std::set<El::Int>> edges;
  std::unordered_map<const Layer*, El::Int> layer_indices;
  for (El::Int node = 0; node < num_layers; ++node) {
    nodes.insert(node);
    layer_indices[layers[node]] = node;
  }
  for (El::Int node = 0; node < num_layers; ++node) {
    const auto& l = layers[node];
    for (const auto& child : l->get_child_layers()) {
      edges[node].insert(layer_indices[child]);
    }
    if (l->get_hint_layer() != nullptr) {
      edges[layer_indices[l->get_hint_layer()]].insert(node);
    }
  }

  // Topologically sort layers
  const auto& sorted_order = graph::topological_sort(nodes, edges);
  reorder_layers(sorted_order);
  ensure_input_layers_first();
}

void model::setup_layer_grid_tags(const std::vector<El::Grid*>& fngrids)
{
  for (auto& layer : this->get_layers()) {
    // Choose process grid to distribute matrices over
    int tag = layer->get_grid_tag();
    if (tag < 0) {
      // Use tag from parent layers if they are all the same. Otherwise
      // use tag 0.
      for (int i = 0; i < layer->get_num_parents(); ++i) {
        auto parent_tag = layer->get_parent_layer(i).get_grid_tag();
        if (i == 0) {
          tag = parent_tag;
        }
        if (tag != parent_tag) {
          tag = -1;
          break;
        }
      }
      if (tag < 0) {
        tag = 0;
      }
    }
    if (tag < 0 || tag >= static_cast<int>(fngrids.size())) {
      LBANN_ERROR("attempted to initialize ",
                  layer->get_type(),
                  " layer \"",
                  layer->get_name(),
                  "\" ",
                  "on invalid grid ",
                  "(grid tag ",
                  tag,
                  ", ",
                  fngrids.size(),
                  " grids available)");
    }
    layer->set_grid_tag(tag);
  }
}

void model::ensure_input_layers_first()
{

  // Find input layers
  std::vector<El::Int> input_layers, other_layers;
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    if (get_layer(i).get_type() == "input") {
      input_layers.push_back(i);
    }
    else {
      other_layers.push_back(i);
    }
  }

  // Reorder layers so input layers are executed first
  std::vector<El::Int> gather_indices;
  gather_indices.insert(gather_indices.end(),
                        input_layers.begin(),
                        input_layers.end());
  gather_indices.insert(gather_indices.end(),
                        other_layers.begin(),
                        other_layers.end());
  reorder_layers(gather_indices);
}

void model::setup_layers(size_t max_mini_batch_size,
                         const std::vector<El::Grid*>& grids_)
{

  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);

    for (const auto& cb : m_callbacks) {
      cb->on_setup_begin(this, &l);
    }

    l.set_model(this);
    l.setup(max_mini_batch_size, grids_);
    l.check_setup();

    for (const auto& cb : m_callbacks) {
      cb->on_setup_end(this, &l);
    }
  }
}

void model::setup_weights()
{

  // Sort weights by name
  // Note: For run-to-run consistency. Names are assumed to be unique.
  std::sort(m_weights.begin(),
            m_weights.end(),
            [](const OwningWeightsPtr& x, const OwningWeightsPtr& y) {
              return x->get_name().compare(y->get_name()) < 0;
            });

  // Setup weights
  for (auto&& w : m_weights) {
    w->setup();
  }
}

void model::add_evaluation_layers(std::unordered_set<Layer*>& layer_set,
                                  std::unordered_set<std::string>& layer_names)
{
  // Add evaluation layers corresponding to objective function layer terms
  for (auto* t : m_objective_function->get_terms()) {
    auto* term = dynamic_cast<layer_term*>(t);
    if (term != nullptr) {
      auto* l_raw_ptr = &term->get_layer();
      if (layer_set.count(l_raw_ptr) == 0) {
        LBANN_ERROR(
          "model \"",
          get_name(),
          "\" has an objective function layer term corresponding to layer \"",
          l_raw_ptr->get_name(),
          "\", which isn't in the model's list of layers");
      }
      if (dynamic_cast<abstract_evaluation_layer<DataType>*>(l_raw_ptr) ==
          nullptr) {

        // Get viewing pointer to layer
        ViewingLayerPtr l_view_ptr;
        for (auto& l : m_layers) {
          if (l.get() == l_raw_ptr) {
            l_view_ptr = l;
            break;
          }
        }
        if (l_view_ptr.lock().get() == nullptr) {
          LBANN_ERROR(get_name(),
                      " could not get viewing pointer for ",
                      l_raw_ptr->get_name(),
                      " layer \"",
                      l_raw_ptr->get_name(),
                      "\"");
        }

        // Create evaluation layer
        OwningLayerPtr eval(abstract_evaluation_layer<DataType>::construct(
          l_raw_ptr->get_comm(),
          l_raw_ptr->get_data_layout(),
          l_raw_ptr->get_device_allocation()));

        // Set evaluation layer name
        El::Int name_index = 1;
        std::string name = l_raw_ptr->get_name() + "_eval";
        while (layer_names.count(name) > 0) {
          name_index++;
          name = l_raw_ptr->get_name() + "_eval" + std::to_string(name_index);
        }
        eval->set_name(name);

        // Update workspace objects
        layer_set.insert(eval.get());
        layer_names.insert(eval->get_name());

        // Add evaluation layer to model
        l_raw_ptr->add_child_layer(eval);
        eval->add_parent_layer(l_view_ptr);
        term->set_layer(eval);
        add_layer(std::move(eval));
      }
    }
  }

  // Add evaluation layers corresponding to layer metrics
  for (auto& ptr : m_metrics) {
    auto* met = dynamic_cast<layer_metric*>(ptr.get());
    if (met != nullptr) {
      auto* l_raw_ptr = &met->get_layer();
      if (layer_set.count(l_raw_ptr) == 0) {
        LBANN_ERROR("layer metric \"",
                    met->name(),
                    "\" corresponds to layer \"",
                    l_raw_ptr->get_name(),
                    "\", which is not in model \"",
                    get_name(),
                    "\"");
      }
      if (!dynamic_cast<abstract_evaluation_layer<DataType>*>(l_raw_ptr)) {

        // Get viewing pointer to layer
        ViewingLayerPtr l_view_ptr;
        for (auto& l : m_layers) {
          if (l.get() == l_raw_ptr) {
            l_view_ptr = l;
            break;
          }
        }
        if (l_view_ptr.lock().get() == nullptr) {
          LBANN_ERROR(get_name(),
                      " could not get viewing pointer for ",
                      l_raw_ptr->get_name(),
                      " layer \"",
                      l_raw_ptr->get_name(),
                      "\"");
        }

        // Create evaluation layer
        OwningLayerPtr eval(abstract_evaluation_layer<DataType>::construct(
          l_raw_ptr->get_comm(),
          l_raw_ptr->get_data_layout(),
          l_raw_ptr->get_device_allocation()));

        // Set evaluation layer name
        El::Int name_index = 1;
        std::string name = l_raw_ptr->get_name() + "_eval";
        while (layer_names.count(name) > 0) {
          name_index++;
          name = l_raw_ptr->get_name() + "_eval" + std::to_string(name_index);
        }
        eval->set_name(name);

        // Update workspace objects
        layer_set.insert(eval.get());
        layer_names.insert(eval->get_name());

        // Add evaluation layer to model
        l_raw_ptr->add_child_layer(eval);
        eval->add_parent_layer(l_view_ptr);
        met->set_layer(eval);
        add_layer(std::move(eval));
      }
    }
  }
}

void model::add_dummy_layers(std::unordered_set<std::string>& layer_names)
{
  for (size_t i = 0; i < m_layers.size(); ++i) {
    auto& l = get_layer(i);
    while (l.get_num_children() < l.get_expected_num_child_layers()) {

      // Create dummy layer
      OwningLayerPtr dummy;
      using args_tuple = std::tuple<data_layout, El::Device>;
      args_tuple args(l.get_data_layout(), l.get_device_allocation());
      if (args == args_tuple(data_layout::DATA_PARALLEL, El::Device::CPU)) {
        dummy.reset(new dummy_layer<DataType,
                                    data_layout::DATA_PARALLEL,
                                    El::Device::CPU>(m_comm));
      }
      if (args == args_tuple(data_layout::MODEL_PARALLEL, El::Device::CPU)) {
        dummy.reset(new dummy_layer<DataType,
                                    data_layout::MODEL_PARALLEL,
                                    El::Device::CPU>(m_comm));
      }
#ifdef LBANN_HAS_GPU
      if (args == args_tuple(data_layout::DATA_PARALLEL, El::Device::GPU)) {
        dummy.reset(new dummy_layer<DataType,
                                    data_layout::DATA_PARALLEL,
                                    El::Device::GPU>(m_comm));
      }
      if (args == args_tuple(data_layout::MODEL_PARALLEL, El::Device::GPU)) {
        dummy.reset(new dummy_layer<DataType,
                                    data_layout::MODEL_PARALLEL,
                                    El::Device::GPU>(m_comm));
      }
#endif // LBANN_HAS_GPU
      if (dummy == nullptr) {
        LBANN_ERROR("Could not construct dummy layer corresponding to layer \"",
                    l.get_name(),
                    "\" in model \"",
                    get_name(),
                    "\"");
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
      l.add_child_layer(dummy);
      dummy->add_parent_layer(m_layers[i]);
      add_layer(std::move(dummy));
    }
  }
}

void model::add_split_layers(std::unordered_set<std::string>& layer_names)
{
  for (size_t i = 0; i < m_layers.size(); ++i) {
    auto& l = get_layer(i);

    // Add split layer if layer expects one child but has multiple
    if (l.get_expected_num_child_layers() == 1 && l.get_num_children() != 1) {

      // Create split layer
      OwningLayerPtr split;
      using args_tuple = std::tuple<std::type_index, data_layout, El::Device>;
      args_tuple args(l.get_output_datatype(),
                      l.get_data_layout(),
                      l.get_device_allocation());

#define PROTO_DEVICE_LAYOUT(T_datatype, T_layout, T_device)                              \
      if (args == args_tuple(std::type_index(typeid(T_datatype)), T_layout, T_device)) { \
        split.reset(new split_layer<T_datatype, T_layout, T_device>(m_comm));            \
      }

#define PROTO_DEVICE(T_datatype, T_device)                                    \
      PROTO_DEVICE_LAYOUT(T_datatype, data_layout::DATA_PARALLEL, T_device);  \
      PROTO_DEVICE_LAYOUT(T_datatype, data_layout::MODEL_PARALLEL, T_device);

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE_LAYOUT

      if (split == nullptr) {
        LBANN_ERROR("Could not construct split layer corresponding to layer \"",
                    l.get_name(),
                    "\" in model \"",
                    get_name(),
                    "\"");
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

      // Copy parallel strategy from parent.
      ParallelStrategy& ps = split->get_parallel_strategy();
      ParallelStrategy& orig_ps = l.get_parallel_strategy();
      ps = orig_ps;

      // Setup relationships between split layer and child layers
      for (int j = 0; j < l.get_num_children(); ++j) {
        auto& child = const_cast<Layer&>(l.get_child_layer(j));
        split->add_child_layer(l.get_child_layer_pointer(j));
        child.replace_parent_layer(split, child.find_parent_layer_index(l));
      }

      // Setup relationship between current layer and split layer
      l.clear_child_layers();
      l.add_child_layer(split);
      split->add_parent_layer(m_layers[i]);

      // Add split layer to layer list
      add_layer(std::move(split));
    }
  }
}

void model::insert_layer(OwningLayerPtr&& new_layer,
                         std::string const& preceding_layer_name)
{

  // Find preceding layer after which to insert new layer
  int preceding_layer_index = -1;
  for (int i = 0; i < this->get_num_layers(); ++i) {
    if (this->get_layer(i).get_name() == preceding_layer_name) {
      preceding_layer_index = i;
      break;
    }
  }
  if (preceding_layer_index < 0) {
    LBANN_ERROR("Attempted to insert layer after ",
                "layer \"",
                preceding_layer_name,
                "\", ",
                "but no such layer exists");
  }

  auto& l = get_layer(preceding_layer_index);

  // Set checks to ensure there is only one parent and one child
  LBANN_ASSERT(l.get_num_parents() == 1);
  LBANN_ASSERT(l.get_num_children() == 1);

  // Child of preceding layer
  auto& child =
    const_cast<Layer&>(l.get_child_layer(0)); // assuming only one child

  // Setup relationship between new layer and child layer
  new_layer->add_child_layer(l.get_child_layer_pointer(0));
  child.replace_parent_layer(new_layer, child.find_parent_layer_index(l));

  // Setup relationship between current (parent) layer and new layer
  l.clear_child_layers();
  l.add_child_layer(new_layer);
  new_layer->add_parent_layer(m_layers[preceding_layer_index]);

  // Add new_layer to layer list
  add_layer(std::move(new_layer));
}

void model::remove_layer(std::string const& removable_layer_name)
{

  // Find index of removable layer
  int removable_layer_index = -1;
  for (int i = 0; i < this->get_num_layers(); ++i) {
    if (this->get_layer(i).get_name() == removable_layer_name) {
      removable_layer_index = i;
      break;
    }
  }
  if (removable_layer_index < 0) {
    LBANN_ERROR("Attempted to remove layer",
                " \"",
                removable_layer_name,
                "\", ",
                "but no such layer exists");
  }

  auto& l = get_layer(removable_layer_index);

  // Set checks to ensure there is only one parent and one child
  LBANN_ASSERT(l.get_num_parents() == 1);
  LBANN_ASSERT(l.get_num_children() == 1);

  // Other checks also have to be done - like cannot delete activation function
  // between two layers, etc

  // Parent and child of removable layer
  auto& parent =
    const_cast<Layer&>(l.get_parent_layer(0)); // assuming only one parent
  auto& child =
    const_cast<Layer&>(l.get_child_layer(0)); // assuming only one child

  // Setup relationship between parent layer and child layer
  child.replace_parent_layer(l.get_parent_layer_pointer(0),
                             child.find_parent_layer_index(l));
  parent.replace_child_layer(l.get_child_layer_pointer(0),
                             parent.find_child_layer_index(l));

  // Remove weights for the old layer
  // NOTE : We assume that layers do not share weights
  auto old_weights_ptrs =
    m_layers[removable_layer_index]->get_weights_pointers();
  for (auto const& w : old_weights_ptrs) {
    this->remove_weights(std::shared_ptr<weights>(w)->get_name());
  }

  // Destroy memory of removable layer - for now, remove from m_layers
  m_layers.erase(m_layers.cbegin() + removable_layer_index);
}

void model::replace_layer(OwningLayerPtr&& new_layer,
                          std::string const& old_layer_name)
{

  // Find old layer
  int old_layer_index = -1;
  for (int i = 0; i < this->get_num_layers(); ++i) {
    if (this->get_layer(i).get_name() == old_layer_name) {
      old_layer_index = i;
      break;
    }
  }

  if (old_layer_index < 0) {
    LBANN_ERROR("Attempted to replace layer",
                " \"",
                old_layer_name,
                "\", ",
                "but no such layer exists");
  }

  // Old Layer
  auto& l = get_layer(old_layer_index);

  // Set checks to ensure there is only one parent and one child
  LBANN_ASSERT(l.get_num_parents() == 1);
  LBANN_ASSERT(l.get_num_children() == 1);

  // Parent and child of old Layer
  auto& parent =
    const_cast<Layer&>(l.get_parent_layer(0)); // assuming only one parent
  auto& child =
    const_cast<Layer&>(l.get_child_layer(0)); // assuming only one child

  // Setup relationship between the new layer and child of old layer (which
  // becomes child of new layer)
  new_layer->add_child_layer(l.get_child_layer_pointer(0));
  child.replace_parent_layer(new_layer, child.find_parent_layer_index(l));

  // Setup relationship between parent of old layer (which becomes parent of new
  // layer) and new layer
  parent.replace_child_layer(new_layer, parent.find_child_layer_index(l));
  new_layer->add_parent_layer(l.get_parent_layer_pointer(0));

  // Remove weights for the old layer
  // NOTE : We assume that layers do not share weights
  auto old_weights_ptrs = m_layers[old_layer_index]->get_weights_pointers();
  for (auto const& w : old_weights_ptrs) {
    this->remove_weights(std::shared_ptr<weights>(w)->get_name());
  }

  // Destroy memory of old layer - for now, remove from m_layers
  m_layers.erase(m_layers.cbegin() + old_layer_index);

  // Add new layer to layer list
  add_layer(std::move(new_layer));
}

// =============================================
// Execution
// =============================================

// At the start of the epoch, set the execution mode and make sure
// that each layer points to this model
void model::reset_mode(ExecutionContext& context, execution_mode mode)
{
  if (mode == execution_mode::invalid) {
    m_execution_context = nullptr;
    return;
  }
  m_execution_context = static_cast<observer_ptr<ExecutionContext>>(&context);
  //  set_execution_mode(mode);
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    get_layer(i).set_model(this);
  }
}

// At the end of the epoch, clean up the objective function and metrics
void model::reset_epoch_statistics(execution_mode mode)
{
  get_objective_function()->reset_statistics(mode);
  for (const auto& m : m_metrics) {
    m->reset_statistics(mode);
  }
}

void model::evaluate_metrics(execution_mode mode,
                             size_t current_mini_batch_size)
{
  for (const auto& m : m_metrics) {
    m->evaluate(mode, current_mini_batch_size);
  }
}

void model::clear_gradients()
{
  for (auto&& w : m_weights) {
    auto&& opt = w->get_optimizer();
    if (opt != nullptr) {
      opt->clear_gradient();
    }
  }
}

void model::forward_prop(execution_mode mode)
{
  LBANN_CALIPER_MARK_FUNCTION;
  do_model_forward_prop_begin_cbs(mode);

  // Clear activations in reference counter
  m_activation_refcnt.clear();

  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& l = get_layer(i);

    if (this->is_subgraph_parallelism_enabled()) {
      if (l.get_run_layer_in_subgraph() || l.get_name() == "layer1") {
        do_layer_forward_prop_begin_cbs(mode, &l);
        l.forward_prop();
        do_layer_forward_prop_end_cbs(mode, &l);
      }
      else {
        // To Do: Fix last batch problem in sub-graph parallelism
        // experimental code to fix last batch problem in subgraph parallelism
      }
    }
    else {
      do_layer_forward_prop_begin_cbs(mode, &l);
      l.forward_prop();
      do_layer_forward_prop_end_cbs(mode, &l);
    }
  }
  do_model_forward_prop_end_cbs(mode);
}

void model::backward_prop(bool compute_weight_grads_only)
{
  LBANN_CALIPER_MARK_FUNCTION;

  // Layers disabled due to not propagating error signals through
  std::unordered_set<const Layer*> disabled_layers;
  auto const& arg_parser = global_argument_parser();
  bool const envvar_disable_layers =
    !arg_parser.get<bool>(LBANN_OPTION_NO_BACKPROP_DISABLE);

  do_model_backward_prop_begin_cbs();

  for (El::Int i = get_num_layers() - 1; i >= 0; --i) {

    // Perform backward prop step on current layer
    auto& l = get_layer(i);
    bool enable_layer = (!envvar_disable_layers ||
                         disabled_layers.find(&l) == disabled_layers.end());

    // Check if all children skip gradient backpropagation
    if (enable_layer && envvar_disable_layers) {
      bool all_children_skip_gradient = l.get_num_children() > 0;
      for (auto& child : l.get_child_layers()) {
        if (disabled_layers.find(child) != disabled_layers.end())
          continue;
        all_children_skip_gradient &=
          (child->get_backprop_requirements() == PROPAGATE_NOTHING);
        if (!all_children_skip_gradient)
          break;
      }

      // Start disabling layers from this point onwards
      if (all_children_skip_gradient) {
        enable_layer = false;
        disabled_layers.insert(&l);
      }
    }

    if (this->is_subgraph_parallelism_enabled()) {
      if (l.get_run_layer_in_subgraph()) {
        do_layer_backward_prop_begin_cbs(&l);
        if (enable_layer)
          l.back_prop();
        do_layer_backward_prop_end_cbs(&l);
      }
      else {
        // To Do: Fix last batch problem in sub-graph parallelism
        // experimental code to fix last batch problem in subgraph parallelism
      }
    }
    else {
      do_layer_backward_prop_begin_cbs(&l);
      if (enable_layer)
        l.back_prop();
      do_layer_backward_prop_end_cbs(&l);
    }

    // Terminate early if all gradients have been computed
    bool all_gradients_computed = true;
    for (auto&& w : m_weights) {
      auto&& opt = w->get_optimizer();
      if (opt != nullptr && opt->get_num_gradient_sources() != 0) {
        all_gradients_computed = false;
        break;
      }
    }

    // in parent having less resources case
    // last slice layer does not run as gradients are not present for ranks that
    // are not
    //  in parent grid leading to hang

    // Tim or Tom: What is your suggestation?
    if (compute_weight_grads_only && all_gradients_computed &&
        this->is_subgraph_parallelism_enabled() == false) {
      break;
    }
  }

  do_model_backward_prop_end_cbs();
}

void model::update_weights()
{
  LBANN_CALIPER_MARK_FUNCTION;
  do_model_optimize_begin_cbs();

  // AMP: Check gradients for NaNs and infinities.
  // If any are found, this iteration will be skipped.
  // If not, the gradients will be unscaled.
  bool skip_step = false;
  if (is_amp_enabled()) {
    for (auto rit = m_weights.rbegin(); rit != m_weights.rend(); ++rit) {
      auto& w = **rit;
      auto&& opt = w.get_optimizer();
      if (opt != nullptr) {
        skip_step = !opt->is_gradient_finite_and_unscale(m_amp_scale_factor);
        if (skip_step) {
          break;
        }
      }
    }
  }

  if (!skip_step) {
    // Apply optimization step to weights
    // Note: Heuristically, forward prop consumes weights in the same
    // order as m_weights and backprop computes weights gradients in
    // reverse order. Also, we often launch a non-blocking allreduce
    // after a weights gradient has been computed. Thus, iterating in
    // reverse order will use gradients that have already finished their
    // allreduce, giving more time for more recent allreduces to finish.
    for (auto rit = m_weights.rbegin(); rit != m_weights.rend(); ++rit) {
      auto& w = **rit;
      auto&& opt = w.get_optimizer();

      if (opt != nullptr) {
        do_weight_optimize_begin_cbs(&w);
        opt->step();
        do_weight_optimize_end_cbs(&w);
      }
    }
  }

  // AMP: Update loss scale.
  if (is_amp_enabled()) {
    if (skip_step) {
      m_amp_cur_steps = 0;
      ++m_amp_cur_skipped_steps;
      // Keep scale factor to the smallest positive normalized value for
      // floats. Even when EvalType is double, we may cast to float.
      m_amp_scale_factor = std::max(
        static_cast<EvalType>(std::numeric_limits<float>::min()),
        m_amp_scale_factor * m_amp_backoff_factor);
      // Warn if we've been skipping too many steps.
      // Check exact number to avoid printing repeatedly.
      if (m_amp_cur_skipped_steps == 10) {
        LBANN_WARNING(
          "AMP skipped ten steps in a row, your model may have issues with AMP");
      }
    } else {
      if (m_amp_cur_steps + 1 == m_amp_growth_interval) {
        m_amp_cur_steps = 0;
        m_amp_cur_skipped_steps = 0;
        // Prevent scale factor from overflowing to inf when cast to
        // float.
        m_amp_scale_factor = std::min(
          static_cast<EvalType>(std::numeric_limits<float>::max()),
          m_amp_scale_factor * m_amp_growth_factor);
      } else {
        ++m_amp_cur_steps;
      }
    }
    get_objective_function()->set_amp_scale(m_amp_scale_factor);
  }

  do_model_optimize_end_cbs();
}

bool model::update_layers()
{
  bool finished = true;
  for (El::Int i = get_num_layers() - 1; i >= 0; --i) {
    finished = get_layer(i).update() && finished;
  }
  return finished;
}

void model::reconcile_weight_values()
{
  LBANN_ERROR("This should not be called. Method is a candidate for removal");
  // Launch non-blocking communication to reconcile weights
  // Note: Heuristically, forward prop consumes weights in the same
  // order as m_weights. Also, weights tend to get larger as you get
  // deeper into a neural network. Thus, iterating in reverse order
  // means that we perform the expensive communication first, covering
  // up the launch overheads for the subsequent cheap communication.
  std::vector<Al::request> reqs;
  reqs.reserve(m_weights.size());
  for (auto rit = m_weights.rbegin(); rit != m_weights.rend(); ++rit) {
    auto& w = **rit;
    reqs.emplace_back();
    w.reconcile_values(reqs.back());
  }

  // Wait for communication to finish
  for (auto& req : reqs) {
    m_comm->wait(req);
  }
}

void model::enable_amp(EvalType init_scale_factor,
                       EvalType growth_factor,
                       EvalType backoff_factor,
                       size_t growth_interval) {
  m_amp_enabled = true;
  m_amp_scale_factor = init_scale_factor;
  m_amp_growth_factor = growth_factor;
  m_amp_backoff_factor = backoff_factor;
  m_amp_growth_interval = growth_interval;
}

// =============================================
// Callbacks
// =============================================

void model::do_setup_end_cbs()
{
  for (const auto& cb : m_callbacks) {
    cb->on_setup_end(this);
  }
}

void model::do_model_forward_prop_begin_cbs(execution_mode mode)
{
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_begin(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::tournament:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_begin(this);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void model::do_model_forward_prop_end_cbs(execution_mode mode)
{
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_end(this);
      }
      break;
    case execution_mode::validation:
    case execution_mode::tournament:
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
void model::do_layer_forward_prop_begin_cbs(execution_mode mode, Layer* l)
{
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_begin(this, l);
      }
      break;
    case execution_mode::validation:
    case execution_mode::tournament:
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
void model::do_layer_forward_prop_end_cbs(execution_mode mode, Layer* l)
{
  for (const auto& cb : m_callbacks) {
    switch (mode) {
    case execution_mode::training:
      if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
        cb->on_forward_prop_end(this, l);
      }
      break;
    case execution_mode::validation:
    case execution_mode::tournament:
    case execution_mode::testing:
      cb->on_evaluate_forward_prop_end(this, l);
      break;
    default:
      LBANN_ERROR("invalid execution mode");
    }
  }
}

void model::do_model_backward_prop_begin_cbs()
{
  for (const auto& cb : m_callbacks) {
    if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_begin(this);
    }
  }
}

void model::do_model_backward_prop_end_cbs()
{
  for (const auto& cb : m_callbacks) {
    if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_end(this);
    }
  }
}

void model::do_layer_backward_prop_begin_cbs(Layer* l)
{
  for (const auto& cb : m_callbacks) {
    if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_begin(this, l);
    }
  }
}

void model::do_layer_backward_prop_end_cbs(Layer* l)
{
  for (const auto& cb : m_callbacks) {
    if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_end(this, l);
    }
  }
}

void model::do_model_optimize_begin_cbs()
{
  for (const auto& cb : m_callbacks) {
    if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_begin(this);
    }
  }
}

void model::do_model_optimize_end_cbs()
{
  for (const auto& cb : m_callbacks) {
    if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_end(this);
    }
  }
}

void model::do_weight_optimize_begin_cbs(weights* w)
{
  for (const auto& cb : m_callbacks) {
    if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_begin(this, w);
    }
  }
}

void model::do_weight_optimize_end_cbs(weights* w)
{
  for (const auto& cb : m_callbacks) {
    if (get_execution_context().get_step() % cb->get_batch_interval() == 0) {
      cb->on_optimize_end(this, w);
    }
  }
}

// =============================================
// Summarizer
// =============================================

void model::summarize_stats(lbann_summary& summarizer)
{
  const auto& c = get_execution_context();
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    get_layer(i).summarize_stats(summarizer, c.get_step());
  }
  summarizer.reduce_scalar(
    "objective",
    m_objective_function->get_mean_value(c.get_execution_mode()),
    c.get_step());
  summarizer.reduce_scalar("objective_evaluation_time",
                           m_objective_function->get_evaluation_time(),
                           c.get_step());
  summarizer.reduce_scalar("objective_differentiation_time",
                           m_objective_function->get_differentiation_time(),
                           c.get_step());
  m_objective_function->reset_counters();
  double total_metric_time = 0.0;
  for (auto&& m : m_metrics) {
    total_metric_time += m->get_evaluate_time();
    m->reset_counters();
  }
  summarizer.reduce_scalar("metric_evaluation_time",
                           total_metric_time,
                           c.get_step());
}

void model::summarize_matrices(lbann_summary& summarizer)
{
  const auto& c = get_execution_context();
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    get_layer(i).summarize_matrices(summarizer, c.get_step());
  }
}

// =============================================
// Checkpointing
// =============================================

/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_model_header
{
  uint32_t callback_type;
};

bool model::save_to_checkpoint_shared(persist& p)
{
  const std::string trainer_dir = p.get_checkpoint_dir();

  // This "pushes" the model-specific directory to the "stack". After
  // the call, p.get_checkpoint_dir() returns the model-specific
  // directory.
  p.open_checkpoint_dir(file::join_path(trainer_dir, this->get_name()),
                        m_comm->am_trainer_master());

  // Make sure that the master has had a chance to create the directories
  // (trb 12/14/2020): I don't think this matters; all output is from
  //                   the trainer master...
  m_comm->trainer_barrier();

  // Open the stream for writing
  std::ofstream ofs;
  if (m_comm->am_trainer_master()) {
    ofs.open(file::join_path(p.get_checkpoint_dir(), "model.bin"));
    LBANN_ASSERT(ofs.good());
  }

  // Write the checkpoint
  {
    lbann::RootedBinaryOutputArchive ar(ofs, m_comm->get_trainer_grid());
    ar(*this);
  }

  p.open_checkpoint_dir(trainer_dir, false);
  return true;
}

bool model::load_from_checkpoint_shared(persist& p)
{
  const std::string trainer_dir = p.get_checkpoint_dir();
  p.open_restart(file::join_path(trainer_dir, get_name()));
  // Assume checkpoint reload from epoch end not step end

  std::ifstream ifs;
  if (m_comm->am_trainer_master()) {
    ifs.open(file::join_path(p.get_checkpoint_dir(), "model.bin"));
    LBANN_ASSERT(ifs.good());
  }

  // Restore the checkpoint
  {
    lbann::RootedBinaryInputArchive ar(ifs, m_comm->get_trainer_grid());
    ar(*this);
  }

  m_model_is_setup = false;
  p.set_restart_dir(trainer_dir);
#ifdef LBANN_HAS_GPU
  hydrogen::gpu::SynchronizeDevice();
#endif // LBANN_HAS_GPU
  return true;
}

bool model::save_to_checkpoint_distributed(persist& p)
{
  const std::string trainer_dir = p.get_checkpoint_dir();
  p.open_checkpoint_dir(file::join_path(trainer_dir, get_name()), true);

  // Make sure that the master has had a chance to create the directories
  m_comm->trainer_barrier();

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  {
    std::ofstream ofs(file::join_path(p.get_checkpoint_dir(), "model.bin"));
    cereal::BinaryOutputArchive ar(ofs);
    ar(*this);
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  {
    std::ofstream ofs_xml(file::join_path(p.get_checkpoint_dir(), "model.xml"));
    cereal::XMLOutputArchive ar(ofs_xml);
    ar(*this);
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES

  p.open_checkpoint_dir(trainer_dir, false);
  return true;
}

bool model::load_from_checkpoint_distributed(persist& p)
{
  const std::string trainer_dir = p.get_checkpoint_dir();
  p.open_restart(file::join_path(trainer_dir, get_name()));

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  {
    std::ifstream ifs(file::join_path(p.get_checkpoint_dir(), "model.bin"));
    cereal::BinaryInputArchive ar(ifs);
    ar(*this);
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

  m_model_is_setup = false;
  p.set_restart_dir(trainer_dir);
  return true;
}

void model::write_proto(lbann_data::Model& proto)
{
  if (!get_comm()->am_trainer_master())
    return;

  proto.Clear();
  proto.set_name(this->get_name());
  this->get_objective_function()->write_proto(
    *proto.mutable_objective_function());
  for (auto const* metric : this->get_metrics()) {
    auto* met = proto.add_metric()->mutable_layer_metric();
    met->set_name(metric->name());
    met->set_unit(metric->get_unit());
  }
  for (auto* layer : this->get_layers()) {
    layer->write_proto(*proto.add_layer());
  }
  for (auto* weights : this->get_weights()) {
    weights->write_proto(*proto.add_weights());
  }
  for (auto* callback : this->get_callbacks()) {
    callback->write_proto(*proto.add_callback());
  }
  for (auto const& l : m_layers) {
    auto* l_msg = proto.add_layer();
    l->write_proto(*l_msg);
  }

  // Unused proto fields
  // proto.set_data_layout(string_value);
  // proto.set_num_epochs(int64_value);
  // proto.set_num_batches(int64_value);
  // proto.set_evaluation_frequency(int64_value);
  // proto.set_subgraph_communication(SubGraphCommunication_value);
  // proto.enable_subgraph_topology(bool_value);
  // proto.subgraph_parent_grid_resources(int64_value);
  // proto.set_disable_cuda(bool_value);
  // proto.set_summarizer(Summarizer_value);
}

void model::save_model()
{
  for (auto&& c : m_callbacks) {
    if (auto* cb = dynamic_cast<callback::save_model*>(c.get())) {
      cb->do_save_model(this);
      return;
    }
  }
  if (m_comm->am_trainer_master()) {
    LBANN_WARNING(
      "save_model was called, but the callback_save_model was not loaded");
  }
}

#ifdef LBANN_HAS_DISTCONV
void model::setup_distconv()
{
  std::ostringstream dc_enabled, dc_disabled;
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& layer = get_layer(i);
    if (layer.distconv_enabled()) {
      dc_enabled << " " << layer.get_name();
    }
    else {
      dc_disabled << " " << layer.get_name();
    }
  }

  // Short-circuit if this model isn't using distconv! Like most of
  // the models in LBANN's app directory!
  if (dc_enabled.tellp() == typename std::ostringstream::pos_type{0})
    return;

  if (m_comm->am_world_master()) {
    std::cout << "\nDistconv-enabled layers:\n\t" << dc_enabled.str() << "\n\n"
              << "Distconv-disabled layers:\n\t" << dc_disabled.str() << "\n\n";

#ifdef LBANN_DEBUG
    std::cout << "Parallel Strategy description - ";
    print_parallel_strategy_header(std::cout);
#endif
    endl(std::cout);
  }
  setup_distributions();
  print_distributions();
  // Setup fp tensors
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& layer = get_layer(i);
    if (!layer.distconv_enabled())
      continue;
    layer.get_distconv_adapter().setup_fp_tensors();
  }
  // Setup bp tensors in an reverse order
  for (El::Int i = get_num_layers() - 1; i >= 0; --i) {
    auto& layer = get_layer(i);
    if (!layer.distconv_enabled())
      continue;
    layer.get_distconv_adapter().setup_bp_tensors();
  }
  // Final setup.
  auto workspace_capacity = dc::get_workspace_capacity();
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    auto& layer = get_layer(i);
    if (!layer.distconv_enabled())
      continue;
    layer.get_distconv_adapter().setup_layer(workspace_capacity);
  }
}

void model::setup_distributions()
{
  tensor_overlap_constraints constraints;
  // Initialize the distributions and constraints
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    if (!get_layer(i).distconv_enabled())
      continue;
    get_layer(i).get_distconv_adapter().setup_distributions(constraints);
  }
  // Add inter-layer distribution constraints
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    if (!get_layer(i).distconv_enabled())
      continue;

    get_layer(i).get_distconv_adapter().impose_adjacent_overlap_constraints(
      constraints);
  }
  constraints.find_valid_overlap();
}

void model::print_distributions() const
{
  std::ostringstream ss;
  for (El::Int i = 0; i < get_num_layers(); ++i) {
    const auto& layer = get_layer(i);
    if (layer.distconv_enabled()) {
      ss << layer.get_name() << " disributions: "
         << "prev_activations: "
         << layer.get_distconv_adapter().get_prev_activations_dist()
         << ", activations: "
         << layer.get_distconv_adapter().get_activations_dist()
         << ", error_signals: "
         << layer.get_distconv_adapter().get_error_signals_dist()
         << ", prev_error_signals: "
         << layer.get_distconv_adapter().get_prev_activations_dist() << "\n";
    }
    else {
      ss << layer.get_name() << ": distconv disabled"
         << "\n";
    }
  }
  dc::MPIRootPrintStreamDebug() << ss.str();
}
#endif // LBANN_HAS_DISTCONV

} // namespace lbann

#define LBANN_CLASS_NAME model
#define LBANN_SKIP_CEREAL_REGISTRATION
#include <lbann/macros/register_class_with_cereal.hpp>
