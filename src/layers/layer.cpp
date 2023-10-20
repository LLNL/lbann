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

#include "lbann/layers/layer.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/models/model.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/summary_impl.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/weights/weights.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

#include "lbann/proto/layers.pb.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lbann {

Layer::Layer() : m_frozen(false)
{

  // Initialize layer name
  static int num_layers = 0;
  m_name = "layer" + std::to_string(num_layers);
  num_layers++;

  // Reset timing counters
  reset_counters();
}

Layer::Layer(const Layer& other)
  : m_expected_num_parent_layers(other.m_expected_num_parent_layers),
    m_expected_num_child_layers(other.m_expected_num_child_layers),
    m_model(other.m_model),
    m_frozen(other.m_frozen),
    m_fp_time(other.m_fp_time),
    m_fp_compute_time(other.m_fp_compute_time),
    m_bp_time(other.m_bp_time),
    m_bp_compute_time(other.m_bp_compute_time),
    m_update_time(other.m_update_time),
    m_name(other.m_name),
    m_runs_inplace(other.m_runs_inplace),
    m_parent_layers(other.m_parent_layers),
    m_child_layers(other.m_child_layers),
    m_weights(other.m_weights),
    m_output_dims_list(other.m_output_dims_list),
    m_hint_layer(other.m_hint_layer)
{}

Layer& Layer::operator=(const Layer& other)
{

  // Shallow copies
  m_expected_num_parent_layers = other.m_expected_num_parent_layers;
  m_expected_num_child_layers = other.m_expected_num_child_layers;
  m_model = other.m_model;
  m_frozen = other.m_frozen;
  m_fp_time = other.m_fp_time;
  m_fp_compute_time = other.m_fp_compute_time;
  m_bp_time = other.m_bp_time;
  m_bp_compute_time = other.m_bp_compute_time;
  m_update_time = other.m_update_time;
  m_name = other.m_name;
  m_parent_layers = other.m_parent_layers;
  m_child_layers = other.m_child_layers;
  m_weights = other.m_weights;
  m_output_dims_list = other.m_output_dims_list;
  m_hint_layer = other.m_hint_layer;
  m_runs_inplace = other.m_runs_inplace;

  return *this;
}

description Layer::get_description() const
{

  // Construct description object
  std::stringstream ss;
  ss << get_name() << " (" << get_type() << ")";
  description desc(ss.str());

  // Input dimensions
  const auto& parents = get_parent_layers();
  if (!parents.empty()) {
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < parents.size(); ++i) {
      ss << (i > 0 ? ", " : "");
      const auto& dims = get_input_dims(i);
      for (size_t j = 0; j < dims.size(); ++j) {
        ss << (j == 0 ? "" : "x") << dims[j];
      }
      ss << " (from ";
      if (parents[i] == nullptr) {
        ss << "unknown layer";
      }
      else {
        ss << parents[i]->get_type() << " layer "
           << "\"" << parents[i]->get_name() << "\"";
      }
      ss << ")";
    }
    desc.add("Input dimensions", ss.str());
  }

  // Output dimensions
  const auto& children = get_child_layers();
  if (!children.empty()) {
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < children.size(); ++i) {
      ss << (i > 0 ? ", " : "");
      const auto& dims = get_output_dims(i);
      for (size_t j = 0; j < dims.size(); ++j) {
        ss << (j == 0 ? "" : "x") << dims[j];
      }
      ss << " (to ";
      if (children[i] == nullptr) {
        ss << "unknown layer";
      }
      else {
        ss << children[i]->get_type() << " layer "
           << "\"" << children[i]->get_name() << "\"";
      }
      ss << ")";
    }
    desc.add("Output dimensions", ss.str());
  }

  // Weights
  if (has_weights()) {
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < num_weights(); ++i) {
      const auto& w = get_weights(i);
      ss << (i > 0 ? ", " : "");
      const auto& dims = w.get_dims();
      ss << w.get_name() << " (";
      for (size_t j = 0; j < dims.size(); ++j) {
        ss << (j > 0 ? "x" : "") << dims[j];
      }
      ss << ")";
    }
    desc.add("Weights", ss.str());
  }

  // Data layout
  ss.str(std::string{});
  ss.clear();
  switch (get_data_layout()) {
  case data_layout::DATA_PARALLEL:
    ss << "data-parallel";
    break;
  case data_layout::MODEL_PARALLEL:
    ss << "model-parallel";
    break;
  case data_layout::invalid:
  default:
    ss << "invalid";
  }
  desc.add("Data layout", ss.str());

  // Device
  ss.str(std::string{});
  ss.clear();
  switch (get_device_allocation()) {
  case El::Device::CPU:
    ss << "CPU";
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    ss << "GPU";
    break;
#endif // LBANN_HAS_GPU
  default:
    ss << "unknown";
  }
  desc.add("Device", ss.str());

  // DataType
  desc.add("Data type", get_datatype_name());

  // Sub-grid
  desc.add("Process grid", get_grid_tag());

  // Freeze state
  if (is_frozen()) {
    desc.add("Frozen");
  }

  if (this->m_runs_inplace) {
    desc.add("In-place");
  }

#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled()) {
    const auto& ps = get_parallel_strategy();
    ss.str(std::string{});
    ss.clear();
    ss << ps;
    desc.add("Parallel Strategy", ss.str());
  }
#endif // LBANN_HAS_DISTCONV

  return desc;
}

lbann_comm* Layer::get_comm() const
{
  if (m_model == nullptr) {
    LBANN_ERROR("attempted to get communicator from ",
                get_type(),
                " layer \"",
                get_name(),
                "\" ",
                "before it was configured");
  }
  return m_model->get_comm();
}

int Layer::get_grid_tag() const noexcept { return m_grid_tag; }

void Layer::set_grid_tag(int tag) { m_grid_tag = tag; }

bool Layer::update()
{
  if (m_frozen) {
    return true;
  }
  // Apply any updates.
  const auto update_compute_start = get_time();
  const auto layer_done = update_compute();
  m_update_time += get_time() - update_compute_start;
  return layer_done;
}

void Layer::reset_counters()
{
  m_fp_time = EvalType(0);
  m_fp_compute_time = EvalType(0);
  m_bp_time = EvalType(0);
  m_bp_compute_time = EvalType(0);
  m_update_time = EvalType(0);
}

void Layer::summarize_stats(lbann_summary& summarizer, int step)
{
  std::string prefix = m_name + "/";
  summarizer.reduce_scalar(prefix + "fp_time", m_fp_time, step);
  summarizer.reduce_scalar(prefix + "bp_time", m_bp_time, step);
  summarizer.reduce_scalar(prefix + "update_time", m_update_time, step);
  summarizer.reduce_scalar_all(prefix + "fp_time", m_fp_time, step);
  summarizer.reduce_scalar_all(prefix + "bp_time", m_bp_time, step);
  summarizer.reduce_scalar_all(prefix + "update_time", m_update_time, step);
  reset_counters();
  // Combine the optimizer step time from all the weights.
  double step_time = 0.0;
  for (size_t i = 0; i < num_weights(); ++i) {
    auto& w = get_weights(i);
    auto* opt = w.get_optimizer();
    if (opt) {
      step_time += opt->get_step_time();
      opt->reset_counters();
    }
  }
  summarizer.reduce_scalar(prefix + "opt_time", step_time, step);
  summarizer.reduce_scalar_all(prefix + "opt_time", step_time, step);
}

// ===================================================================
// Tensor dimension access functions
// ===================================================================

std::vector<int> Layer::get_input_dims(size_t input_index) const
{
  const auto& parent = get_parent_layer(input_index);
  const auto parent_output_index = parent.find_child_layer_index(*this);
  return parent.get_output_dims(parent_output_index);
}

int Layer::get_input_size(size_t input_index) const
{
  return get_linear_size(get_input_dims(input_index));
}

std::vector<int> Layer::get_output_dims(size_t output_index) const
{
  const size_t num_outputs = get_num_children();
  if (m_output_dims_list.size() != num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of output tensor "
        << "in layer \"" << get_name() << "\" "
        << "before they are initialized";
    LBANN_ERROR(err.str());
  }
  else if (output_index >= num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of invalid output tensor "
        << "in layer \"" << get_name() << "\" "
        << "(requested index " << output_index << ", but there are "
        << num_outputs << " output tensors)";
    LBANN_ERROR(err.str());
  }
  return m_output_dims_list[output_index];
}

int Layer::get_output_size(size_t output_index) const
{
  return get_linear_size(get_output_dims(output_index));
}

void Layer::set_output_dims(std::vector<int> dims, size_t output_index)
{
  if (static_cast<int>(m_output_dims_list.size()) != get_num_children() ||
      m_output_dims_list.size() <= output_index) {
    // Handles case where dims are set before child layers are set
    m_output_dims_list.resize(El::Max(get_num_children(), output_index + 1));
  }
  m_output_dims_list[output_index] = std::move(dims);
}

El::Int Layer::infer_mini_batch_size_from_parents() const
{
  El::Int inferred_mini_batch_size = 0;
  std::string inferred_parent_layer_name;
  for (int i = 0; i < get_num_parents(); ++i) {
    // Set the mini-batch size based on the parent tensors
    const auto& parent = get_parent_layer(i);
    const auto& parent_output = parent.get_activations(*this);
    if (inferred_mini_batch_size == 0) {
      inferred_mini_batch_size = parent_output.Width();
      inferred_parent_layer_name = parent.get_name();
    }
    else if (inferred_mini_batch_size != parent_output.Width()) {
      // Check mini-batch matrix dimensions
      LBANN_ERROR("Layer ",
                  get_name(),
                  " has multiple parents with different mini-batch sizes: ",
                  inferred_parent_layer_name,
                  "=",
                  inferred_mini_batch_size,
                  " vs ",
                  parent.get_name(),
                  "=",
                  parent_output.Width());
    }
  }
  return inferred_mini_batch_size;
}

std::vector<ViewingWeightsPtr> Layer::get_weights_pointers() const
{
  return m_weights;
}

void Layer::set_weights_pointers(std::vector<ViewingWeightsPtr> ptrs)
{
  m_weights = std::move(ptrs);
}

// FIXME (trb 05/28/2020): IMO, this function name is somewhat
// misleading. It's not "replacing" anything -- it's overwriting the
// weights values of "this" with the weights values of "other_layer",
// which is left intact.
//
// ALSO, really what it does is copies the first "number of weights
// 'this' expects to have" and ignores any others that might be
// present in "other_layer".
//
// The use-cases of this function are outside the scope of my current
// work, so I'm "refactoring in-place" and leaving this documentation
// for a future refactor.
void Layer::replace_weights(Layer const& other_layer)
{

  auto const other_num_weights = other_layer.num_weights();
  auto const my_num_weights = this->num_weights();

  // Minimal sanity check; see longer note above.
  if (other_num_weights < my_num_weights)
    LBANN_ERROR("Expected at least ",
                my_num_weights,
                " weights in layer \"",
                other_layer.get_name(),
                "\" but found ",
                other_num_weights);

  using IdxT = typename std::decay<decltype(my_num_weights)>::type;
  for (IdxT ii = 0; ii < my_num_weights; ++ii) {
    auto const& other_layer_weights = other_layer.get_weights(ii);
    this->get_weights(ii).set_values(other_layer_weights.get_values_sharded());
  }
}

void Layer::set_hint_layer(ViewingLayerPtr l) { m_hint_layer = std::move(l); }

const Layer* Layer::get_hint_layer() const { return m_hint_layer.lock().get(); }

void Layer::freeze()
{
  m_frozen = true;
  for (size_t i = 0; i < num_weights(); ++i) {
    get_weights(i).freeze();
  }
}

void Layer::unfreeze()
{
  m_frozen = false;
  for (size_t i = 0; i < num_weights(); ++i) {
    get_weights(i).unfreeze();
  }
}

bool Layer::is_frozen() const
{
  for (size_t i = 0; i < num_weights(); ++i) {
    const auto& w = get_weights(i);
    if (w.is_frozen() != m_frozen) {
      LBANN_ERROR("layer ",
                  get_name(),
                  " and weight ",
                  w.get_name(),
                  " of it are inconsistently frozen");
    }
  }
  return m_frozen;
}

void Layer::setup(size_t max_mini_batch_size,
                  const std::vector<El::Grid*>& grids)
{
  setup_pointers();
  setup_dims();
  setup_matrices(grids);

#ifdef LBANN_HAS_DISTCONV
  prepare_distconv();
#endif // LBANN_HAS_DISTCONV
  setup_data(max_mini_batch_size);
  if (using_gpus()) {
    setup_gpu();
  }
}

namespace {

std::string get_parent_names(Layer const& l)
{
  std::ostringstream ss;
  for (int i = 0; i < l.get_num_parents(); ++i) {
    ss << (i > 0 ? ", " : "") << l.get_parent_layer(i).get_name();
  }
  return ss.str();
}

std::string get_child_names(Layer const& l)
{
  std::ostringstream ss;
  for (int i = 0; i < l.get_num_children(); ++i) {
    ss << (i > 0 ? ", " : "") << l.get_child_layer(i).get_name();
  }
  return ss.str();
}
} // namespace

void Layer::setup_pointers()
{

  // Check that the parent pointers are valid
  for (int i = 0; i < get_num_parents(); ++i) {
    const auto& parent = get_parent_layer(i);
    const auto index_in_parent = parent.find_child_layer_index(*this);
    if (static_cast<int>(index_in_parent) >= parent.get_num_children()) {
      LBANN_ERROR(parent.get_type(),
                  " layer \"",
                  parent.get_name(),
                  "\" ",
                  "is a parent of ",
                  get_type(),
                  " layer \"",
                  get_name(),
                  "\", ",
                  "but \"",
                  get_name(),
                  "\" is not a child of \"",
                  parent.get_name(),
                  "\"");
    }
  }

  // Check that the child pointers are valid
  for (int i = 0; i < get_num_children(); ++i) {
    const auto& child = get_child_layer(i);
    const auto index_in_child = child.find_parent_layer_index(*this);
    if (static_cast<int>(index_in_child) >= child.get_num_parents()) {
      LBANN_ERROR(child.get_type(),
                  " layer \"",
                  child.get_name(),
                  "\" ",
                  "is a child of ",
                  get_type(),
                  " layer \"",
                  get_name(),
                  "\", ",
                  "but \"",
                  get_name(),
                  "\" is not a parent of \"",
                  child.get_name(),
                  "\"");
    }
  }

  // Check that the number of parents/children are valid
  if (m_expected_num_parent_layers >= 0 &&
      get_num_parents() != m_expected_num_parent_layers) {
    LBANN_ERROR(get_type(),
                " layer \"",
                get_name(),
                "\" "
                "expects ",
                m_expected_num_parent_layers,
                " parent layers, ",
                "but found ",
                get_num_parents(),
                " (",
                get_parent_names(*this),
                ")");
  }
  if (m_expected_num_child_layers >= 0 &&
      get_num_children() != m_expected_num_child_layers) {
    LBANN_ERROR(get_type(),
                " layer \"",
                get_name(),
                "\" "
                "expects ",
                m_expected_num_child_layers,
                " child layers, ",
                "but found ",
                get_num_children(),
                " (",
                get_child_names(*this),
                ")");
  }

  // Set whether this layer will run in-place

  // Check for environment variable that disables this behavior
  auto const& arg_parser = global_argument_parser();
  bool const envvar_disable_inplace =
    arg_parser.get<bool>(LBANN_OPTION_NO_INPLACE);
  if (!this->can_run_inplace() || envvar_disable_inplace) {
    // TODO (later): Support distconv-enabled layers
    this->m_runs_inplace = false;
  }
  else {
    bool can_run_inplace = true;

    // If a layer needs its own previous activations for backprop, it cannot
    // run in-place
    if (this->get_backprop_requirements() & PREV_ACTIVATIONS) {
      can_run_inplace = false;
    }

    // For now, disable in-place operation for layers with multiple parents
    // or children until behavior is well-defined. TODO (later): Support
    if (get_num_parents() > 1 || get_num_children() > 1)
      can_run_inplace = false;

    if (can_run_inplace) {
      // If any of the parents needs its output activations for
      // backprop, this layer cannot run in-place.
      for (int i = 0; i < get_num_parents(); ++i) {
        const auto& parent = get_parent_layer(i);

        int bp_requirements = parent.get_backprop_requirements();
        if (bp_requirements & ACTIVATIONS) {
          can_run_inplace = false;
          break;
        }
      }
    }

    // TODO:
    // If any of the children is a viewing layer (Identity, Reshape, etc.),
    // there is a bug (issue #2274) in which a layer deallocates memory too soon
    // and deep-copying the tensors during backprop fails. THE FOLLOWING LINES
    // SHOULD BE REMOVED AFTER NEW TENSORS ARE MERGED
    if (can_run_inplace) {
      for (int i = 0; i < get_num_children(); ++i) {
        const auto& child = get_child_layer(i);
        if (child.get_type() == "identity" || child.get_type() == "reshape" ||
            child.get_type() == "identity_zero") {
          can_run_inplace = false;
          break;
        }
      }
    }

    this->m_runs_inplace = can_run_inplace;
  }
}

void Layer::setup_dims()
{
  m_output_dims_list.resize(get_num_children());
  const auto* hint_layer = get_hint_layer();
  if (hint_layer != nullptr) {
    const auto& hint_dims = hint_layer->get_output_dims();
    for (auto& output_dims : m_output_dims_list) {
      output_dims = hint_dims;
    }
  }
  else if (get_num_parents() > 0) {
    const auto& input_dims = get_input_dims();
    for (auto& output_dims : m_output_dims_list) {
      if (output_dims.empty()) {
        output_dims = input_dims;
      }
    }
  }
}

void Layer::check_setup()
{
  std::stringstream err;

  // Check tensor dimensions
  for (int i = 0; i < get_num_parents(); ++i) {
    const auto& dims = get_input_dims(i);
    if (dims.empty()) {
      err << "layer \"" << get_name() << "\" has "
          << "uninitialized input tensor dimensions "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
    if (std::any_of(dims.begin(), dims.end(), [](int d) { return d <= 0; })) {
      err << "layer \"" << get_name() << "\" has invalid "
          << "input tensor dimensions (";
      for (size_t j = 0; j < dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << dims[j];
      }
      err << " at index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    const auto& dims = get_output_dims(i);
    if (dims.empty()) {
      err << "layer \"" << get_name() << "\" has "
          << "uninitialized output tensor dimensions "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
    if (std::any_of(dims.begin(), dims.end(), [](int d) { return d <= 0; })) {
      err << "layer \"" << get_name() << "\" has invalid "
          << "output tensor dimensions (";
      for (size_t j = 0; j < dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << dims[j];
      }
      err << " at index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
}

weights const& Layer::get_weights(size_t idx) const
{
  if (idx >= m_weights.size()) {
    LBANN_ERROR("attempted to access invalid weights object of ",
                get_type(),
                " layer \"",
                get_name(),
                "\" ",
                "(requested index ",
                idx,
                ", but there are ",
                num_weights(),
                " weights objects)");
  }
  const auto w = m_weights[idx].lock().get();
  if (w == nullptr) {
    LBANN_ERROR(get_type(),
                " layer \"",
                get_name(),
                "\"",
                "has an invalid reference to weights ",
                idx);
  }
  return *w;
}

weights& Layer::get_weights(size_t idx)
{
  return const_cast<weights&>(
    static_cast<Layer const&>(*this).get_weights(idx));
}

void Layer::add_as_gradient_source()
{
  for (size_t i = 0; i < num_weights(); ++i) {
    auto& w = get_weights(i);
    auto* opt = w.get_optimizer();
    if (opt != nullptr) {
      opt->add_gradient_source(this);
    }
  }
}

void Layer::remove_as_gradient_source()
{
  for (size_t i = 0; i < num_weights(); ++i) {
    auto& w = get_weights(i);
    auto* opt = w.get_optimizer();
    if (opt != nullptr) {
      opt->remove_gradient_source(this);
    }
  }
}

void Layer::back_prop()
{
  // This bit is preprocessed out since the LBANN_CALIPER macro
  // won't help us out here.
#ifdef LBANN_HAS_CALIPER
  auto const scope_name = this->get_type() + "_layer:back_prop";
  LBANN_CALIPER_MARK_SCOPE(scope_name.c_str());
#endif

  allocate_new_gradients_();
  back_prop_impl_();
  propagate_error_signals_to_parents_();
  clear_prev_error_signals_();

  // Release the now-unnecessary full weight views
  for (size_t i = 0; i < this->num_weights(); ++i) {
    this->get_weights(i).release_full_weights();
  }
}

void Layer::write_proto(lbann_data::Layer& proto) const
{
  proto.Clear();
  proto.set_name(get_name());
  for (auto const* parent : this->get_parent_layers()) {
    proto.add_parents(parent->get_name());
  }
  for (auto const* child : this->get_child_layers()) {
    proto.add_children(child->get_name());
  }
  for (size_t ii = 0; ii < this->num_weights(); ii++)
    proto.add_weights(this->get_weights(ii).get_name());

  proto.set_device_allocation(to_string(this->get_device_allocation()));
  proto.set_data_layout(to_string(this->get_data_layout()));
  if (this->get_hint_layer())
    proto.set_hint_layer(this->get_hint_layer()->get_name());
  // FIXME(KLG): Ignore for now. (Tom's problem)
  // proto.set_parallel_strategy();

  this->write_specific_proto(proto);
}

#ifdef LBANN_HAS_ONNX
void Layer::fill_onnx_node(onnx::GraphProto& graph) const
{
  auto* node = graph.add_node();
  for (auto const* parent : this->get_parent_layers()) {
    size_t idx = parent->find_child_layer_index(*this);
    node->add_input(parent->get_name() + "_" + std::to_string(idx));
  }
  for (size_t ii = 0; ii < this->num_weights(); ii++)
    node->add_input(this->get_weights(ii).get_name());

  for (auto const* child : this->get_child_layers()) {
    size_t idx = this->find_child_layer_index(*child);
    node->add_output(this->get_name() + "_" + std::to_string(idx));
  }
  node->set_name(this->get_name());
  node->set_op_type(this->get_onnx_op_type());
  node->set_domain("");
  node->set_doc_string(this->get_type());
}

std::string Layer::get_onnx_op_type() const
{
  LBANN_ERROR("ONNX export is not supported for ",
              this->get_type(),
              " layer \"",
              this->get_name(),
              "\"");
  return "";
}
#endif // LBANN_HAS_ONNX

const Layer& Layer::get_parent_layer(size_t index) const
{
  if (index >= m_parent_layers.size()) {
    LBANN_ERROR("attempted to access invalid parent layer of ",
                get_type(),
                " layer \"",
                get_name(),
                "\" ",
                "(requested index ",
                index,
                ", but there are ",
                m_parent_layers.size(),
                " parents)");
  }
  const auto l = m_parent_layers[index].lock().get();
  if (l == nullptr) {
    LBANN_ERROR(get_type(),
                " layer \"",
                get_name(),
                "\"",
                "has an invalid reference to parent layer ",
                index);
  }
  return *l;
}

const Layer& Layer::get_child_layer(size_t index) const
{
  if (index >= m_child_layers.size()) {
    LBANN_ERROR("attempted to access invalid child layer of ",
                get_type(),
                " layer \"",
                get_name(),
                "\" ",
                "(requested index ",
                index,
                ", but there are ",
                m_child_layers.size(),
                " children)");
  }
  const auto l = m_child_layers[index].lock().get();
  if (l == nullptr) {
    LBANN_ERROR(get_type(),
                " layer \"",
                get_name(),
                "\"",
                "has an invalid reference to child layer ",
                index);
  }
  return *l;
}

std::vector<const Layer*> Layer::get_parent_layers() const
{
  std::vector<const Layer*> list;
  for (int i = 0; i < get_num_parents(); ++i) {
    list.push_back(&get_parent_layer(i));
  }
  return list;
}

std::vector<const Layer*> Layer::get_child_layers() const
{
  std::vector<const Layer*> list;
  for (int i = 0; i < get_num_children(); ++i) {
    list.push_back(&get_child_layer(i));
  }
  return list;
}

size_t Layer::find_parent_layer_index(const Layer& l) const
{
  for (int i = 0; i < get_num_parents(); ++i) {
    if (&get_parent_layer(i) == &l) {
      return i;
    }
  }
  LBANN_ERROR(l.get_type(),
              " layer \"",
              l.get_name(),
              "\" ",
              "is not a parent layer of ",
              this->get_type(),
              " layer \"",
              this->get_name(),
              "\"");
  return get_num_parents();
}

size_t Layer::find_child_layer_index(const Layer& l) const
{
  for (int i = 0; i < get_num_children(); ++i) {
    if (&get_child_layer(i) == &l) {
      return i;
    }
  }
  LBANN_ERROR(l.get_type(),
              " layer \"",
              l.get_name(),
              "\" ",
              "is not a child layer of ",
              this->get_type(),
              " layer \"",
              this->get_name(),
              "\"");
  return get_num_children();
}

void Layer::add_parent_layer(ViewingLayerPtr l)
{
  const auto* l_ptr = l.lock().get();
  if (l_ptr == nullptr || l_ptr == this) {
    return;
  }
  for (int i = 0; i < get_num_parents(); ++i) {
    if (l_ptr == &get_parent_layer(i)) {
      return;
    }
  }
  m_parent_layers.emplace_back(std::move(l));
}

void Layer::add_child_layer(ViewingLayerPtr l)
{
  const auto* l_ptr = l.lock().get();
  if (l_ptr == nullptr || l_ptr == this) {
    return;
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (l_ptr == &get_child_layer(i)) {
      return;
    }
  }
  m_child_layers.emplace_back(std::move(l));
}

void Layer::replace_parent_layer(ViewingLayerPtr l, size_t index)
{
  if (l.expired()) {
    LBANN_ERROR("attempted to replace parent ",
                index,
                " of ",
                get_type(),
                " layer \"",
                get_name(),
                "\" ",
                "with an invalid layer pointer");
  }
  if (index >= m_parent_layers.size()) {
    LBANN_ERROR("attempted to replace parent ",
                index,
                " of ",
                get_type(),
                " layer \"",
                get_name(),
                "\", ",
                "which only has ",
                m_parent_layers.size(),
                " parents");
  }
  m_parent_layers[index] = std::move(l);
}

void Layer::replace_child_layer(ViewingLayerPtr l, size_t index)
{
  if (l.expired()) {
    LBANN_ERROR("attempted to replace child ",
                index,
                " of ",
                get_type(),
                " layer \"",
                get_name(),
                "\" ",
                "with an invalid layer pointer");
  }
  if (index >= m_child_layers.size()) {
    LBANN_ERROR("attempted to replace child ",
                index,
                " of ",
                get_type(),
                " layer \"",
                get_name(),
                "\", ",
                "which only has ",
                m_child_layers.size(),
                " children");
  }
  m_child_layers[index] = std::move(l);
}

ViewingLayerPtr Layer::get_parent_layer_pointer(size_t index) const
{
  if (index >= m_parent_layers.size()) {
    LBANN_ERROR("attempted to get pointer to parent ",
                index,
                " of ",
                get_type(),
                " layer \"",
                get_name(),
                "\", ",
                "which only has ",
                m_parent_layers.size(),
                " parents");
  }
  return m_parent_layers[index];
}

ViewingLayerPtr Layer::get_child_layer_pointer(size_t index) const
{
  if (index >= m_child_layers.size()) {
    LBANN_ERROR("attempted to get pointer to child ",
                index,
                " of ",
                get_type(),
                " layer \"",
                get_name(),
                "\", ",
                "which only has ",
                m_child_layers.size(),
                " children");
  }
  return m_child_layers[index];
}

std::vector<ViewingLayerPtr> Layer::get_layer_pointers()
{
  std::vector<ViewingLayerPtr> layers;
  for (const auto& l : m_parent_layers) {
    layers.push_back(l);
  }
  for (const auto& l : m_child_layers) {
    layers.push_back(l);
  }
  layers.push_back(m_hint_layer);
  return layers;
}

void Layer::set_layer_pointers(std::vector<ViewingLayerPtr> layers)
{
  const size_t expected_size =
    (m_parent_layers.size() + m_child_layers.size() + 1);
  if (layers.size() != expected_size) {
    LBANN_ERROR(
      "attempted to set layer pointers with an invalid number of pointers");
  }
  size_t pos = 0;
  for (auto& parent : m_parent_layers) {
    parent = layers[pos];
    pos++;
  }
  for (auto& child : m_child_layers) {
    child = layers[pos];
    pos++;
  }
  m_hint_layer = layers[pos];
  pos++;
}

#ifdef LBANN_HAS_DISTCONV
void Layer::prepare_distconv()
{
  if (distconv_enabled()) {
    setup_distconv_adapter();
  }
}

bool Layer::distconv_enabled() const
{

  // Return immediately if distconv support is known
  if (m_distconv_enabled_set) {
    return m_distconv_enabled;
  }

  // Check if distconv is disabled in arguments
  auto const& arg_parser = global_argument_parser();
  if (arg_parser.get<bool>(LBANN_OPTION_DISABLE_DISTCONV)) {
    m_distconv_enabled = false;
    m_distconv_enabled_set = true;
    return false;
  }

  // Check if distconv is enabled
  const auto& ps = get_parallel_strategy();
  ParallelStrategy default_zero_ps;
  if (ps == default_zero_ps || ps.enable_subgraph) {
    // Distconv is disabled if no parallel strategy is defined or if
    // sub-graph parallelism is enabled
    m_distconv_enabled = false;
  }
  else {
    m_distconv_enabled = is_distconv_supported();
  }
  m_distconv_enabled_set = true;
  return m_distconv_enabled;
}

bool Layer::keep_original_inputs(int index) const
{
  return !(distconv_enabled() &&
           !get_distconv_adapter().parent_copy_required(index));
}

bool Layer::keep_original_outputs(int index) const
{
  return !(distconv_enabled() &&
           !get_distconv_adapter().child_copy_required(index));
}

bool Layer::keep_original_gradient_wrt_outputs(int index) const
{
  return keep_original_outputs(index);
}

bool Layer::keep_original_gradient_wrt_inputs(int index) const
{
  return keep_original_inputs(index);
}

distconv_adapter& Layer::get_distconv_adapter()
{
  return const_cast<distconv_adapter&>(
    static_cast<const Layer&>(*this).get_distconv_adapter());
}

const distconv_adapter& Layer::get_distconv_adapter() const
{
  if (m_dc == nullptr) {
    LBANN_ERROR("Trying to access distconv adapter for layer, ",
                get_name(),
                ", without setting up");
  }
  return *m_dc;
}
#endif // LBANN_HAS_DISTCONV

} // namespace lbann
