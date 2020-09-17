////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"

#include <layers.pb.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

// Asynchronous memory transfers for input data
// Note: This introduces a race condition. It is possible for the
// input data to be modified by another layer before it is used by
// this layer.
// #define ASYNC_INPUT_MEMORY_TRANSFER

namespace lbann {

Layer::Layer(lbann_comm *comm)
  : m_comm(comm),
    m_frozen(false) {

  // Initialize layer name
  static int num_layers = 0;
  m_name = "layer" + std::to_string(num_layers);
  num_layers++;

  // Reset timing counters
  reset_counters();

}

Layer::Layer(const Layer& other) :
  m_comm(other.m_comm),
  m_parent_layers(other.m_parent_layers),
  m_child_layers(other.m_child_layers),
  m_expected_num_parent_layers(other.m_expected_num_parent_layers),
  m_expected_num_child_layers(other.m_expected_num_child_layers),
  m_model(other.m_model),
  m_frozen(other.m_frozen),
  m_fp_time(other.m_fp_time),
  m_fp_compute_time(other.m_fp_compute_time),
  m_bp_time(other.m_bp_time),
  m_bp_compute_time(other.m_bp_compute_time),
  m_update_time(other.m_update_time),
  m_name(other.m_name),
  m_weights(other.m_weights),
  m_output_dims_list(other.m_output_dims_list),
  m_hint_layer(other.m_hint_layer) {
}

Layer& Layer::operator=(const Layer& other) {

  // Shallow copies
  m_comm = other.m_comm;
  m_parent_layers = other.m_parent_layers;
  m_child_layers = other.m_child_layers;
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
  m_weights = other.m_weights;
  m_output_dims_list = other.m_output_dims_list;
  m_hint_layer = other.m_hint_layer;

  return *this;
}

description Layer::get_description() const {

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
      } else {
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
      } else {
        ss << children[i]->get_type() << " layer "
           << "\"" << children[i]->get_name() << "\"";
      }
      ss << ")";
    }
    desc.add("Output dimensions", ss.str());
  }

  // Weights
  const auto weights_list = m_weights;
  if (!weights_list.empty()) {
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < weights_list.size(); ++i) {
      ss << (i > 0 ? ", " : "");
      if (weights_list[i] == nullptr) {
        ss << "unknown weights";
      } else {
        const auto& dims = weights_list[i]->get_dims();
        ss << weights_list[i]->get_name() << " (";
        for (size_t j = 0; j < dims.size(); ++j) {
          ss << (j > 0 ? "x" : "") << dims[j];
        }
        ss << ")";
      }
    }
    desc.add("Weights", ss.str());
  }

  // Data layout
  ss.str(std::string{});
  ss.clear();
  switch (get_data_layout()) {
  case data_layout::DATA_PARALLEL:  ss << "data-parallel";  break;
  case data_layout::MODEL_PARALLEL: ss << "model-parallel"; break;
  case data_layout::invalid:
  default:
    ss << "invalid";
  }
  desc.add("Data layout", ss.str());

  // Device
  ss.str(std::string{});
  ss.clear();
  switch (get_device_allocation()) {
  case El::Device::CPU: ss << "CPU";     break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU: ss << "GPU";     break;
#endif // LBANN_HAS_GPU
  default:              ss << "unknown";
  }
  desc.add("Device", ss.str());

  // DataType
  desc.add("Data type", get_datatype_name());

  // Freeze state
  if (is_frozen()) {
    desc.add("Frozen");
  }

  return desc;
}

bool Layer::update() {
  if (m_frozen) { return true; }
  // Apply any updates.
  const auto update_compute_start = get_time();
  const auto layer_done = update_compute();
  m_update_time += get_time() - update_compute_start;
  return layer_done;
}

void Layer::reset_counters() {
  m_fp_time         = EvalType(0);
  m_fp_compute_time = EvalType(0);
  m_bp_time         = EvalType(0);
  m_bp_compute_time = EvalType(0);
  m_update_time     = EvalType(0);
}

void Layer::summarize_stats(lbann_summary& summarizer, int step) {
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
  for (auto const& w : m_weights) {
    optimizer *opt = w->get_optimizer();
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

std::vector<int> Layer::get_input_dims(int input_index) const {

  // Get parent layer
  const auto& num_inputs = get_num_parents();
  if (input_index < 0 || input_index >= num_inputs) {
    std::stringstream err;
    err << "attempted to access dimensions of invalid input tensor "
        << "in layer \"" << get_name() << "\" "
        << "(requested index " << input_index << ", but there are "
        << num_inputs << " input tensors)";
    LBANN_ERROR(err.str());
  } else if (m_parent_layers[input_index] == nullptr) {
    std::stringstream err;
    err << "layer \"" << get_name() << "\" "
        << "has a null pointer to parent layer "
        << "(index " << input_index << ")";
    LBANN_ERROR(err.str());
  }
  const auto& parent = *m_parent_layers[input_index];

  // Get dimensions of corresponding output tensor in parent layer
  const auto num_parent_outputs = parent.get_num_children();
  const int parent_output_index = parent.find_child_layer_index(this);
  if (parent_output_index >= num_parent_outputs) {
    std::stringstream err;
    err << "layer \"" << parent.get_name() << "\" is a parent of "
        << "layer \"" << get_name() << "\", but "
        << "\"" << get_name() << "\" is not a child of "
        << "\"" << parent.get_name() << "\"";
    LBANN_ERROR(err.str());
  }
  return parent.get_output_dims(parent_output_index);

}

// ===========================================================
// Tensor dimension access functions
// ===========================================================

int Layer::get_input_size(int input_index) const {
  const auto& dims = get_input_dims(input_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<int>());
  }
}

std::vector<int> Layer::get_output_dims(int output_index) const {
  const auto num_outputs = get_num_children();
  if ((int) m_output_dims_list.size() != num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of output tensor "
        << "in layer \"" << get_name() << "\" "
        << "before they are initialized";
    LBANN_ERROR(err.str());
  } else if (output_index < 0 || output_index >= num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of invalid output tensor "
        << "in layer \"" << get_name() << "\" "
        << "(requested index " << output_index << ", but there are "
        << num_outputs << " output tensors)";
    LBANN_ERROR(err.str());
  }
  return m_output_dims_list[output_index];
}

int Layer::get_output_size(int output_index) const {
  const auto& dims = get_output_dims(output_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<int>());
  }
}

void Layer::set_output_dims(std::vector<int> dims, int output_index) {
  if ((int) m_output_dims_list.size() != get_num_children()
      || (int) m_output_dims_list.size() <= output_index) {
    // Handles case where dims are set before child layers are set
    m_output_dims_list.resize(std::max(get_num_children(),
                                       output_index + 1));
  }
  m_output_dims_list[output_index] = dims;
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
void Layer::replace_weights(Layer const& other_layer) {

  auto const other_num_weights = other_layer.num_weights();
  auto const my_num_weights = this->num_weights();

  // Minimal sanity check; see longer note above.
  if (other_num_weights < my_num_weights)
    LBANN_ERROR("Expected at least ", my_num_weights, " weights in layer \"",
                other_layer.get_name(), "\" but found ", other_num_weights);

  using IdxT = typename std::decay<decltype(my_num_weights)>::type;
  for (IdxT ii = 0; ii < my_num_weights; ++ii) {
    auto const& other_layer_weights = other_layer.get_weights(ii);
    this->get_weights(ii).set_values(other_layer_weights.get_values());
  }
}

void Layer::freeze() {
  m_frozen = true;
  for(auto& w : m_weights) {
    w->freeze();
  }
}

void Layer::unfreeze() {
  m_frozen = false;
  for(auto& w : m_weights) {
    w->unfreeze();
  }
}

bool Layer::is_frozen() const {
  for(auto& w : m_weights) {
    if (w->is_frozen() != m_frozen) {
      LBANN_ERROR("layer ", get_name(), " and weight ", w->get_name(), \
                  " of it are inconsistently frozen");
    }
  }
  return m_frozen;
}

void Layer::setup(size_t max_mini_batch_size, DataReaderMetaData& dr_metadata) {
  setup_pointers();
  setup_dims(dr_metadata);
  setup_matrices(m_comm->get_trainer_grid());
#ifdef LBANN_HAS_DISTCONV
  prepare_distconv();
#endif // LBANN_HAS_DISTCONV
  setup_data(max_mini_batch_size);
  if (using_gpus()) { setup_gpu(); }
}

void Layer::setup_pointers() {
  std::stringstream err;

  // Check that the parent pointers are valid
  for (size_t i = 0; i < m_parent_layers.size(); ++i) {
    const auto* parent = m_parent_layers[i];
    if (parent == nullptr) {
      err << "layer \"" << get_name() << "\" "
          << "has a null pointer to parent layer " << i;
      LBANN_ERROR(err.str());
    }
    const auto& parent_children = parent->m_child_layers;
    if (std::find(parent_children.begin(), parent_children.end(), this)
        == parent_children.end()) {
      err << "layer \"" << parent->get_name() << "\" is a parent of "
          << "layer \"" << get_name() << "\", but "
          << "\"" << get_name() << "\" is not a child of "
          << "\"" << parent->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
  }

  // Check that the child pointers are valid
  for (size_t i = 0; i < m_child_layers.size(); ++i) {
    const auto* child = m_child_layers[i];
    if (child == nullptr) {
      err << "layer \"" << get_name() << "\" "
          << "has a null pointer to child layer " << i;
      LBANN_ERROR(err.str());
    }
    const auto& child_parents = child->m_parent_layers;
    if (std::find(child_parents.begin(), child_parents.end(), this)
        == child_parents.end()) {
      err << "layer \"" << child->get_name() << "\" is a child of "
          << "layer \"" << get_name() << "\", but "
          << "\"" << get_name() << "\" is not a parent of "
          << "\"" << child->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
  }

  // Check that the number of parents/children are valid
  if(m_expected_num_parent_layers >= 0
     && get_num_parents() != m_expected_num_parent_layers) {
    err << get_type() << " layer \"" << get_name() << "\" "
        << "expects " << m_expected_num_parent_layers << " "
        << "parent layer" << (m_expected_num_parent_layers != 1 ? "s" : "")
        << ", but found " << get_num_parents();
    if (get_num_parents() > 0) {
      err << " (";
      for (int i = 0; i < get_num_parents(); ++i) {
        err << (i > 0 ? ", " : "")
            << "\"" << m_parent_layers[i]->get_name() << "\"";
      }
      err << ")";
    }
    LBANN_ERROR(err.str());
  }
  if(m_expected_num_child_layers >= 0
     && get_num_children() != m_expected_num_child_layers) {
    err << get_type() << " layer \"" << get_name() << "\" "
        << "expects " << m_expected_num_child_layers << " "
        << "child layer" << (m_expected_num_child_layers != 1 ? "s" : "")
        << ", but found " << get_num_children();
    if (get_num_children() > 0) {
      err << " (";
      for (int i = 0; i < get_num_children(); ++i) {
        err << (i > 0 ? ", " : "")
            << "\"" << m_child_layers[i]->get_name() << "\"";
      }
      err << ")";
    }
    LBANN_ERROR(err.str());
  }

}

void Layer::setup_dims(DataReaderMetaData& dr_metadata) {
  m_output_dims_list.resize(get_num_children());
  if (m_hint_layer != nullptr) {
    const auto& hint_dims = m_hint_layer->get_output_dims();
    for (auto& output_dims : m_output_dims_list) {
      output_dims = hint_dims;
    }
  } else if (get_num_parents() > 0) {
    const auto& input_dims = get_input_dims();
    for (auto& output_dims : m_output_dims_list) {
      if (output_dims.empty()) {
        output_dims = input_dims;
      }
    }
  }
}

void Layer::check_setup() {
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
    if (std::any_of(dims.begin(), dims.end(),
                    [](int d) { return d <= 0; })) {
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
    if (std::any_of(dims.begin(), dims.end(),
                    [](int d) { return d <= 0; })) {
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

void Layer::back_prop() {
  allocate_new_gradients_();
  back_prop_impl_();
  propagate_error_signals_to_parents_();
  clear_prev_error_signals_();
}


bool Layer::save_to_checkpoint_shared(persist& p) const {
  return true;
}

bool Layer::load_from_checkpoint_shared(persist& p) {
  return true;
}

bool Layer::save_to_checkpoint_distributed(persist& p) const {
  return true;
}

bool Layer::load_from_checkpoint_distributed(persist& p) {
  return true;
}

void Layer::write_proto(lbann_data::Layer* proto) const {
  proto->Clear();
  proto->set_name(get_name());
  proto->set_type(get_type());
  if(!m_parent_layers.empty()) proto->set_bottom(m_parent_layers.front()->get_name());
  proto->set_top(get_name());
  //Add weights
  for (auto const& w : m_weights) {
    auto weight_proto = proto->add_weights_data();
    w->write_proto(weight_proto);
  }
}

std::string Layer::get_layer_names(const std::vector<const Layer*>& list) {
  std::string layer_names = ((list.size()==0u || !list[0])? "" : list[0]->get_name());

  for (size_t i=1u; i < list.size(); ++i) {
    if (list[i]) layer_names += ", " + list[i]->get_name();
  }
  return layer_names;
}

void Layer::add_parent_layer(const Layer* parent) {
  const auto parent_pos = std::find(m_parent_layers.begin(),
                                    m_parent_layers.end(),
                                    parent);
  if (parent != nullptr
      && parent != this
      && parent_pos == m_parent_layers.end()) {
    m_parent_layers.push_back(parent);
  }
}

void Layer::add_child_layer(const Layer* child) {
  const auto child_pos = std::find(m_child_layers.begin(),
                                   m_child_layers.end(),
                                   child);
  if (child != nullptr
      && child != this
      && child_pos == m_child_layers.end()) {
    m_child_layers.push_back(child);
  }
}

std::vector<Layer*> Layer::get_layer_pointers() {
  std::vector<Layer*> layers;
  for (const auto* parent: m_parent_layers) {
    layers.push_back(const_cast<Layer*>(parent));
  }
  for (const auto* child: m_child_layers) {
    layers.push_back(const_cast<Layer*>(child));
  }
  layers.push_back(const_cast<Layer*>(m_hint_layer));
  return layers;
}

void Layer::set_layer_pointers(std::vector<Layer*> layers) {
  const size_t expected_size = (m_parent_layers.size()
                                + m_child_layers.size()
                                + 1);
  if (layers.size() != expected_size) {
    LBANN_ERROR("attempted to set layer pointers with an invalid number of pointers");
  }
  size_t pos = 0;
  for (auto& parent: m_parent_layers) {
    parent = static_cast<const Layer*>(layers[pos]);
    pos++;
  }
  for (auto& child: m_child_layers) {
    child = static_cast<const Layer*>(layers[pos]);
    pos++;
  }
  m_hint_layer = layers[pos];
  pos++;
}

#ifdef LBANN_HAS_DISTCONV
void Layer::prepare_distconv() {
  if (distconv_enabled()) {
    setup_distconv_adapter();
  }
}

bool Layer::distconv_enabled() const {
  if (!m_distconv_enabled_set) {
    // Distconv is disabled if no parallel strategy is defined. When no
    // strategy is defined, the layer has the default strategy of all
    // zeros, which is invalid, thus should not be used when distconv is
    // used.
    const auto &ps = get_parallel_strategy();
    ParallelStrategy default_zero_ps;
    if (ps == default_zero_ps) {
      dc::MPIRootPrintStreamDebug()
          << "Disable " << get_name()
          << " as it does not have a parallel strategy.";
      m_distconv_enabled = false;
      m_distconv_enabled_set = true;
    }
  }

  if (!m_distconv_enabled_set) {
    // Finally, check whether a layer is supported by distconv.
    m_distconv_enabled = is_distconv_supported();
    m_distconv_enabled_set = true;
  }

  return m_distconv_enabled;
}

bool Layer::keep_original_inputs(int index) const {
  return !(distconv_enabled() && !get_distconv_adapter().parent_copy_required(index));
}

bool Layer::keep_original_outputs(int index) const {
  return !(distconv_enabled() && !get_distconv_adapter().child_copy_required(index));
}

bool Layer::keep_original_gradient_wrt_outputs(int index) const {
  return keep_original_outputs(index);
}

bool Layer::keep_original_gradient_wrt_inputs(int index) const {
  return keep_original_inputs(index);
}

distconv_adapter& Layer::get_distconv_adapter() {
  return const_cast<distconv_adapter&>(
      static_cast<const Layer&>(*this).get_distconv_adapter());
}

const distconv_adapter& Layer::get_distconv_adapter() const {
  if (m_dc == nullptr) {
    LBANN_ERROR("Trying to access distconv adapter for layer, ",
                get_name(), ", without setting up");
  }
  return *m_dc;
}
#endif // LBANN_HAS_DISTCONV

}  // namespace lbann
