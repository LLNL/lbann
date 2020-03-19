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

#include "lbann/layers/distconv_adapter.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

distconv_adapter::distconv_adapter(Layer &layer)
    : m_layer(layer) {
}

Layer& distconv_adapter::layer() { return m_layer; }
const Layer& distconv_adapter::layer() const { return m_layer; }

std::string distconv_adapter::get_name() const {
  return layer().get_name();
}

int distconv_adapter::get_num_dims() const {
  // Use the dimension of either input or output data.
  auto nd = layer().get_num_parents() > 0 ? layer().get_input_dims().size() :
      layer().get_output_dims().size();
  nd += 1; // input and output dimensions do not have the sample dimension.
  if (!(nd == 4 || nd == 5)) {
    LBANN_ERROR(get_name(), ": Unsupported number of dimensions: ", nd);
  }
  return nd;
}

int distconv_adapter::get_num_spatial_dims() const {
  return get_num_dims() - 2;
}

void distconv_adapter::setup_fp_tensors(const dc::Dist &input_dist,
                                      const dc::Dist &output_dist) {
  setup_original_prev_activations();
  setup_prev_activations(input_dist);
  setup_activations(output_dist);
  setup_original_activations();
}

void distconv_adapter::setup_bp_tensors(const dc::Dist &prev_error_signals_dist,
                                        const dc::Dist &error_signals_dist) {
  setup_original_prev_error_signals();
  setup_prev_error_signals(prev_error_signals_dist);
  setup_error_signals(error_signals_dist);
  setup_original_error_signals();
}

bool distconv_adapter::parent_copy_in_required(size_t input_index) const {
  if (input_index < m_parent_copy_in_required.size()) {
    return m_parent_copy_in_required.at(input_index);
  } else {
    LBANN_ERROR("Out of range error! parent_copy_in_required size: ",
                m_parent_copy_in_required.size(),
                ", index: ", input_index);
  }
}

bool distconv_adapter::parent_shuffle_required(size_t input_index) const {
  if (input_index < m_parent_shuffle_required.size()) {
    return m_parent_shuffle_required.at(input_index);
  } else {
    LBANN_ERROR("Out of range error! parent_shuffle_required size: ",
                m_parent_shuffle_required.size(),
                ", index: ", input_index);
  }
}

bool distconv_adapter::child_copy_out_required(size_t output_index) const {
  if (output_index < m_child_copy_out_required.size()) {
    return m_child_copy_out_required.at(output_index);
  } else {
    LBANN_ERROR("Out of range error! child_copy_out_required size: ",
                m_child_copy_out_required.size(),
                ", index: ", output_index);
  }
}

bool distconv_adapter::child_shuffle_required(size_t output_index) const {
  if (output_index < m_child_shuffle_required.size()) {
    return m_child_shuffle_required.at(output_index);
  } else {
    LBANN_ERROR("Out of range error! child_shuffle_required size: ",
                m_child_shuffle_required.size(),
                ", index: ", output_index);
  }
}

void distconv_adapter::setup_inter_layer_adaptation() {
  assert_always(layer().distconv_enabled());

  const auto &ps = layer().get_parallel_strategy();
  for (const auto &p: layer().get_parent_layers()) {
    m_parent_copy_in_required.push_back(!p->distconv_enabled());
    m_parent_shuffle_required.push_back(
        (!p->distconv_enabled()) ||
        (ps != p->get_parallel_strategy()));
  }

  for (const auto &c: layer().get_child_layers()) {
    m_child_copy_out_required.push_back(!c->distconv_enabled());
    m_child_shuffle_required.push_back(
        (!c->distconv_enabled()) ||
        (ps != c->get_parallel_strategy()));
  }

  std::stringstream ss;
  std::stringstream parent_copyin_ss;
  std::stringstream parent_shuffle_ss;
  for (int i = 0; i < layer().get_num_parents(); ++i) {
    if (m_parent_copy_in_required[i]) {
      parent_copyin_ss << " " << i;
    }
    if (m_parent_shuffle_required[i]) {
      parent_shuffle_ss << " " << i;
    }
  }
  std::stringstream child_copyout_ss;
  std::stringstream child_shuffle_ss;
  for (int i = 0; i < layer().get_num_children(); ++i) {
    if (m_child_copy_out_required[i]) {
      child_copyout_ss << " " << i;
    }
    if (m_child_shuffle_required[i]) {
      child_shuffle_ss << " " << i;
    }
  }
  if (!parent_copyin_ss.str().empty()) {
    ss << " parent copyin:" << parent_copyin_ss.str() << ",";
  }
  if (!parent_shuffle_ss.str().empty()) {
    ss << " parent shuffle:" << parent_shuffle_ss.str() << ",";
  }
  if (!child_copyout_ss.str().empty()) {
    ss << " child copyout:" << child_copyout_ss.str() << ",";
  }
  if (!child_shuffle_ss.str().empty()) {
    ss << " child shuffle:" << child_shuffle_ss.str();
  }
  if (ss.str().size() > 0) {
    dc::MPIRootPrintStreamInfo() << get_name() << ":" << ss.str();
  }
}

void distconv_adapter::setup_keep_original_tensors() {
  assert_always(layer().distconv_enabled());
  bool env_set = std::getenv("DISTCONV_KEEP_ORIGINAL_TENSORS") != nullptr;
  for (auto b: m_parent_copy_in_required) {
    m_keep_original_input.push_back(env_set || b);
  }
  for (auto b: m_child_copy_out_required) {
    m_keep_original_output.push_back(env_set || b);
  }
  return;
}

bool distconv_adapter::keep_original_input(size_t input_index) const {
  if (input_index < m_keep_original_input.size()) {
    return m_keep_original_input.at(input_index);
  } else {
    LBANN_ERROR("Out of range error! m_keep_original_input size: ",
                m_keep_original_input.size(),
                ", index: ", input_index);
  }
}

bool distconv_adapter::keep_original_output(size_t output_index) const {
  if (output_index < m_keep_original_output.size()) {
    return m_keep_original_output.at(output_index);
  } else {
    LBANN_ERROR("Out of range error! m_keep_original_output size: ",
                m_keep_original_output.size(),
                ", index: ", output_index);
  }
}

bool distconv_adapter::keep_original() const {
  for (int i = 0; i < layer().get_num_parents(); ++i) {
    if (!keep_original_input(i)) return false;
  }
  for (int i = 0; i < layer().get_num_children(); ++i) {
    if (!keep_original_output(i)) return false;
  }
  return true;
}

}  // namespace lbann
