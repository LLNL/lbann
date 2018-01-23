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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/greedy_layerwise_autoencoder.hpp"
#include "lbann/layers/io/target/reconstruction.hpp"
#include "lbann/layers/learning/learning.hpp"
#include "lbann/objective_functions/loss_functions/loss_function.hpp"

namespace lbann {

greedy_layerwise_autoencoder::greedy_layerwise_autoencoder(lbann_comm *comm,
                                                           int mini_batch_size,
                                                           objective_function *obj_fn,
                                                           optimizer *default_optimizer)
  : sequential_model(comm, mini_batch_size, obj_fn, default_optimizer),
    m_phase(-1), m_num_phases(0), m_reconstruction(nullptr) {}

greedy_layerwise_autoencoder::greedy_layerwise_autoencoder(
  const greedy_layerwise_autoencoder& other)
  : sequential_model(other),
    m_phase(0),
    m_num_phases(other.m_num_phases),
    m_sections(other.m_sections),
    m_reconstruction(nullptr) {
  set_phase(other.m_phase);
}

greedy_layerwise_autoencoder& greedy_layerwise_autoencoder::operator=(
  const greedy_layerwise_autoencoder& other) {
  sequential_model::operator=(other);
  m_num_phases = other.m_num_phases;
  m_sections = other.m_sections;
  m_phase = 0;
  set_phase(other.m_phase);
  return *this;
}

greedy_layerwise_autoencoder::~greedy_layerwise_autoencoder() {
  if (m_reconstruction != nullptr) delete m_reconstruction;
}

void greedy_layerwise_autoencoder::setup_layer_topology() {
  sequential_model::setup_layer_topology();

  // Divide model into sections
  // Note: first half are encoder sections and second half are decoder
  // sections
  for (size_t i = 1; i < m_layers.size() - 1; ++i) {
    if (dynamic_cast<learning_layer*>(m_layers[i]) != nullptr) {
      m_sections.push_back(i);
    }
  }
  m_sections.push_back(m_layers.size() - 1);
  const int num_sections = m_sections.size() - 1;
  m_num_phases = (num_sections + 1) / 2;

  // Check that section input and output dimensions are valid
  for (int i = 0; i < m_num_phases; ++i) {

    // Beginning and end of encoder and decoder
    const int encoder_start = m_sections[i];
    const int encoder_end = m_sections[i+1];
    const int decoder_start = m_sections[num_sections-i-1];
    const int decoder_end = m_sections[num_sections-i];

    // Encoder input and decoder output should match
    std::vector<int> input_dims, output_dims;
    input_dims = m_layers[encoder_start]->get_prev_neuron_dims();
    output_dims = m_layers[decoder_end-1]->get_neuron_dims();
    if (input_dims != output_dims) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "input dimensions of layer " << m_layers[encoder_start]->get_name()
          << " (" << input_dims[0];
      for (size_t j = 1; j < input_dims.size(); ++i) {
        err << "x" << input_dims[j];
      }
      err << ") does not match "
          << "output dimensions of layer " << m_layers[decoder_end-1]->get_name()
          << " (" << output_dims[0];
      for (size_t j = 1; j < output_dims.size(); ++i) {
        err << "x" << output_dims[j];
      }
      err << ")";
      throw lbann_exception(err.str());
    }

    // Encoder output and decoder input should match
    output_dims = m_layers[encoder_end-1]->get_neuron_dims();
    input_dims = m_layers[decoder_start]->get_prev_neuron_dims();
    if (input_dims != output_dims) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "output dimensions of layer " << m_layers[encoder_end-1]->get_name()
          << " (" << output_dims[0];
      for (size_t j = 1; j < output_dims.size(); ++i) {
        err << "x" << output_dims[j];
      }
      err << ") does not match "
          << "input dimensions of layer " << m_layers[decoder_start]->get_name()
          << " (" << input_dims[0];
      for (size_t j = 1; j < input_dims.size(); ++i) {
        err << "x" << input_dims[j];
      }
      err << ")";
      throw lbann_exception(err.str());
    }

  }

}

void greedy_layerwise_autoencoder::train(int num_epochs) {
  do_train_begin_cbs();

  // Train each autoencoder phase on several epochs
  for (int phase = 0; phase < m_num_phases; ++phase) {
    set_phase(phase);
    for (int epoch = 0; epoch < num_epochs / m_num_phases; ++epoch) {
      if (get_terminate_training()) { goto train_end; }
      reset_mode_and_model(execution_mode::training);
      do_epoch_begin_cbs();
      while (!train_mini_batch()) {}
      evaluate(execution_mode::validation);
      m_current_epoch++;
      do_epoch_end_cbs();
      reset_epoch_statistics(execution_mode::training);
    }
  }

 train_end:
  restore_sequential_model();
  do_train_end_cbs();
}

void greedy_layerwise_autoencoder::set_phase(int phase) {

  // Restore sequential model first
  restore_sequential_model();
  m_phase = phase;
  if (phase < 0) {
    return;
  }

  // Determine encoder and decoder being trained
  const int num_sections = m_sections.size() - 1;
  const int encoder_start = m_sections[phase];
  const int encoder_end = m_sections[phase+1];
  const int decoder_start = m_sections[num_sections-phase-1];
  const int decoder_end = m_sections[num_sections-phase];
  auto& encoder_parents = m_layers[encoder_start]->get_parent_layers();
  auto& encoder_children = m_layers[encoder_end-1]->get_child_layers();
  auto& decoder_parents = m_layers[decoder_start]->get_parent_layers();
  auto& decoder_children = m_layers[decoder_end-1]->get_child_layers();

  // Initialize reconstruction layer
  if (m_reconstruction != nullptr) delete m_reconstruction;
  Layer* original_layer = m_layers[encoder_start-1];
  switch (encoder_parents[0]->get_data_layout()) {
  case data_layout::MODEL_PARALLEL:
    m_reconstruction = new reconstruction_layer<data_layout::MODEL_PARALLEL>(m_comm, original_layer);
    break;
  case data_layout::DATA_PARALLEL:
    m_reconstruction = new reconstruction_layer<data_layout::DATA_PARALLEL>(m_comm, original_layer);
    break;
  default:
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid data layout for reconstruction layer";
    throw lbann_exception(err.str());
  }
  m_reconstruction->set_name("reconstruction_phase" + std::to_string(phase));

  // Setup layer pointers to train encoder and decoder
  encoder_children[0] = m_layers[decoder_start];
  decoder_parents[0] = m_layers[encoder_end-1];
  decoder_children[0] = m_reconstruction;
  m_reconstruction->add_parent_layer(m_layers[decoder_end-1]);

  // Set objective function to reconstruction layer
  for (auto term : m_objective_function->get_terms()) {
    auto* loss = dynamic_cast<loss_function*>(term);
    if (loss != nullptr) {
      loss->set_target_layer((target_layer*) m_reconstruction);
    }
  }

}

void greedy_layerwise_autoencoder::restore_sequential_model() {

  // Return if model is already sequential
  if (m_phase < 0) {
    return;
  }
  m_phase = -1;

  // Restore sequential layer order
  for (int m_section : m_sections) {
    Layer* prev_layer = m_layers[m_section-1];
    Layer* next_layer = m_layers[m_section];
    auto& prev_layer_children = prev_layer->get_child_layers();
    auto& next_layer_parents = next_layer->get_parent_layers();
    prev_layer_children[0] = next_layer;
    next_layer_parents[0] = prev_layer;
  }

  // Delete reconstruction layer
  if (m_reconstruction != nullptr) delete m_reconstruction;
  m_reconstruction = nullptr;

  // Restore objective function to target layer
  for (auto term : m_objective_function->get_terms()) {
    auto* loss = dynamic_cast<loss_function*>(term);
    if (loss != nullptr) {
      loss->set_target_layer((target_layer*) m_layers.back());
    }
  }

}

void greedy_layerwise_autoencoder::forward_prop(execution_mode mode) {

  // Use base implementation if model is sequential
  if (m_phase < 0) {
    sequential_model::forward_prop(mode);
    return;
  }
  do_model_forward_prop_begin_cbs(mode);

  // Determine encoder and decoder to train
  const int num_sections = m_sections.size() - 1;
  const int encoder_start = m_sections[m_phase];
  const int encoder_end = m_sections[m_phase+1];
  const int decoder_start = m_sections[num_sections-m_phase-1];
  const int decoder_end = m_sections[num_sections-m_phase];

  // Forward prop on layers that are already trained
  for (int i = 0; i < encoder_start; ++i) {
    Layer* layer = m_layers[i];
    do_layer_forward_prop_begin_cbs(mode, layer);
    layer->forward_prop();
    do_layer_forward_prop_end_cbs(mode, layer);
  }

  // Forward prop on encoder being trained
  for (int i = encoder_start; i < encoder_end; ++i) {
    Layer* layer = m_layers[i];
    do_layer_forward_prop_begin_cbs(mode, layer);
    layer->forward_prop();
    do_layer_forward_prop_end_cbs(mode, layer);
  }

  // Forward prop on decoder being trained
  for (int i = decoder_start; i < decoder_end; ++i) {
    Layer* layer = m_layers[i];
    do_layer_forward_prop_begin_cbs(mode, layer);
    layer->forward_prop();
    do_layer_forward_prop_end_cbs(mode, layer);
  }

  // Forward prop on reconstruction layer
  do_layer_forward_prop_begin_cbs(mode, m_reconstruction);
  m_reconstruction->forward_prop();
  do_layer_forward_prop_end_cbs(mode, m_reconstruction);

  do_model_forward_prop_end_cbs(mode);
}

void greedy_layerwise_autoencoder::backward_prop() {

  // Use base implementation if model is sequential
  if (m_phase < 0) {
    sequential_model::backward_prop();
    return;
  }
  do_model_backward_prop_begin_cbs();

  // Determine encoder and decoder to train
  const int num_sections = m_sections.size() - 1;
  const int encoder_start = m_sections[m_phase];
  const int encoder_end = m_sections[m_phase+1];
  const int decoder_start = m_sections[num_sections-m_phase-1];
  const int decoder_end = m_sections[num_sections-m_phase];

  // Backward prop on reconstruction layer
  do_layer_backward_prop_begin_cbs(m_reconstruction);
  m_reconstruction->back_prop();
  do_layer_backward_prop_end_cbs(m_reconstruction);

  // Backward prop on decoder being trained
  for (int i = decoder_end; i < decoder_start; --i) {
    Layer* layer = m_layers[i];
    do_layer_backward_prop_begin_cbs(layer);
    layer->back_prop();
    do_layer_backward_prop_end_cbs(layer);
  }

  // Backward prop on encoder being trained
  for (int i = encoder_end; i >= encoder_start; --i) {
    Layer* layer = m_layers[i];
    do_layer_backward_prop_begin_cbs(layer);
    layer->back_prop();
    do_layer_backward_prop_end_cbs(layer);
  }

  do_model_backward_prop_end_cbs();
}

}  // namespace lbann
