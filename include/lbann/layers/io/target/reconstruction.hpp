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

#ifndef LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED
#define LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/layers/io/target/generic_target_layer.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include "lbann/utils/random.hpp"

namespace lbann {
template <data_layout T_layout, El::Device Dev>
class reconstruction_layer : public generic_target_layer {
 private:

  /** Original layer to reconstruct. */
  Layer *m_original_layer;

 public:
  reconstruction_layer(lbann_comm *comm,
                       Layer *original_layer)
    :  generic_target_layer(comm),
       m_original_layer(original_layer) {}

  reconstruction_layer* copy() const override {
    throw lbann_exception("reconstruction_layer can't be copied");
    return nullptr;
  }

  std::string get_type() const override { return "reconstruction"; }

  std::string get_description() const override {
    return std::string{} + " reconstruction_layer " +
                           " original: " + m_original_layer->get_name() +
                           " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  /** Set original layer. */
  void set_original_layer(Layer *original_layer) {
    m_original_layer = original_layer;
  }

  void setup_dims() override {
    generic_target_layer::setup_dims();
    this->m_neuron_dims = m_original_layer->get_neuron_dims();
    this->m_num_neuron_dims = m_original_layer->get_num_neuron_dims();
    this->m_num_neurons = m_original_layer->get_num_neurons();
    if(this->m_num_neurons != this->m_num_prev_neurons) {
      throw lbann_exception("reconstruction_layer: original layer ("
                            + std::to_string(this->m_num_neurons)
                            + ") and reconstruction layer ("
                            + std::to_string(this->m_num_prev_neurons)
                            +") do not have the same number of neurons");
    }
  }

 protected:

  void fp_compute() override {
    El::Copy(m_original_layer->get_activations(), *m_ground_truth);
  }

  void bp_compute() override {}

 public:

  void summarize_stats(lbann_summary& summarizer, int step) override {
    std::string tag = this->m_name + "/ReconstructionCost";
    execution_mode mode = this->m_model->get_execution_mode();
    summarizer.reduce_scalar(tag, this->m_model->get_objective_function()->get_mean_value(mode), step);
    // Skip target layer (for now).
    //    io_layer::summarize_stats(summarizer, step);
  }

  std::vector<Layer*> get_layer_pointers() override {
    std::vector<Layer*> layers = generic_target_layer::get_layer_pointers();
    layers.push_back(m_original_layer);
    return layers;
  }

  void set_layer_pointers(std::vector<Layer*> layers) override {
    m_original_layer = layers.back();
    layers.pop_back();
    generic_target_layer::set_layer_pointers(layers);
  }

};

}

#endif  // LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED
