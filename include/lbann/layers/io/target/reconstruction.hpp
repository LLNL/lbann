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
#include "lbann/layers/io/target/target_layer.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include "lbann/utils/random.hpp"

namespace lbann {
template <data_layout T_layout>
class reconstruction_layer : public target_layer {
 private:

  /** Original layer to reconstruct. */
  Layer *m_original_layer;

 public:
  /// @todo note that the reconstruction layer used to use weight_initialization::glorot_uniform
  reconstruction_layer(int index,
                       lbann_comm *comm,
                       Layer *original_layer)
    :  target_layer(comm, {}, false),
       m_original_layer(original_layer) {
    // Setup the data distribution
    initialize_distributed_matrices();
    this->m_index = index;
  }

  reconstruction_layer(const reconstruction_layer& other) :
    target_layer(other),
    m_original_layer(other.m_original_layer) {}

  reconstruction_layer& operator=(const reconstruction_layer& other) {
    m_original_layer = other.m_original_layer;
  }

  reconstruction_layer* copy() const {
    throw lbann_exception("reconstruction_layer can't be copied");
    return nullptr;
  }

  virtual std::string get_type() const override { return "reconstruction"; }

  //virtual inline void initialize_distributed_matrices();
  virtual inline void initialize_distributed_matrices() {
    target_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_dims() {
    target_layer::setup_dims();
    this->m_neuron_dims = m_original_layer->get_neuron_dims();
    this->m_num_neuron_dims = m_original_layer->get_num_neuron_dims();
    this->m_num_neurons = m_original_layer->get_num_neurons();
    if(this->m_num_neurons != this->m_num_prev_neurons) {
      throw lbann_exception("reconstruction_layer: original layer and reconstruction layer do not have the same number of neurons");
    }
  }

 protected:

  void fp_compute() {

    // Get activations from original layer
    m_original_layer->get_fp_output(*this->m_activations_v, this);

    // Compute and record the objective function score
    objective_functions::objective_function *obj_fn = this->m_neural_network_model->m_obj_fn;
    obj_fn->compute_value(*this->m_prev_activations,
                          *this->m_activations_v);

    // Compute metrics
    const int curr_mini_batch_size = this->m_neural_network_model->get_current_mini_batch_size();
    for (auto&& m : this->m_neural_network_model->get_metrics()) {
      double num_errors = m->compute_metric(*this->m_prev_activations, *this->m_activations_v);
      m->record_error(num_errors, curr_mini_batch_size);
    }

  }

  void bp_compute() {
    this->m_neural_network_model->m_obj_fn->compute_gradient(*this->m_prev_activations,
                                                             *this->m_activations_v,
                                                             *this->m_error_signal_v);
  }

 public:
  bool update_compute() {
    if(this->m_execution_mode == execution_mode::training) {
      double start = get_time();
      this->update_time += get_time() - start;
    }
    return true;
  }

  void summarize_stats(lbann_summary& summarizer, int step) {
    std::string tag = "layer" + std::to_string(this->m_index)
      + "/ReconstructionCost";
    summarizer.reduce_scalar(tag, this->m_neural_network_model->m_obj_fn->get_mean_value(), step);
    // Skip target layer (for now).
    io_layer::summarize_stats(summarizer, step);
  }

};

}

#endif  // LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED
