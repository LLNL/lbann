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
  Layer *m_original_layer;
  double aggregate_cost;
  long num_forwardprop_steps;
  AbsDistMat *original_layer_act_v;

 public:
  /// @todo note that the reconstruction layer used to use weight_initialization::glorot_uniform
  reconstruction_layer(int index,
                       lbann_comm *comm,
                       Layer *original_layer)
    :  target_layer(comm, {}, false), m_original_layer(original_layer) {
    // Setup the data distribution
    initialize_distributed_matrices();
    this->m_index = index;
    aggregate_cost = 0.0;
    num_forwardprop_steps = 0;
  }

  reconstruction_layer* copy() const {
    throw lbann_exception("reconstruction_layer can't be copied");
    return nullptr;
  }

  std::string get_name() const { return "reconstruction"; }

  //virtual inline void initialize_distributed_matrices();
  virtual inline void initialize_distributed_matrices() {
    target_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_dims() {
    target_layer::setup_dims();
    this->m_num_neurons = m_original_layer->get_num_neurons();
    this->m_neuron_dims = m_original_layer->get_neuron_dims();
    this->m_num_neuron_dims = m_original_layer->get_num_neuron_dims();
  }

 protected:
  void fp_set_std_matrix_view() {
    int64_t cur_mini_batch_size = this->m_neural_network_model->get_current_mini_batch_size();

    target_layer::fp_set_std_matrix_view();

    //view of original layer
    AbsDistMat& orig_acts = m_original_layer->get_activations();
    original_layer_act_v = orig_acts.Construct(orig_acts.Grid(),orig_acts.Root());
    El::View(*original_layer_act_v, orig_acts, El::ALL, El::IR(0, cur_mini_batch_size));
  }


  void fp_compute() {
     //Copy prev (decoder) activations for greedy layer wise training
    El::Copy(*this->m_prev_activations,*this->m_activations);
    // Compute cost will be sum of squared error of fp_input (linearly transformed to m_activations)
    // and original layer fp_input/original input
    this->m_neural_network_model->m_obj_fn->compute_value(*this->m_prev_activations,
                                                          *original_layer_act_v);
  }

  void bp_compute() {
    // Compute error signal
    this->m_neural_network_model->m_obj_fn->compute_gradient(*this->m_prev_activations,
                                                             *original_layer_act_v,
                                                             *this->m_error_signal_v);

    //m_prev_error_signal_v is the error computed by objective function
    //is really not previous, but computed in this layer
    //@todo: rename as obj_error_signal
  }

 public:
  //@todo: call base class
  execution_mode get_execution_mode() const {
    return this->m_execution_mode;
  }

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

  void epoch_print() const {
    double avg_cost = this->m_neural_network_model->m_obj_fn->get_mean_value();
    if (this->m_comm->am_world_master()) {
      std::vector<double> avg_costs(this->m_comm->get_num_models());
      this->m_comm->intermodel_gather(avg_cost, avg_costs);
      for (size_t i = 0; i < avg_costs.size(); ++i) {
        std::cout << "model " << i << " average reconstruction cost: " << avg_costs[i] << std::endl;
      }
    } else {
      this->m_comm->intermodel_gather(avg_cost, this->m_comm->get_world_master());
    }
  }

};

}

#endif  // LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED
