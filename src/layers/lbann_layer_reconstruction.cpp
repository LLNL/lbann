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

#include "lbann/layers/lbann_layer_reconstruction.hpp"
#include "lbann/lbann_Elemental_extensions.h"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include "lbann/utils/lbann_random.hpp"

using namespace std;
using namespace El;

lbann::reconstruction_layer::reconstruction_layer(data_layout data_dist, size_t index,lbann_comm *comm,
    optimizer *opt,/*needed?*/
    const uint miniBatchSize,
    Layer *original_layer,
    activation_type activation,
    const weight_initialization init)
  :  target_layer(data_dist, comm, miniBatchSize, {}, false),m_original_layer(original_layer),
m_weight_initialization(init) {

  m_type = layer_type::reconstruction;
  Index = index;
  NumNeurons = original_layer->NumNeurons;
  this->m_optimizer = opt; // Manually assign the optimizer since target layers normally set this to NULL
  aggregate_cost = 0.0;
  num_forwardprop_steps = 0;
  // Initialize activation function
  m_activation_fn = new_activation(activation);
  // Done in base layer constructor
  /*switch(data_dist) {
    case data_layout::MODEL_PARALLEL:
      initialize_model_parallel_distribution();  base layer
      break;
    case data_layout::DATA_PARALLEL:
      initialize_data_parallel_distribution();
      break;
    default:
      throw lbann_exception(std::string{} + __FILE__ + " " +
                            std::to_string(__LINE__) +
                            "Invalid data layout selected");
  }*/
}

void lbann::reconstruction_layer::setup(int num_prev_neurons) {
  target_layer::setup(num_prev_neurons);
  Layer::setup(num_prev_neurons);

  // Initialize weight-bias matrix
  Zeros(*m_weights, NumNeurons, num_prev_neurons);

  // Initialize weights
  initialize_matrix(*m_weights, m_weight_initialization, num_prev_neurons, NumNeurons);

  // Initialize other matrices
  Zeros(*m_error_signal, num_prev_neurons, m_mini_batch_size); // m_error_signal holds the product of m_weights^T * m_prev_error_signal
  Zeros(*m_activations, NumNeurons, m_mini_batch_size); //clear up m_activations before copying fp_input to it
  Zeros(*m_weights_gradient, NumNeurons,num_prev_neurons); //clear up before filling with new results
  Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size); //clear up before filling with new results
  Zeros(*m_prev_activations, num_prev_neurons, m_mini_batch_size);

  // Initialize optimizer
  if(m_optimizer != NULL) {
    m_optimizer->setup(m_weights);
  }

}

void lbann::reconstruction_layer::fp_linearity() {
  //m_activations is linear transformation of m_weights * m_prev_activations^T
  Gemm(NORMAL, NORMAL, (DataType) 1., *m_weights, *m_prev_activations_v, (DataType) 0.0, *m_activations_v);

  int64_t curr_mini_batch_size = neural_network_model->get_current_mini_batch_size();
  DistMat original_layer_act_v;
  //view of original layer
  View(original_layer_act_v,*(m_original_layer->m_activations),IR(0,m_original_layer->m_activations->Height()),IR(0,curr_mini_batch_size));
  // Compute cost will be sum of squared error of fp_input (linearly transformed to m_activations)
  // and original layer fp_input/original input
  DataType avg_error = neural_network_model->obj_fn->compute_obj_fn(*m_activations_v, original_layer_act_v);
  aggregate_cost += avg_error;
  num_forwardprop_steps++;
}

void lbann::reconstruction_layer::bp_linearity() {

  // delta = (activation - y)
  // delta_w = delta * activation_prev^T

  int64_t curr_mini_batch_size = neural_network_model->get_current_mini_batch_size();
  DistMat original_layer_act_v;

  //view of original layer
  View(original_layer_act_v,*(m_original_layer->m_activations),IR(0,m_original_layer->m_activations->Height()),IR(0,curr_mini_batch_size));

  // Compute error signal
  neural_network_model->obj_fn->compute_obj_fn_derivative(m_prev_layer_type, *m_activations_v, original_layer_act_v,*m_prev_error_signal_v);

  //m_prev_error_signal_v is the error computed by objective function
  //is really not previous, but computed in this layer
  //@todo: rename as obj_error_signal

  // Compute the partial delta update for the next lower layer
  Gemm(TRANSPOSE, NORMAL, DataType(1), *m_weights, *m_prev_error_signal_v, DataType(0), *m_error_signal_v);

  // Compute update for activation weights
  Gemm(NORMAL, TRANSPOSE, DataType(1)/get_effective_minibatch_size(), *m_prev_error_signal_v,
       *m_prev_activations_v,DataType(0), *m_weights_gradient);
}


execution_mode lbann::reconstruction_layer::get_execution_mode() {
  return m_execution_mode;
}

bool lbann::reconstruction_layer::update() {
  double start = get_time();
  Layer::update();
  if(m_execution_mode == execution_mode::training) {
    m_optimizer->update(m_weights_gradient);
  }
  update_time += get_time() - start;
  return true;
}

void lbann::reconstruction_layer::summarize(lbann_summary& summarizer, int64_t step) {
  Layer::summarize(summarizer, step);
  std::string tag = "layer" + std::to_string(static_cast<long long>(Index))
                    + "/ReconstructionCost";
  summarizer.reduce_scalar(tag, average_cost(), step);
}

void lbann::reconstruction_layer::epoch_print() const {
  double avg_cost = average_cost();
  if (comm->am_world_master()) {
    std::vector<double> avg_costs(comm->get_num_models());
    comm->intermodel_gather(avg_cost, avg_costs);
    for (size_t i = 0; i < avg_costs.size(); ++i) {
      std::cout << "model " << i << " average reconstruction cost: " << avg_costs[i] << std::endl;
    }
  } else {
    comm->intermodel_gather(avg_cost, comm->get_world_master());
  }
}

void lbann::reconstruction_layer::epoch_reset() {
  Layer::epoch_reset();
  reset_cost();
}

void lbann::reconstruction_layer::reset_cost() {
  aggregate_cost = 0.0;
  num_forwardprop_steps = 0;
}

DataType lbann::reconstruction_layer::average_cost() const {
  return aggregate_cost / num_forwardprop_steps;
}

void lbann::reconstruction_layer::fp_nonlinearity() {
  // Forward propagation
  m_activation_fn->forwardProp(*m_activations_v);
}

void lbann::reconstruction_layer::bp_nonlinearity() {
  // Backward propagation
  m_activation_fn->backwardProp(*m_weighted_sum_v);
  if (m_activation_type != activation_type::ID) {
    Hadamard(*m_prev_error_signal_v, *m_weighted_sum_v, *m_prev_error_signal_v);
  }
}
