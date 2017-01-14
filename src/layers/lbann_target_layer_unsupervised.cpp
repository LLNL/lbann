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

#include "lbann/layers/lbann_target_layer_unsupervised.hpp"
#include "lbann/lbann_Elemental_extensions.h"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include "lbann/utils/lbann_random.hpp"

using namespace std;
using namespace El;

lbann::target_layer_unsupervised::target_layer_unsupervised(size_t index,lbann_comm* comm,
                                                            Optimizer* optimizer,/*needed?*/
                                                              const uint miniBatchSize,
                                                              Layer* original_layer,
                                                              const weight_initialization init)
  :  target_layer(comm, miniBatchSize, {}, false),m_original_layer(original_layer),
     m_weight_initialization(init)
{

  Index = index;
  NumNeurons = original_layer->NumNeurons;
  this->optimizer = optimizer; // Manually assign the optimizer since target layers normally set this to NULL
  aggregate_cost = 0.0;
  num_forwardprop_steps = 0;
}

void lbann::target_layer_unsupervised::setup(int num_prev_neurons) {
  target_layer::setup(num_prev_neurons);
  Layer::setup(num_prev_neurons);
  if(optimizer != NULL) {
    optimizer->setup(num_prev_neurons, NumNeurons);
  }
  // Initialize weight-bias matrix
  Zeros(*m_weights, NumNeurons, num_prev_neurons);

  // Initialize weights
  DistMat weights;
  View(weights, *m_weights, IR(0,NumNeurons), IR(0,num_prev_neurons));
  switch(m_weight_initialization) {
  case weight_initialization::uniform:
      uniform_fill(weights, weights.Height(), weights.Width(),
                   DataType(0), DataType(1));
      break;
  case weight_initialization::normal:
      gaussian_fill(weights, weights.Height(), weights.Width(),
                    DataType(0), DataType(1));
      break;
  case weight_initialization::glorot_normal: {
      const DataType var = 2.0 / (num_prev_neurons + NumNeurons);
      gaussian_fill(weights, weights.Height(), weights.Width(),
                    DataType(0), sqrt(var));
      break;
  }
  case weight_initialization::glorot_uniform: {
      const DataType var = 2.0 / (num_prev_neurons + NumNeurons);
      uniform_fill(weights, weights.Height(), weights.Width(),
                   DataType(0), sqrt(3*var));
      break;
  }
  case weight_initialization::he_normal: {
      const DataType var = 1.0 / num_prev_neurons;
      gaussian_fill(weights, weights.Height(), weights.Width(),
                    DataType(0), sqrt(var));
      break;
  }
    case weight_initialization::he_uniform: {
      const DataType var = 1.0 / num_prev_neurons;
      uniform_fill(weights, weights.Height(), weights.Width(),
                   DataType(0), sqrt(3*var));
      break;
  }
    case weight_initialization::zero: // Zero initialization is default
    default:
      Zero(weights);
      break;
  }

  // Initialize other matrices
  Zeros(*m_error_signal, num_prev_neurons, m_mini_batch_size); // m_error_signal holds the product of m_weights^T * m_prev_error_signal
  Zeros(*m_activations, NumNeurons, m_mini_batch_size); //clear up m_activations before copying fp_input to it
  Zeros(*m_weights_gradient, NumNeurons,num_prev_neurons); //clear up before filling with new results
  Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size); //clear up before filling with new results
  Zeros(*m_prev_activations, num_prev_neurons, m_mini_batch_size);

}

void lbann::target_layer_unsupervised::fp_linearity()
{
  //m_activations is linear transformation of m_weights * m_prev_activations^T
  Gemm(NORMAL, NORMAL, (DataType) 1., *m_weights, *m_prev_activations_v, (DataType) 0.0, *m_activations_v);

  int64_t curr_mini_batch_size = neural_network_model->get_current_mini_batch_size();
  DistMatrixReadProxy<DataType,DataType,MC,MR> DsNextProxy(*m_original_layer->m_activations);
  DistMat& DsNext = DsNextProxy.Get();
  DistMat DsNext_v;
  View(DsNext_v, DsNext, IR(0, DsNext.Height()), IR(0, curr_mini_batch_size));
  //DsNext is proxy of original layer
  // Compute cost will be sum of squared error of fp_input (linearly transformed to m_activations)
  // and original layer fp_input/original input (DsNext)
  DataType avg_error = neural_network_model->obj_fn->compute_obj_fn(*m_activations_v, DsNext_v);
  aggregate_cost += avg_error;
  num_forwardprop_steps++;
}

void lbann::target_layer_unsupervised::bp_linearity()
{
  DistMatrixReadProxy<DataType,DataType,MC,MR> DsNextProxy(*m_original_layer->m_activations);
  DistMat& DsNext = DsNextProxy.Get();
  // delta = (activation - y)
  // delta_w = delta * activation_prev^T
  //@todo: Optimize (may be we dont need this double copy)
  //Activation in this layer is same as linear transformation of its input, no nonlinearity
  //@todo: Optimize (check that may be we dont need this double copy)

  int64_t curr_mini_batch_size = neural_network_model->get_current_mini_batch_size();
  DistMat DsNext_v;
  View(DsNext_v, DsNext, IR(0, DsNext.Height()), IR(0, curr_mini_batch_size));
  Copy(*m_activations_v, *m_prev_error_signal_v);
  Axpy(-1., DsNext_v, *m_prev_error_signal_v); // Per-neuron error
  // Compute the partial delta update for the next lower layer
  Gemm(TRANSPOSE, NORMAL, (DataType) 1., *m_weights, *m_prev_error_signal_v, (DataType) 0., *m_error_signal_v);

  // Compute update for activation weights
  Gemm(NORMAL, TRANSPOSE, (DataType) 1.0/get_effective_minibatch_size(), *m_prev_error_signal_v,
       *m_prev_activations_v, (DataType) 0., *m_weights_gradient);
}


execution_mode lbann::target_layer_unsupervised::get_execution_mode() {
  return m_execution_mode;
}

bool lbann::target_layer_unsupervised::update()
{
  if(m_execution_mode == execution_mode::training) {
    optimizer->update_weight_bias_matrix(*m_weights_gradient, *m_weights);
  }
  return true;
}

void lbann::target_layer_unsupervised::summarize(lbann_summary& summarizer, int64_t step) {
  Layer::summarize(summarizer, step);
  std::string tag = "layer" + std::to_string(static_cast<long long>(Index))
    + "/ReconstructionCost";
  summarizer.reduce_scalar(tag, average_cost(), step);
}

void lbann::target_layer_unsupervised::epoch_print() const {
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

void lbann::target_layer_unsupervised::epoch_reset() {
  Layer::epoch_reset();
  reset_cost();
}

void lbann::target_layer_unsupervised::reset_cost() {
  aggregate_cost = 0.0;
  num_forwardprop_steps = 0;
}

DataType lbann::target_layer_unsupervised::average_cost() const {
  return aggregate_cost / num_forwardprop_steps;
}
