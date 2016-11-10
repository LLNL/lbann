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
//#include "lbann/utils/lbann_exception.hpp"
#include "lbann/lbann_Elemental_extensions.h"
#include <string>
#include "lbann/utils/lbann_random.hpp"
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

using namespace std;
using namespace El;

lbann::target_layer_unsupervised::target_layer_unsupervised(size_t index,lbann_comm* comm,
                                                              Optimizer* optimizer,/*needed?*/
                                                              const uint miniBatchSize,
                                                              Layer* original_layer,
                                                              const weight_initialization init)
  :  Layer(index, comm, optimizer, miniBatchSize),m_original_layer(original_layer),
     diff(comm->get_model_grid()),m_minibatch_cost(comm->get_model_grid()),
      m_weight_initialization(init)
{

  Index = index;
  NumNeurons = original_layer->NumNeurons;
  aggregate_cost = 0.0;
  num_backprop_steps = 0;
}

void lbann::target_layer_unsupervised::setup(int num_prev_neurons) {


  Layer::setup(num_prev_neurons);
  if(optimizer != NULL) {
    optimizer->setup(num_prev_neurons+1, NumNeurons);
  }

  // Initialize weight-bias matrix
  Zeros(*m_weights, NumNeurons, num_prev_neurons+1);

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
  Zeros(*m_error_signal, num_prev_neurons + 1, m_mini_batch_size); // m_error_signal holds the product of m_weights^T * m_prev_error_signal
  Zeros(*m_activations, NumNeurons, m_mini_batch_size); //clear up m_activations before copying fp_input to it
  Zeros(*m_weights_gradient, NumNeurons,num_prev_neurons + 1); //clear up before filling with new results
  Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size); //clear up before filling with new results
  Zeros(diff, NumNeurons, m_mini_batch_size); //holds squared diff of "ground truth" and computed value
  Zeros(m_minibatch_cost, m_mini_batch_size, 1); //holds sum of squared diff for each sample in a minibatch
}

///@todo update this to use the new fp_linearity framework ?? not needed??
DataType lbann::target_layer_unsupervised::forwardProp(DataType prev_WBL2NormSum) {
  DistMatrixReadProxy<DataType,DataType,MC,MR> XProxy(*fp_input);
  DistMat& X = XProxy.Get();

  //m_activations is linear transformation of m_weights * X^T
  Gemm(NORMAL, NORMAL, (DataType) 1., *m_weights, X, (DataType) 0., *m_activations);
  int num_errors = 0;
  //not used
  return num_errors;
}

void lbann::target_layer_unsupervised::backProp() {
  /// Copy the results (ground truth) to the m_error_signal variable for access by the next lower layer
  /// And for reconstruction_cost
  //@todo use m_activations for input layer and fp_input for others
  //if(m_original_layer->Index == 0) m_original_layer->fp_input = m_original_layer->m_activations;
  DistMatrixReadProxy<DataType,DataType,MC,MR> DsNextProxy(*m_original_layer->m_activations);
  DistMat& DsNext = DsNextProxy.Get();
  DistMatrixReadProxy<DataType,DataType,MC,MR> XProxy(*fp_input);
  DistMat& X = XProxy.Get();

  // delta = (activation - y)
  // delta_w = delta * activation_prev^T
  //@todo: Optimize (may be we dont need this double copy)
  //Activation in this layer is same as linear transformation of its input, no linearity
  //@todo: Optimize (check that may be we dont need this double copy)
  Copy(*m_activations, *m_prev_error_signal);
  Axpy(-1., DsNext, *m_prev_error_signal); // Per-neuron error
  // Compute the partial delta update for the next lower layer
  Gemm(TRANSPOSE, NORMAL, (DataType) 1., *m_weights, *m_prev_error_signal, (DataType) 0., *m_error_signal);
  if (m_execution_mode == execution_mode::training) {
    //DsNext is proxy of original layer
    // Compute cost will be sum of squared error of fp_input (linearly transformed to m_activations)
    // and original layer fp_input/original input (DsNext)
    DataType avg_error = this->reconstruction_cost(DsNext);
    //draw_image(DsNext,m_activations);
    aggregate_cost += avg_error;
    num_backprop_steps++;
  }

  // by divide mini-batch size
  Gemm(NORMAL, TRANSPOSE, (DataType) 1.0/get_effective_minibatch_size(), *m_prev_error_signal,
       X, (DataType) 0., *m_weights_gradient); //??
}

// Compute the cost function
//Sum of squared errors
DataType lbann::target_layer_unsupervised::reconstruction_cost(const DistMat& Y) {
  //sumerrors += ((X[m][0] - XP[m][0]) * (X[m][0] - XP[m][0]));
  DataType avg_error = 0.0, total_error = 0.0;
  //copy original layer (DsNext) to temporary diff (optimize)??
  Copy(Y, diff); //optimize, need copy?
  //compute difference between original and computed input x(Y)-x_bar(m_activations)
  Axpy(-1.,*m_activations,diff);
  //square the differences
  EntrywiseMap(diff, (std::function<DataType(DataType)>)([](DataType z)->DataType{return z*z;}));
  // sum up squared in a column (i.e., per minibatch/image)
  Zeros(m_minibatch_cost, m_mini_batch_size, 1);
  ColumnSum(diff,m_minibatch_cost);

  // Sum the local, total error
  const Int local_height = m_minibatch_cost.LocalHeight();
  for(int r = 0; r < local_height; r++) {
      total_error += m_minibatch_cost.GetLocal(r, 0);
  }
  total_error = mpi::AllReduce(total_error, m_minibatch_cost.DistComm());

  //avg_error = -1.0 * total_error / m_mini_batch_size;
  avg_error = total_error / m_mini_batch_size;
  return avg_error;
}

//draw image here for debugging original = DsNext, computed = m_activations
//void draw_image(const DistMat& original, DistMat& computed)


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
      std::cout << "Model " << i << " average reconstruction cost: " << avg_costs[i] <<
        std::endl;
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
  num_backprop_steps = 0;
}

DataType lbann::target_layer_unsupervised::average_cost() const {
  return aggregate_cost / num_backprop_steps;
}
