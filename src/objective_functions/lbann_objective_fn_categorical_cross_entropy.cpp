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

#include "lbann/objective_functions/lbann_objective_fn_categorical_cross_entropy.hpp"
#include <sys/types.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::objective_functions::categorical_cross_entropy::categorical_cross_entropy(lbann_comm* comm)
  : objective_fn("categorical_cross_entropy"), 
    m_log_predictions(comm->get_model_grid()),
    m_log_predictions_v(comm->get_model_grid()),
    m_cross_entropy_cost(comm->get_model_grid()),
    m_cross_entropy_cost_v(comm->get_model_grid()),
    m_minibatch_cost(comm->get_model_grid()),
    m_minibatch_cost_v(comm->get_model_grid())
{
  this->type = obj_fn_type::categorical_cross_entropy;
}

lbann::objective_functions::categorical_cross_entropy::~categorical_cross_entropy() {
  m_log_predictions.Empty();
  m_log_predictions_v.Empty();
  m_cross_entropy_cost.Empty();
  m_cross_entropy_cost_v.Empty();
  m_minibatch_cost.Empty();
  m_minibatch_cost_v.Empty();
}

void lbann::objective_functions::categorical_cross_entropy::setup(int num_neurons, int mini_batch_size) {
  Zeros(m_log_predictions, num_neurons, mini_batch_size);
  Zeros(m_cross_entropy_cost, num_neurons, mini_batch_size);
  Zeros(m_minibatch_cost, mini_batch_size, 1);
}

void lbann::objective_functions::categorical_cross_entropy::fp_set_std_matrix_view(int64_t cur_mini_batch_size) {
  // Set the view based on the size of the current mini-batch
  View(m_log_predictions_v, m_log_predictions, IR(0, m_log_predictions.Height()), IR(0, cur_mini_batch_size));
  View(m_cross_entropy_cost_v, m_cross_entropy_cost, IR(0, m_cross_entropy_cost.Height()), IR(0, cur_mini_batch_size));
  // Note that these matrices are transposed (column sum matrices) and thus the mini-batch size effects the number of rows, not columns
  View(m_minibatch_cost_v, m_minibatch_cost, IR(0, cur_mini_batch_size), IR(0, m_minibatch_cost.Width()));
}

/// Compute the cross-entropy cost function - comparing the activations from the previous layer and the ground truth (activations of this layer)
/// cost=-1/m*(sum(sum(groundTruth.*log(a3))))
/// predictions_v - a.k.a. coding_dist - coding distribution (e.g. prev_activations)
/// groundtruth_v - a.k.a. true_dist - true distribution (e.g. activations)
double lbann::objective_functions::categorical_cross_entropy::compute_categorical_cross_entropy(ElMat &predictions_v, ElMat &groundtruth_v) {
    DataType avg_error = 0.0, total_error = 0.0;
    int64_t cur_mini_batch_size = groundtruth_v.Width();

    Copy(predictions_v, m_log_predictions_v);
    EntrywiseMap(m_log_predictions_v, (std::function<DataType(const DataType&)>)([](const DataType& z)->DataType{return log(z);}));

    Hadamard(groundtruth_v, m_log_predictions_v, m_cross_entropy_cost_v);
    Zeros(m_minibatch_cost_v, cur_mini_batch_size, 1);
    ColumnSum(m_cross_entropy_cost_v, m_minibatch_cost_v);

    // Sum the local, total error
    const Int local_height = m_minibatch_cost_v.LocalHeight();
    for(int r = 0; r < local_height; r++) {
      total_error += m_minibatch_cost_v.GetLocal(r, 0);
    }
    total_error = mpi::AllReduce(total_error, m_minibatch_cost_v.DistComm());

    return total_error;
}

/// Compute the average categorical cross entropy over the mini-batch
double lbann::objective_functions::categorical_cross_entropy::compute_obj_fn(ElMat &predictions_v, ElMat &groundtruth_v) {
    double avg_error = 0.0, total_error = 0.0;
    int64_t cur_mini_batch_size = groundtruth_v.Width();

    total_error = compute_categorical_cross_entropy(predictions_v, groundtruth_v);

    avg_error = -1.0 * total_error / cur_mini_batch_size;

    return avg_error;
}

void lbann::objective_functions::categorical_cross_entropy::compute_obj_fn_derivative(ElMat &predictions_v, ElMat &groundtruth_v, ElMat &error_signal_v) {
  /// Compute the error between the target values and the previous layer's activations
  /// Copy the results to the m_error_signal variable for access by the next lower layer
  Copy(predictions_v, error_signal_v); // delta = (activation - y)
  Axpy(-1., groundtruth_v, error_signal_v); // Per-neuron error

  return;
}
