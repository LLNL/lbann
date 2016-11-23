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

lbann::categorical_cross_entropy::categorical_cross_entropy(lbann_comm* comm) 
  : objective_fn(), 
    m_activations_cost(comm->get_model_grid()), 
    m_activations_cost_v(comm->get_model_grid()), 
    m_minibatch_cost(comm->get_model_grid()) {}

lbann::categorical_cross_entropy::~categorical_cross_entropy() {
  m_activations_cost.Empty();
  m_activations_cost_v.Empty();
  m_minibatch_cost.Empty();
}

void lbann::categorical_cross_entropy::setup(int num_neurons, int mini_batch_size) {
  //  Zeros(m_activcations_cost, num_neurons, m_mini_batch_size);
  Zeros(m_activations_cost, num_neurons, mini_batch_size);
  Zeros(m_minibatch_cost, mini_batch_size, 1);
}

void lbann::categorical_cross_entropy::fp_set_std_matrix_view(int64_t cur_mini_batch_size) {
  // Set the view based on the size of the current mini-batch
  View(m_activations_cost_v, m_activations_cost, IR(0, m_activations_cost.Height()), IR(0, cur_mini_batch_size));
}

/// Compute the cross-entropy cost function - comparing the activations from the previous layer and the ground truth (activations of this layer)
/// cost=-1/m*(sum(sum(groundTruth.*log(a3))))
/// coding_dist - coding distribution (e.g. prev_activations)
/// true_dist - true distribution (e.g. activations)
DataType lbann::categorical_cross_entropy::compute_obj_fn(ElMat &prev_activations_v, ElMat &activations_v) {
    DataType avg_error = 0.0, total_error = 0.0;
    int64_t cur_mini_batch_size = activations_v.Width();

    EntrywiseMap(prev_activations_v, (std::function<DataType(DataType)>)([](DataType z)->DataType{return log(z);})); /// @todo check to see if this modifies the data of the lower layer

    Hadamard(activations_v, prev_activations_v, m_activations_cost_v);
    Zeros(m_minibatch_cost, cur_mini_batch_size, 1); // Clear the entire array
    ColumnSum(m_activations_cost_v, m_minibatch_cost);

    // Sum the local, total error
    const Int local_height = m_minibatch_cost.LocalHeight();
    for(int r = 0; r < local_height; r++) {
      total_error += m_minibatch_cost.GetLocal(r, 0);
    }
    total_error = mpi::AllReduce(total_error, m_minibatch_cost.DistComm());

    avg_error = -1.0 * total_error / cur_mini_batch_size;
    return avg_error;
}
