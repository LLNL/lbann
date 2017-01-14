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

#include "lbann/objective_functions/lbann_objective_fn_mean_squared_error.hpp"
#include <sys/types.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::mean_squared_error::mean_squared_error(lbann_comm* comm) 
  : objective_fn(), 
    m_sum_squared_errors(comm->get_model_grid()), 
    m_sum_squared_errors_v(comm->get_model_grid()), 
    m_minibatch_cost(comm->get_model_grid()) {}

lbann::mean_squared_error::~mean_squared_error() {
  m_sum_squared_errors.Empty();
  m_sum_squared_errors_v.Empty();
  m_minibatch_cost.Empty();
}

void lbann::mean_squared_error::setup(int num_neurons, int mini_batch_size) {
  Zeros(m_sum_squared_errors, num_neurons, mini_batch_size);
  Zeros(m_minibatch_cost, mini_batch_size, 1);
}

void lbann::mean_squared_error::fp_set_std_matrix_view(int64_t cur_mini_batch_size) {
  // Set the view based on the size of the current mini-batch
  // @todo: This is buggy and not used -- remove
  // possible bug of creating view of a larger matrix (cur_mini_batch) from a smaller matrix (m_sum_sq_error.Width()) 
  //View(m_sum_squared_errors_v, m_sum_squared_errors, IR(0, m_sum_squared_errors.Height()), IR(0, cur_mini_batch_size));
}

/// Compute mean squared error
/// sumerrors += ((X[m][0] - XP[m][0]) * (X[m][0] - XP[m][0]));
DataType lbann::mean_squared_error::compute_obj_fn(ElMat &predictions_v, ElMat &groundtruth_v) {
  DataType avg_error = 0.0, total_error = 0.0;
  int64_t cur_mini_batch_size = groundtruth_v.Width();

  // copy activations from the previous layer into the temporary matrix m_sum_squared_errors
  Copy(predictions_v, m_sum_squared_errors); //optimize, need copy?
  // compute difference between original and computed input x(Y)-x_bar(m_activations)
  Axpy(-1., groundtruth_v, m_sum_squared_errors);
  //square the differences
  EntrywiseMap(m_sum_squared_errors, (std::function<DataType(DataType)>)([](DataType z)->DataType{return z*z;}));
  // sum up squared in a column (i.e., per minibatch/image)
  Zeros(m_minibatch_cost, cur_mini_batch_size, 1);
  ColumnSum(m_sum_squared_errors, m_minibatch_cost);

  // Sum the local, total error
  const Int local_height = m_minibatch_cost.LocalHeight();
  for(int r = 0; r < local_height; r++) {
      total_error += m_minibatch_cost.GetLocal(r, 0);
  }
  total_error = mpi::AllReduce(total_error, m_minibatch_cost.DistComm());

  avg_error = total_error / cur_mini_batch_size;
  return avg_error;
}
