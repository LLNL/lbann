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

lbann::objective_functions::mean_squared_error::mean_squared_error(lbann_comm* comm)
  : objective_fn("mean_squared_error"), 
    m_squared_errors(comm->get_model_grid()),
    m_squared_errors_v(comm->get_model_grid()),
    m_sum_squared_errors(comm->get_model_grid()),
    m_sum_squared_errors_v(comm->get_model_grid())
{
  this->type = obj_fn_type::mean_squared_error;
}

lbann::objective_functions::mean_squared_error::~mean_squared_error() {
  m_squared_errors.Empty();
  m_squared_errors_v.Empty();
  m_sum_squared_errors.Empty();
  m_sum_squared_errors_v.Empty();
}

void lbann::objective_functions::mean_squared_error::setup(int num_neurons, int mini_batch_size) {
  Zeros(m_squared_errors, num_neurons, mini_batch_size);
  Zeros(m_sum_squared_errors, mini_batch_size, 1);
}

void lbann::objective_functions::mean_squared_error::fp_set_std_matrix_view(int64_t cur_mini_batch_size) {
  // Set the view based on the size of the current mini-batch
  View(m_squared_errors_v, m_squared_errors, IR(0, m_squared_errors.Height()), IR(0, cur_mini_batch_size));
  // Note that these matrices are transposed (column sum matrices) and thus the mini-batch size effects the number of rows, not columns
  View(m_sum_squared_errors_v, m_sum_squared_errors, IR(0, cur_mini_batch_size), IR(0, m_sum_squared_errors.Width()));
}

/// Compute mean squared error
/// sumerrors += ((X[m][0] - XP[m][0]) * (X[m][0] - XP[m][0]));
double lbann::objective_functions::mean_squared_error::compute_mean_squared_error(ElMat &predictions_v, ElMat &groundtruth_v) {
  double avg_error = 0.0, total_error = 0.0;
  int64_t cur_mini_batch_size = groundtruth_v.Width();

  // copy activations from the previous layer into the temporary matrix m_squared_errors
  Copy(predictions_v, m_squared_errors_v); //optimize, need copy?
  // compute difference between original and computed input x(Y)-x_bar(m_activations)
  Axpy(-1., groundtruth_v, m_squared_errors_v);
  //square the differences
  EntrywiseMap(m_squared_errors_v, (std::function<DataType(const DataType&)>)([](const DataType& z)->DataType{return z*z;}));
  // sum up squared in a column (i.e., per minibatch/image)
  Zeros(m_sum_squared_errors, cur_mini_batch_size, 1); // Clear the entire array
  ColumnSum(m_squared_errors_v, m_sum_squared_errors_v);

  // Sum the local, total error
  const Int local_height = m_sum_squared_errors_v.LocalHeight();
  for(int r = 0; r < local_height; r++) {
      total_error += m_sum_squared_errors_v.GetLocal(r, 0);
  }
  total_error = mpi::AllReduce(total_error, m_sum_squared_errors_v.DistComm());
  return total_error;
}

/// Compute the average mean squared error over the mini-batch
double lbann::objective_functions::mean_squared_error::compute_obj_fn(ElMat &predictions_v, ElMat &groundtruth_v) {
  DataType avg_error = 0.0, total_error = 0.0;
  int64_t cur_mini_batch_size = groundtruth_v.Width();

  total_error = compute_mean_squared_error(predictions_v, groundtruth_v);

  avg_error = total_error / cur_mini_batch_size;

  return avg_error;
}

/// @todo this is implementing the same behavior as previously done, but needs to be corrected.
void lbann::objective_functions::mean_squared_error::compute_obj_fn_derivative(ElMat &predictions_v, ElMat &groundtruth_v, ElMat &error_signal_v) {
  /// Compute the error between the target values and the previous layer's activations
  /// Copy the results to the m_error_signal variable for access by the next lower layer
  Copy(predictions_v, error_signal_v); // delta = (activation - y)
  Axpy(-1., groundtruth_v, error_signal_v); // Per-neuron error

  return;
}
