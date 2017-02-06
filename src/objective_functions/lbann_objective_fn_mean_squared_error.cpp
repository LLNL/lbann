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
    m_errors(comm->get_model_grid()),
    m_errors_v(comm->get_model_grid())
{
  this->type = obj_fn_type::mean_squared_error;
}

lbann::objective_functions::mean_squared_error::~mean_squared_error() {
  m_errors.Empty();
  m_errors_v.Empty();
}

void lbann::objective_functions::mean_squared_error::setup(int num_neurons, int mini_batch_size) {
  Zeros(m_errors, num_neurons, mini_batch_size);
}

void lbann::objective_functions::mean_squared_error::fp_set_std_matrix_view(int64_t cur_mini_batch_size) {
  // Set the view based on the size of the current mini-batch
  View(m_errors_v, m_errors, ALL, IR(0, cur_mini_batch_size));
}

/// Compute mean squared error
/** MSE = (predictions-groundtruth)^T (predictions-groundtruth)
 */
double lbann::objective_functions::mean_squared_error::compute_mean_squared_error(ElMat &predictions_v, ElMat &groundtruth_v) {
  Copy(predictions_v, m_errors_v);
  Axpy(DataType(-1), groundtruth_v, m_errors_v);
  return Pow(FrobeniusNorm(m_errors_v), 2);
}

/// Compute the average mean squared error over the mini-batch
double lbann::objective_functions::mean_squared_error::compute_obj_fn(ElMat &predictions_v, ElMat &groundtruth_v) {
  Int cur_mini_batch_size = groundtruth_v.Width();

  double total_error = compute_mean_squared_error(predictions_v, groundtruth_v);

  double avg_error = total_error / cur_mini_batch_size;

  return avg_error;
}

/// Compute derivative of mean squared error objective function
void lbann::objective_functions::mean_squared_error::compute_obj_fn_derivative(layer_type prev_layer_type,
                                                                               ElMat &predictions_v,
                                                                               ElMat &groundtruth_v,
                                                                               ElMat &error_signal_v) {
  Copy(predictions_v, error_signal_v);
  Axpy(DataType(-1), groundtruth_v, error_signal_v);
  Scale(DataType(2), error_signal_v);
}
