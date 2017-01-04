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

#include "lbann/metrics/lbann_metric_mean_squared_error.hpp"
#include "lbann/models/lbann_model.hpp"

using namespace std;
using namespace El;

lbann::metrics::mean_squared_error::mean_squared_error(lbann_comm* comm)
  : metric(comm),
    internal_obj_fn(comm)
{
  this->type = metric_type::mean_squared_error;
}

lbann::metrics::mean_squared_error::~mean_squared_error() {
  internal_obj_fn.~mean_squared_error();
}

void lbann::metrics::mean_squared_error::setup(int num_neurons, int mini_batch_size) {
  metric::setup(num_neurons, mini_batch_size);
  // Setup the internal objective function
  internal_obj_fn.setup(num_neurons, mini_batch_size);
}

void lbann::metrics::mean_squared_error::fp_set_std_matrix_view(int64_t cur_mini_batch_size) {
  // Set the view based on the size of the current mini-batch
  internal_obj_fn.fp_set_std_matrix_view(cur_mini_batch_size);
}

double lbann::metrics::mean_squared_error::compute_metric(ElMat& predictions_v, ElMat& groundtruth_v) {
  double num_errors = internal_obj_fn.compute_mean_squared_error(predictions_v, groundtruth_v);
  return num_errors;
}

double lbann::metrics::mean_squared_error::report_metric(execution_mode mode) {
  statistics *stats = get_statistics(mode);
  double error_per_epoch = stats->m_error_per_epoch;
  long samples_per_epoch = stats->m_samples_per_epoch;

  double mse = error_per_epoch / samples_per_epoch;
  string score = std::to_string(mse);

  std::cout << _to_string(type) << " reporting a metric with " << error_per_epoch << " errors and " << samples_per_epoch << " samples, a mse of " << mse << " and a score of " << score << endl;
  return mse;
}
