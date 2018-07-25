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

#include "lbann/objective_functions/loss_functions/mean_squared_error.hpp"

namespace lbann {

void mean_squared_error_loss::start_evaluate_compute(
  const AbsDistMat& predictions,
  const AbsDistMat& ground_truth) {
  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();

  // Matrix parameters
  const El::Int height = predictions.Height();
  const El::Int width = predictions.Width();
  const El::Int local_height = predictions_local.Height();
  const El::Int local_width = predictions_local.Width();

  // Compute sum of squared errors
  EvalType sum = EvalType(0);
  int nthreads = omp_get_num_threads();
  std::vector<EvalType> local_sum(nthreads, EvalType(0));
  LBANN_OMP_TASKLOOP_COLLAPSE2
  for(El::Int col = 0; col < local_width; ++col) {
    for(El::Int row = 0; row < local_height; ++row) {
      const EvalType true_val = ground_truth_local(row, col);
      const EvalType pred_val = predictions_local(row, col);
      const EvalType error = true_val - pred_val;
      const int tid = omp_get_thread_num();
      local_sum[tid] += error * error;
    }
  }
  for (int i = 0; i < nthreads; ++i) {
    sum += local_sum[i];
  }
  m_sum = sum / (height * width);  // Can't reduce on class members.
  get_comm().nb_allreduce(&m_sum, 1, predictions.DistComm(), m_allreduce_req);
}

EvalType mean_squared_error_loss::finish_evaluate_compute(
  const AbsDistMat& predictions,
  const AbsDistMat& ground_truth) {
  get_comm().wait(m_allreduce_req);
  return m_sum;
}

void mean_squared_error_loss::differentiate_compute(const AbsDistMat& predictions,
                                                    const AbsDistMat& ground_truth,
                                                    AbsDistMat& gradient) {

  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  Mat& gradient_local = gradient.Matrix();

  // Matrix parameters
  const int height = gradient.Height();
  const El::Int local_height = gradient_local.Height();
  const El::Int local_width = gradient_local.Width();

  // Compute gradient
  const DataType scale = DataType(2) / height;
  LBANN_OMP_TASKLOOP_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      const DataType pred_val = predictions_local(row, col);
      gradient_local(row, col) = (pred_val - true_val) * scale;
    }
  }

}

}  // namespace lbann
