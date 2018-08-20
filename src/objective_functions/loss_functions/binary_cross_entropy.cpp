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

#include "lbann/objective_functions/loss_functions/binary_cross_entropy.hpp"

namespace lbann {

#ifdef LBANN_DEBUG
namespace binary_cross_entropy_debug {
  /** Check inputs for binary cross entropy.
   *  Throws an exception if the inputs are invalid, e.g. if the
   *  inputs are not in [0,1] or if the binary cross entropy is
   *  singular.
   */
  void check_entry(int global_row, int global_col,
                   DataType true_val, DataType pred_val) {
    if (!(true_val >= DataType(0) && true_val <= DataType(1))) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "binary cross entropy requires ground truth in [0,1], but "
          << "ground_truth(" << global_row << "," << global_col << ")"
          << "=" << true_val;
      throw lbann_exception(err.str());
    }
    if (!(pred_val >= DataType(0) && pred_val <= DataType(1))) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "binary cross entropy requires predictions in [0,1], but "
          << "predictions(" << global_row << "," << global_col << ")"
          << "=" << pred_val;
      throw lbann_exception(err.str());
    }
    if ((pred_val == DataType(0) && true_val != DataType(0))
        || (pred_val == DataType(1) && true_val != DataType(1))) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "binary cross entropy is singular since "
          << "predictions(" << global_row << "," << global_col << ")"
          << "=" << pred_val << " and "
          << "ground_truth(" << global_row << "," << global_col << ")"
          << "=" << true_val;
      throw lbann_exception(err.str());
    }
  }
} // namespace binary_cross_entropy_debug
#endif // LBANN_DEBUG

EvalType binary_cross_entropy::finish_evaluate_compute(
  const AbsDistMat& predictions, const AbsDistMat& ground_truth) {

  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  
  // Matrix parameters
  const int width = predictions.Width();
  const int local_height = predictions_local.Height();
  const int local_width = predictions_local.Width();

  // Compute sum of cross entropy terms
  EvalType sum = 0;
  #pragma omp parallel for reduction(+:sum) collapse(2)
  for (int col = 0; col < local_width; ++col) {
    for (int row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      const DataType pred_val = predictions_local(row, col);
      #ifdef LBANN_DEBUG
      binary_cross_entropy_debug::check_entry(ground_truth.GlobalRow(row),
                                              ground_truth.GlobalCol(col),
                                              true_val,
                                              pred_val);
      #endif // LBANN_DEBUG
      if (true_val > DataType(0)) {
        sum += - true_val * std::log(pred_val);
      }
      if (true_val < DataType(1)) {
        sum += - (EvalType(1) - true_val) * std::log(EvalType(1) - pred_val);
      }
    }
  }

  // Compute mean objective function value across mini-batch
  return get_comm().allreduce(sum / width, predictions.DistComm());

}

void binary_cross_entropy::differentiate_compute(const AbsDistMat& predictions,
                                                 const AbsDistMat& ground_truth,
                                                 AbsDistMat& gradient) {

  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  Mat& gradient_local = gradient.Matrix();

  // Matrix parameters
  const IntType local_height = gradient_local.Height();
  const IntType local_width = gradient_local.Width();

  // Compute gradient
  #pragma omp parallel for collapse(2)
  for (IntType col = 0; col < local_width; ++col) {
    for (IntType row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      const DataType pred_val = predictions_local(row, col);
      #ifdef LBANN_DEBUG
      binary_cross_entropy_debug::check_entry(ground_truth.GlobalRow(row),
                                              ground_truth.GlobalCol(col),
                                              true_val,
                                              pred_val);
      #endif // LBANN_DEBUG
      DataType grad_val = DataType(0);
      if (true_val > DataType(0)) {
        grad_val += - true_val / pred_val;
      }
      if (true_val < DataType(1)) {
        grad_val += (DataType(1) - true_val) / (DataType(1) - pred_val);
      }
      gradient_local(row, col) = grad_val;
    }
  }

}

} // namespace lbann
