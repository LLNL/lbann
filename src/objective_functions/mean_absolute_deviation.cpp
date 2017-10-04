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

#include "lbann/objective_functions/mean_absolute_deviation.hpp"

namespace lbann {

namespace objective_functions {

void mean_absolute_deviation::compute_value(const AbsDistMat& predictions,
                                       const AbsDistMat& ground_truth) {
  
  // Get local matrices and matrix parameters
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  const El::Int height = predictions.Height();
  const El::Int width = predictions.Width();
  const El::Int local_height = predictions_local.Height();
  const El::Int local_width = predictions_local.Width();

  // Compute sum of absolute deviations
  double sum = 0;
  const El::Int block_size = std::max((int) (64 / sizeof(DataType)), 1);
  #pragma omp parallel for reduction(+:sum) collapse(2)
  for(El::Int col = 0; col < local_width; ++col) {
    for(El::Int block_start = 0; block_start < local_height; block_start += block_size) {
      double block_sum = 0;
      const El::Int block_end = std::min(block_start + block_size, local_height);
      for(El::Int row = block_start; row < block_end; ++row) {
        const double true_val = ground_truth_local(row, col);
        const double pred_val = predictions_local(row, col);
        block_sum += std::abs(true_val - pred_val);
        //const double error = true_val - pred_val;
        //block_sum += error * error;
      }
      sum += block_sum;
    }
  }
  
  // Compute mean absolute deviation
  double mad = sum / (height * width);
  mse = El::mpi::AllReduce(mad, predictions.DistComm());

  // Update objective function value
  add_to_value(mad);

}

/// Compute derivative of mean absolute deviation objective function
void mean_absolute_deviation::compute_gradient(const AbsDistMat& predictions,
                                               const AbsDistMat& ground_truth,
                                               AbsDistMat& gradient) {

  // Get local matrices and matrix parameters
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  Mat& gradient_local = gradient.Matrix();
  const int height = predictions.Height();

  // Compute gradient
  El::IndexDependentFill(gradient_local,
                         (std::function<DataType(El::Int,El::Int)>)
                         ([&predictions_local, &ground_truth_local, height]
                          (El::Int r, El::Int c) -> DataType {
                           const DataType pred_val = predictions_local(r,c);
                           const DataType true_val = ground_truth_local(r,c);
                           if (pred_val > true_val) return DataType(1)/height;
                           else if (pred_val < true_val) return DataType(-1)/height;
                           else return DataType(0);
                           //return 2 * (pred_val - true_val) / height;
                         }));

}

}  // namespace objective_functions

}  // namespace lbann
