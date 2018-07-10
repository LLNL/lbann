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

#include "lbann/metrics/mean_absolute_deviation.hpp"

namespace lbann {

EvalType mean_absolute_deviation_metric::evaluate_compute(const AbsDistMat& prediction,
                                                          const AbsDistMat& ground_truth) {

  // Get matrix dimensions
  const int height = prediction.Height();
  const int local_height = prediction.LocalHeight();
  const int local_width = prediction.LocalWidth();

  // Get local matrices
  const Mat& prediction_local = prediction.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();

  // Compute sum of squared errors
  EvalType sum = 0;
#pragma omp taskloop collapse(2) default(shared) /// @todo reduction(+:sum)
  for(El::Int col = 0; col < local_width; ++col) {
    for(El::Int row = 0; row < local_height; ++row) {
      const EvalType true_val = ground_truth_local(row, col);
      const EvalType pred_val = prediction_local(row, col);
      const EvalType error = true_val - pred_val;
      #pragma omp critical
      sum += error >= DataType(0) ? error : - error;
    }
  }

  // Compute mean value across mini-batch
  return get_comm().allreduce(sum / height, prediction.DistComm());

}

}  // namespace lbann
