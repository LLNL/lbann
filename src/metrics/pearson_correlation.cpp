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

#include "lbann/metrics/pearson_correlation.hpp"
#include "lbann/utils/statistics.hpp"

namespace lbann {

double pearson_correlation_metric::evaluate_compute(const AbsDistMat& prediction,
                                                    const AbsDistMat& ground_truth) {

  // Initialize matrices
  const int height = prediction.Height();
  const int local_height = prediction.LocalHeight();
  const int local_width = prediction.LocalWidth();
  const Mat& prediction_local = prediction.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();

  // Initialize workspace to compute column-wise statistics
  std::vector<double> means(5 * local_width, 0.0);
  double *pred_means    = &means[0 * local_width];
  double *pred_sqmeans  = &means[1 * local_width];
  double *true_means    = &means[2 * local_width];
  double *true_sqmeans  = &means[3 * local_width];
  double *product_means = &means[4 * local_width];

  // Accumulate sums for statistics
  #pragma omp parallel for
  for (int col = 0; col < local_width; ++col) {
    for (int row = 0; row < local_height; ++row) {
      const double pred_val = prediction_local(row, col);
      const double true_val = ground_truth_local(row, col);
      pred_means[col]    += pred_val;
      pred_sqmeans[col]  += pred_val * pred_val;
      true_means[col]    += true_val;
      true_sqmeans[col]  += true_val * true_val;
      product_means[col] += pred_val * true_val;
    }
  }
  get_comm().allreduce(means.data(),
                       means.size(),
                       prediction.ColComm());
  for (auto& x : means) {
    x /= height;
  }

  // Compute Pearson correlation of each column
  double local_sum = 0.0;
  for (int col = 0; col < local_width; ++col) {
    const double pred_mean = pred_means[col];
    const double pred_var  = std::max(pred_sqmeans[col] - pred_mean * pred_mean, 0.0);
    const double true_mean = true_means[col];
    const double true_var  = std::max(true_sqmeans[col] - true_mean * true_mean, 0.0);
    const double cov       = product_means[col] - pred_mean * true_mean;
    local_sum += cov / std::sqrt(pred_var * true_var);
  }
  return get_comm().allreduce(local_sum, prediction.RowComm());

}

}  // namespace lbann
