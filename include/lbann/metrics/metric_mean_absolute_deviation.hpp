///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.  Written by
// the LBANN Research Team (B. Van Essen, et al.) listed in the
// CONTRIBUTORS file. <lbann-dev@llnl.gov>
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

#ifndef LBANN_METRIC_MEAN_ABSOLUTE_DEVIATION_HPP
#define LBANN_METRIC_MEAN_ABSOLUTE_DEVIATION_HPP

#include "lbann/metrics/metric.hpp"
#include "lbann/objective_functions/mean_absolute_deviation.hpp"

namespace lbann {

namespace metrics {

template <data_layout T_layout>
class mean_absolute_deviation : public metric {
 public:
  /// Constructor
  mean_absolute_deviation(lbann_comm *comm) :
    metric(comm) {}
  mean_absolute_deviation(const mean_absolute_deviation<T_layout>& other) = default;
  mean_absolute_deviation& operator=(
    const mean_absolute_deviation<T_layout>& other) = default;

  /// Destructor
  ~mean_absolute_deviation() {}

  mean_absolute_deviation* copy() const { return new mean_absolute_deviation(*this); }

  void setup(int num_neurons, int mini_batch_size) {
    metric::setup(num_neurons, mini_batch_size);
  }
  void fp_set_std_matrix_view(int cur_mini_batch_size) {}
  double compute_metric(AbsDistMat& predictions_v, AbsDistMat& groundtruth_v) {
    
    // Get local matrices and matrix parameters
    const Mat& predictions_local = predictions_v.LockedMatrix();
    const Mat& groundtruth_local = groundtruth_v.LockedMatrix();
    const El::Int height = predictions_v.Height();
    const El::Int width = predictions_v.Width();
    const El::Int local_height = predictions_local.Height();
    const El::Int local_width = predictions_local.Width();

    // Compute mean absolute deviation
    double mad = 0.0;
    for(El::Int col = 0; col < local_width; ++col) {
      for(El::Int row = 0; row < local_height; ++row) {
        const double pred_val = predictions_local(row, col);
        const double true_val = groundtruth_local(row, col);
        mad += std::abs(pred_val - true_val);
      }
    }
    mad /= height * width;
    mad = El::mpi::AllReduce(mad, predictions_v.DistComm());
    return mad;

  }

  double report_metric(execution_mode mode) {
    statistics *stats = get_statistics(mode);
    double error_per_epoch = stats->m_error_per_epoch;
    long samples_per_epoch = stats->m_samples_per_epoch;

    double mad = error_per_epoch / samples_per_epoch;
    //string score = std::to_string(mad);

    return mad;
  }
  double report_lifetime_metric(execution_mode mode) {
    statistics *stats = get_statistics(mode);
    double total_error = stats->m_total_error;
    long total_num_samples = stats->m_total_num_samples;

    double mad = total_error / total_num_samples;
    //string score = std::to_string(mad);

    return mad;
  }

  std::string name() const { return "mean absolute deviation"; }

};

}  // namespace metrics

}  // namespace lbann

#endif  // LBANN_METRIC_MEAN_ABSOLUTE_DEVIATION_HPP
