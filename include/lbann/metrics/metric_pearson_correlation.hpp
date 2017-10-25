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

#ifndef LBANN_METRIC_PEARSON_CORRELATION_HPP
#define LBANN_METRIC_PEARSON_CORRELATION_HPP

#include "lbann/metrics/metric.hpp"
#include "lbann/utils/statistics.hpp"
namespace lbann {
///todo: Add comment
namespace metrics {

template <data_layout T_layout>
class pearson_correlation : public metric {
 public:
  /// Constructor
  pearson_correlation(lbann_comm *comm) :
    metric(comm) {}
  pearson_correlation(const pearson_correlation<T_layout>& other) = default;
  pearson_correlation& operator=(
    const pearson_correlation<T_layout>& other) = default;

  /// Destructor
  ~pearson_correlation() {}

  pearson_correlation* copy() const { return new pearson_correlation(*this); }

  void setup(int num_neurons, int mini_batch_size) {
    metric::setup(num_neurons, mini_batch_size);
  }
  void fp_set_std_matrix_view(int cur_mini_batch_size) {}
  double compute_metric(AbsDistMat& predictions_v, AbsDistMat& groundtruth_v) {
    
    //Get local matrices
    const Mat& predictions_local = predictions_v.LockedMatrix();
    const Mat& groundtruth_local = groundtruth_v.LockedMatrix();
    
    double corr = 0.0;

    // Compute mean and stdev
    DataType pred_mean = 0;
    DataType pred_std = 0;
    DataType true_mean = 0;
    DataType true_std = 0;
    DataType corr_mean = 0;
    DataType corr_std = 0;

    entrywise_mean_and_stdev(predictions_v,pred_mean, pred_std);
    entrywise_mean_and_stdev(groundtruth_v,true_mean, true_std);
    
    //Compute covariance 
    auto sub_pred_mean = [&](const DataType& z) {return z - pred_mean;};
    auto sub_true_mean = [&](const DataType& z) {return z - true_mean;};
     
    AbsDistMat* fsp = predictions_v.Construct(predictions_v.Grid(),
                                               predictions_v.Root());
    AbsDistMat* fst = groundtruth_v.Construct(groundtruth_v.Grid(),
                                               groundtruth_v.Root());
    
    Copy(predictions_v,*fsp);
    Copy(groundtruth_v,*fst);
    
    El::EntrywiseMap(*fsp, El::MakeFunction(sub_pred_mean));
    El::EntrywiseMap(*fst, El::MakeFunction(sub_true_mean));
    
    AbsDistMat* covariance_mat = groundtruth_v.Construct(groundtruth_v.Grid(),
                                               groundtruth_v.Root());

    El::Hadamard(*fsp,*fst, *covariance_mat);

    entrywise_mean_and_stdev(*covariance_mat, corr_mean, corr_std);
    corr = corr_mean/(pred_std*true_std);

    return corr;

  }

  double report_metric(execution_mode mode) {
    statistics *stats = get_statistics(mode);
    double error_per_epoch = stats->m_error_per_epoch;
    long samples_per_epoch = stats->m_samples_per_epoch;

    double corr = error_per_epoch / samples_per_epoch;
    //string score = std::to_string(corr);

    return corr;
  }
  double report_lifetime_metric(execution_mode mode) {
    statistics *stats = get_statistics(mode);
    double total_error = stats->m_total_error;
    long total_num_samples = stats->m_total_num_samples;

    double corr = total_error / total_num_samples;
    //string score = std::to_string(corr);

    return corr;
  }

  std::string name() const { return "pearson correlation"; }

};

}  // namespace metrics

}  // namespace lbann

#endif  // LBANN_METRIC_PEARSON_CORRELATION_HPP
