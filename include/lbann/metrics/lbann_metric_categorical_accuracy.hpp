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

#ifndef LBANN_METRIC_CATEGORICAL_ACCURACY_HPP
#define LBANN_METRIC_CATEGORICAL_ACCURACY_HPP

#include "lbann/metrics/lbann_metric.hpp"

namespace lbann {

namespace metrics {

class categorical_accuracy : public metric {
 public:
  /// Constructor
  categorical_accuracy(data_layout data_dist, lbann_comm *comm);
  categorical_accuracy(const categorical_accuracy& other);
  categorical_accuracy& operator=(const categorical_accuracy& other);

  /// Destructor
  ~categorical_accuracy();
  void initialize_model_parallel_distribution();
  void initialize_data_parallel_distribution();

  void setup(int num_neurons, int mini_batch_size);
  void fp_set_std_matrix_view(int cur_mini_batch_size);
  double compute_metric(ElMat& predictions_v, ElMat& groundtruth_v);

  double report_metric(execution_mode mode);
  double report_lifetime_metric(execution_mode mode);

  std::string name() const { return "categorical accuracy"; }
  std::string display_unit() const { return "%"; }

 protected:
  /// The maximum value within each minibatch (column).
  AbsDistMat *m_prediction_col_maxes;
  /// m_prediction_col_maxes, replicated onto every rank.
  StarMat m_replicated_prediction_col_maxes;
  Mat m_max_index;    /// Local array to hold max indicies
  Mat m_reduced_max_indices;  /// Local array to build global view of maximum indicies

  /// View of m_prediction_col_maxes for the current minibatch size.
  AbsDistMat *m_prediction_col_maxes_v;
  /// View of m_replicated_prediction_col_maxes for the current minibatch size.
  StarMat m_replicated_prediction_col_maxes_v;
  Mat m_max_index_v;
  Mat m_reduced_max_indices_v;

  int m_max_mini_batch_size;
};

}  // namespace metrics

}  // namespace lbann


#endif  // LBANN_METRIC_CATEGORICAL_ACCURACY_HPP
