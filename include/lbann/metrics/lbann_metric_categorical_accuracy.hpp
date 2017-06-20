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

  /// Destructor
  ~categorical_accuracy();
  void initialize_model_parallel_distribution();
  void initialize_data_parallel_distribution();

  void setup(int num_neurons, int mini_batch_size);
  void fp_set_std_matrix_view(int64_t cur_mini_batch_size);
  double compute_metric(ElMat& predictions_v, ElMat& groundtruth_v);

  double report_metric(execution_mode mode);
  double report_lifetime_metric(execution_mode mode);

 protected:
  AbsDistMat *YsColMax; /// Note that the column max matrix has the number of mini-batches on the rows instead of columns
  StarMat YsColMaxStar; /// Fully replicated set of the column max matrix
  Mat m_max_index;    /// Local array to hold max indicies
  Mat m_reduced_max_indices;  /// Local array to build global view of maximum indicies
  //    Mat Y_local;

  AbsDistMat *YsColMax_v;
  StarMat YsColMaxStar_v;
  Mat m_max_index_v;
  Mat m_reduced_max_indices_v;
  //    Mat Y_local_v;

  int64_t m_max_mini_batch_size;
};
}
}


#endif // LBANN_METRIC_CATEGORICAL_ACCURACY_HPP
