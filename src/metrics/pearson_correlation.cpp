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
#include "lbann/layers/io/target/target_layer.hpp"
#include "lbann/utils/statistics.hpp"

namespace lbann {

pearson_correlation_metric::pearson_correlation_metric(lbann_comm* comm)
  : metric(comm),
    m_prediction_means(nullptr),
    m_prediction_stdevs(nullptr),
    m_ground_truth_means(nullptr),
    m_ground_truth_stdevs(nullptr),
    m_covariances(nullptr) {}

pearson_correlation_metric::pearson_correlation_metric(const pearson_correlation_metric& other)
  : metric(other),
    m_prediction_means(other.m_prediction_means),
    m_prediction_stdevs(other.m_prediction_stdevs),
    m_ground_truth_means(other.m_ground_truth_means),
    m_ground_truth_stdevs(other.m_ground_truth_stdevs),
    m_covariances(other.m_covariances) {
  if (m_prediction_means != nullptr) {
    m_prediction_means = m_prediction_means->Copy();
  }
  if (m_prediction_stdevs != nullptr) {
    m_prediction_stdevs = m_prediction_stdevs->Copy();
  }
  if (m_ground_truth_means != nullptr) {
    m_ground_truth_means = m_ground_truth_means->Copy();
  }
  if (m_ground_truth_stdevs != nullptr) {
    m_ground_truth_stdevs = m_ground_truth_stdevs->Copy();
  }
  if (m_covariances != nullptr) {
    m_covariances = m_covariances->Copy();
  }
}

pearson_correlation_metric& pearson_correlation_metric::operator=(const pearson_correlation_metric& other) {
  metric::operator=(other);
  if (m_prediction_means != nullptr)    delete m_prediction_means;
  if (m_prediction_stdevs != nullptr)   delete m_prediction_stdevs;
  if (m_ground_truth_means != nullptr)  delete m_ground_truth_means;
  if (m_ground_truth_stdevs != nullptr) delete m_ground_truth_stdevs;
  if (m_covariances != nullptr)         delete m_covariances;
  m_prediction_means    = other.m_prediction_means;
  m_prediction_stdevs   = other.m_prediction_stdevs;
  m_ground_truth_means  = other.m_ground_truth_means;
  m_ground_truth_stdevs = other.m_ground_truth_stdevs;
  m_covariances         = other.m_covariances;
  if (m_prediction_means != nullptr) {
    m_prediction_means = m_prediction_means->Copy();
  }
  if (m_prediction_stdevs != nullptr) {
    m_prediction_stdevs = m_prediction_stdevs->Copy();
  }
  if (m_ground_truth_means != nullptr) {
    m_ground_truth_means = m_ground_truth_means->Copy();
  }
  if (m_ground_truth_stdevs != nullptr) {
    m_ground_truth_stdevs = m_ground_truth_stdevs->Copy();
  }
  if (m_covariances != nullptr) {
    m_covariances = m_covariances->Copy();
  }
  return *this;
}

pearson_correlation_metric::~pearson_correlation_metric() {
  if (m_prediction_means != nullptr)    delete m_prediction_means;
  if (m_prediction_stdevs != nullptr)   delete m_prediction_stdevs;
  if (m_ground_truth_means != nullptr)  delete m_ground_truth_means;
  if (m_ground_truth_stdevs != nullptr) delete m_ground_truth_stdevs;
  if (m_covariances != nullptr)         delete m_covariances;
}

void pearson_correlation_metric::setup(model& m) {
  metric::setup(m);
  const El::DistData dist_data(get_target_layer().get_prediction());
  if (dist_data.colDist == El::MC
      && dist_data.rowDist == El::MR) {
    m_prediction_means    = new StarMRMat(*dist_data.grid, dist_data.root);
    m_prediction_stdevs   = new StarMRMat(*dist_data.grid, dist_data.root);
    m_ground_truth_means  = new StarMRMat(*dist_data.grid, dist_data.root);
    m_ground_truth_stdevs = new StarMRMat(*dist_data.grid, dist_data.root);
    m_covariances         = new StarMRMat(*dist_data.grid, dist_data.root);
  } else if (dist_data.colDist == El::STAR
             && dist_data.rowDist == El::VC) {
    m_prediction_means    = new StarVCMat(*dist_data.grid, dist_data.root);
    m_prediction_stdevs   = new StarVCMat(*dist_data.grid, dist_data.root);
    m_ground_truth_means  = new StarVCMat(*dist_data.grid, dist_data.root);
    m_ground_truth_stdevs = new StarVCMat(*dist_data.grid, dist_data.root);
    m_covariances         = new StarVCMat(*dist_data.grid, dist_data.root);
  } else {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "prediction matrix has invalid distribution "
        << "(colDist=" << dist_data.colDist << ","
        << "rowDist=" << dist_data.rowDist << ")";
    throw lbann_exception(err.str());
  }
  m_prediction_means->Resize(1, m.get_max_mini_batch_size());
  m_prediction_stdevs->Resize(1, m.get_max_mini_batch_size());
  m_ground_truth_means->Resize(1, m.get_max_mini_batch_size());
  m_ground_truth_stdevs->Resize(1, m.get_max_mini_batch_size());
  m_covariances->Resize(1, m.get_max_mini_batch_size());
}

EvalType pearson_correlation_metric::evaluate_compute(const AbsDistMat& prediction,
                                                      const AbsDistMat& ground_truth) {

  // Compute means, standard deviations, and covariances
  columnwise_mean_and_stdev(prediction, *m_prediction_means, *m_prediction_stdevs);
  columnwise_mean_and_stdev(ground_truth, *m_ground_truth_means, *m_ground_truth_stdevs);
  columnwise_covariance(prediction, ground_truth,
                        *m_prediction_means, *m_ground_truth_means,
                        *m_covariances);

  // Compute Pearson correlation of each column
  // Note: Pearson(x,y) = cov(x,y) / ( stdev(x) * stdev(y) )
  EvalType local_sum = 0.0;
  for (int col = 0; col < m_covariances->LocalWidth(); ++col) {
    const EvalType pred_stdev = m_prediction_stdevs->GetLocal(0, col);
    const EvalType true_stdev = m_ground_truth_stdevs->GetLocal(0, col);
    const EvalType cov        = m_covariances->GetLocal(0, col);
    local_sum += cov / (pred_stdev * true_stdev);
  }
  return get_comm().allreduce(local_sum, m_covariances->DistComm());

}

}  // namespace lbann
