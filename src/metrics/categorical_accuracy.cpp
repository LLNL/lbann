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

#include "lbann/metrics/categorical_accuracy.hpp"
#include "lbann/layers/io/target/generic_target_layer.hpp"

namespace lbann {

categorical_accuracy_metric::categorical_accuracy_metric(lbann_comm *comm)
  : metric(comm), m_prediction_values(nullptr) {}

categorical_accuracy_metric::categorical_accuracy_metric(const categorical_accuracy_metric& other)
  : metric(other),
    m_prediction_values(other.m_prediction_values),
    m_prediction_indices(other.m_prediction_indices) {
  if (m_prediction_values != nullptr) {
    m_prediction_values = m_prediction_values->Copy();
  }
}

categorical_accuracy_metric& categorical_accuracy_metric::operator=(const categorical_accuracy_metric& other) {
  metric::operator=(other);
  if (m_prediction_values != nullptr) delete m_prediction_values;
  m_prediction_values = other.m_prediction_values;
  if (m_prediction_values != nullptr) {
    m_prediction_values = m_prediction_values->Copy();
  }
  m_prediction_indices = other.m_prediction_indices;
  return *this;
}

categorical_accuracy_metric::~categorical_accuracy_metric() {
  if (m_prediction_values != nullptr) delete m_prediction_values;
}

void categorical_accuracy_metric::setup(model& m) {
  metric::setup(m);
  const El::DistData dist_data(get_target_layer().get_prediction());
  const El::Device dev = get_target_layer().get_prediction().GetLocalDevice();
  if (dist_data.colDist == El::MC
      && dist_data.rowDist == El::MR) {
    switch(dev) {
    case El::Device::CPU:
      m_prediction_values = new StarMRMat<El::Device::CPU>(*dist_data.grid, dist_data.root); break;
    case El::Device::GPU:
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid matrix data allocation";
      throw lbann_exception(err.str());
    }
  } else if (dist_data.colDist == El::STAR
             && dist_data.rowDist == El::VC) {
    switch(dev) {
    case El::Device::CPU:
      m_prediction_values = new StarVCMat<El::Device::CPU>(*dist_data.grid, dist_data.root); break;
    case El::Device::GPU:
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid matrix data allocation";
      throw lbann_exception(err.str());
    }
  } else {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "prediction matrix has invalid distribution "
        << "(colDist=" << dist_data.colDist << ","
        << "rowDist=" << dist_data.rowDist << ")";
    throw lbann_exception(err.str());
  }
  m_prediction_values->Resize(1, m.get_max_mini_batch_size());
  m_prediction_indices.resize(m_prediction_values->LocalWidth());
}

EvalType categorical_accuracy_metric::evaluate_compute(const AbsDistMat& prediction,
                                                       const AbsDistMat& ground_truth) {

  // Get matrix dimensions
  const int height = prediction.Height();
  const int width = prediction.Width();
  const int local_height = prediction.LocalHeight();
  const int local_width = prediction.LocalWidth();

  // Get local matrices
  const Mat& prediction_local = prediction.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();

  // Initialize workspace matrices
  m_prediction_values->Resize(1, width);
  CPUMat& prediction_values_local = m_prediction_values->Matrix();
  m_prediction_indices.resize(local_width);

  // Find largest value in each column of prediction matrix
  #pragma omp parallel for
  for (int col = 0; col < local_width; ++col) {
    DataType max_val;
    int max_index;
    if (local_height > 0) {
      max_val = prediction_local(0, col);
      max_index = 0;
    } else {
      max_val = std::numeric_limits<DataType>::lowest();
      max_index = -1;
    }
    for (int row = 1; row < local_height; ++row) {
      const DataType pred_val = prediction_local(row, col);
      if (pred_val > max_val) {
        max_val = pred_val;
        max_index = row;
      }
    }
    prediction_values_local(0, col) = max_val;
    m_prediction_indices[col] = max_index;
  }
  get_comm().allreduce(*m_prediction_values,
                       m_prediction_values->RedundantComm(),
                       El::mpi::MAX);

  // Find first index corresponding to maximum prediction matrix values
  #pragma omp parallel for
  for (int col = 0; col < local_width; ++col) {
    const int row = m_prediction_indices[col];
    if (local_height > 0
        && prediction_local(row, col) >= prediction_values_local(0, col)) {
      m_prediction_indices[col] = prediction.GlobalRow(row);
    } else {
      m_prediction_indices[col] = height;
    }
  }
  get_comm().allreduce(m_prediction_indices.data(),
                       local_width,
                       m_prediction_values->RedundantComm(),
                       El::mpi::MIN);

  // Count number of correct predictions
  int correct_predictions = 0;
  for (int col = 0; col < local_width; ++col) {
    const int global_row = m_prediction_indices[col];
    if (ground_truth.IsLocalRow(global_row)) {
      const int row = ground_truth.LocalRow(global_row);
      if (row == ground_truth.Height()) {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
            << "ground_truth matrix has invalid index "
            << "(row=" << row << " x "
            << "col=" << col << ")";
        throw lbann_exception(err.str());
      }
      if (ground_truth_local(row, col) != DataType(0)) {
        ++correct_predictions;
      }
    }
  }
  correct_predictions = get_comm().model_allreduce(correct_predictions);

  // Return percentage of correct predictions
  return EvalType(100) * correct_predictions;

}

}  // namespace lbann
