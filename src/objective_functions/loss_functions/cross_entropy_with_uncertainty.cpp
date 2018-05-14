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

#include "lbann/objective_functions/loss_functions/cross_entropy_with_uncertainty.hpp"

namespace lbann {

cross_entropy_with_uncertainty::cross_entropy_with_uncertainty(EvalType scale_factor)
  : loss_function(scale_factor),
    m_prediction_sums(nullptr) {}

cross_entropy_with_uncertainty::cross_entropy_with_uncertainty(const cross_entropy_with_uncertainty& other)
  : loss_function(other),
    m_prediction_sums(other.m_prediction_sums) {
  if (m_prediction_sums != nullptr) {
    m_prediction_sums = m_prediction_sums->Copy();
  }
}

cross_entropy_with_uncertainty& cross_entropy_with_uncertainty::operator=(const cross_entropy_with_uncertainty& other) {
  loss_function::operator=(other);
  if (m_prediction_sums != nullptr) delete m_prediction_sums;
  m_prediction_sums = other.m_prediction_sums;
  if (m_prediction_sums != nullptr) {
    m_prediction_sums = m_prediction_sums->Copy();
  }
  return *this;
}

cross_entropy_with_uncertainty::~cross_entropy_with_uncertainty() {
  if (m_prediction_sums != nullptr) delete m_prediction_sums;
}

void cross_entropy_with_uncertainty::setup(model& m) {
  loss_function::setup(m);

  const El::DistData dist(*m_gradient);
  const El::Device dev = m_gradient->GetLocalDevice();
  if (dist.colDist == El::MC && dist.rowDist == El::MR) {
    switch(dev) {
    case El::Device::CPU:
      m_prediction_sums = new StarMRMat<El::Device::CPU>(*dist.grid); break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      m_prediction_sums = new StarMRMat<El::Device::GPU>(*dist.grid); break;
#endif // LBANN_HAS_GPU
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid matrix data allocation";
      throw lbann_exception(err.str());
    }
  } else if (dist.colDist == El::STAR && dist.rowDist == El::VC) {
    switch(dev) {
    case El::Device::CPU:
      m_prediction_sums = new StarVCMat<El::Device::CPU>(*dist.grid); break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      m_prediction_sums = new StarVCMat<El::Device::GPU>(*dist.grid); break;
#endif // LBANN_HAS_GPU
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid matrix data allocation";
      throw lbann_exception(err.str());
    }
  } else {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid matrix distribution";
    throw lbann_exception(err.str());
  }

}

EvalType cross_entropy_with_uncertainty::finish_evaluate_compute(
  const AbsDistMat& predictions, const AbsDistMat& ground_truth) {

  // Initialize workspace
  m_prediction_sums->Resize(1, predictions.Width());
  Mat& prediction_sums_local = m_prediction_sums->Matrix();

  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();

  // Matrix parameters
  const int width = predictions.Width();
  const int local_height = predictions_local.Height();
  const int local_width = predictions_local.Width();

  // Compute sum of predictions
  #pragma omp parallel for
  for (int col = 0; col < local_width; ++col) {
    EvalType pred_sum = EvalType(0);
    for (int row = 0; row < local_height; ++row) {
      if (ground_truth_local(row, col) != EvalType(0)) {
        pred_sum += predictions_local(row, col);
      }
    }
    prediction_sums_local(0, col) = pred_sum;
  }
  get_comm().allreduce(*m_prediction_sums,
                        m_prediction_sums->RedundantComm());

  // Compute mean objective function value
  EvalType local_sum = EvalType(0);
  for (int col = 0; col < local_width; ++col) {
    local_sum += -std::log(prediction_sums_local(0, col));
  }
  return get_comm().allreduce(local_sum / width,
                              m_prediction_sums->DistComm());

}

void cross_entropy_with_uncertainty::differentiate_compute(const AbsDistMat& predictions,
                                                           const AbsDistMat& ground_truth,
                                                           AbsDistMat& gradient) {

  // Local matrices
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  const Mat& prediction_sums_local = m_prediction_sums->LockedMatrix();
  Mat& gradient_local = gradient.Matrix();

  // Matrix parameters
  const El::Int local_height = gradient_local.Height();
  const El::Int local_width = gradient_local.Width();

  // Compute gradient
  #pragma omp parallel for collapse(2)
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      DataType& grad_val = gradient_local(row, col);
      if (true_val != DataType(0)) {
        grad_val = -true_val / prediction_sums_local(0, col);
      } else {
        grad_val = DataType(0);
      }
    }
  }

}

}  // namespace lbann
