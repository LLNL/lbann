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

#include "lbann/objective_functions/loss_functions/polya_negloglike.hpp"

double digamma(double x) {
  double result = 0.0;
  double z = x;
  while (z < 8.0) {
    result -= 1.0/z;
    z += 1.0;
  }
  double s = 1.0/z;
  double s2 = s*s;
  double s4 = s2*s2;
  double s6 = s2*s4;
  double s8 = s4*s4;
  result += std::log(z) - s/2.0 - s2/12.0 + s4/120.0 - s6/252.0 + s8/240.0 - (5.0/660.0)*s4*s6 +(691.0/32760.0)*s6*s6 - s6*s8/12.0;
  return result;
}

namespace lbann {

polya_negloglike::polya_negloglike(EvalType scale_factor)
  : loss_function(scale_factor),
    m_counts(nullptr),
    m_alpha_sums(nullptr),
    m_lgamma_alpha_sums(nullptr),
    m_lgamma_alpha_level_count_sums(nullptr) {}

polya_negloglike::polya_negloglike(const polya_negloglike& other)
  : loss_function(other),
    m_counts(other.m_counts),
    m_alpha_sums(other.m_alpha_sums),
    m_lgamma_alpha_sums(other.m_lgamma_alpha_sums),
    m_lgamma_alpha_level_count_sums(other.m_lgamma_alpha_level_count_sums) {
  if (m_counts != nullptr) {
    m_counts = m_counts->Copy();
  }
  if (m_alpha_sums != nullptr) {
    m_alpha_sums = m_alpha_sums->Copy();
  }
  if (m_lgamma_alpha_sums != nullptr) {
    m_lgamma_alpha_sums = m_lgamma_alpha_sums->Copy();
  }
  if (m_lgamma_alpha_level_count_sums != nullptr) {
    m_lgamma_alpha_level_count_sums = m_lgamma_alpha_level_count_sums->Copy();
  }
}

polya_negloglike& polya_negloglike::operator=(const polya_negloglike& other) {
  loss_function::operator=(other);
  if (m_counts != nullptr) delete m_counts;
  if (m_alpha_sums != nullptr) delete m_alpha_sums;
  if (m_lgamma_alpha_sums != nullptr) delete m_lgamma_alpha_sums;
  if (m_lgamma_alpha_level_count_sums != nullptr) delete m_lgamma_alpha_level_count_sums;
  m_counts = other.m_counts;
  m_alpha_sums = other.m_alpha_sums;
  m_lgamma_alpha_sums = other.m_lgamma_alpha_sums;
  m_lgamma_alpha_level_count_sums = other.m_lgamma_alpha_level_count_sums;
  if (m_counts != nullptr) {
    m_counts = m_counts->Copy();
  }
  if (m_alpha_sums != nullptr) {
    m_alpha_sums = m_alpha_sums->Copy();
  }
  if (m_lgamma_alpha_sums != nullptr) {
    m_lgamma_alpha_sums = m_lgamma_alpha_sums->Copy();
  }
  if (m_lgamma_alpha_level_count_sums != nullptr) {
    m_lgamma_alpha_level_count_sums = m_lgamma_alpha_level_count_sums->Copy();
  }
  return *this;
}

polya_negloglike::~polya_negloglike() {
  if (m_counts != nullptr) delete m_counts;
  if (m_alpha_sums != nullptr) delete m_alpha_sums;
  if (m_lgamma_alpha_sums != nullptr) delete m_lgamma_alpha_sums;
  if (m_lgamma_alpha_level_count_sums != nullptr) delete m_lgamma_alpha_level_count_sums;
}

void polya_negloglike::setup(model& m) {
  loss_function::setup(m);

  const El::DistData dist(*m_gradient);
  const El::Device dev = m_gradient->GetLocalDevice();
  if (dist.colDist == El::MC && dist.rowDist == El::MR) {
    switch(dev) {
    case El::Device::CPU:
      m_counts                        = new StarMRMat<El::Device::CPU>(*dist.grid);
      m_alpha_sums                    = new StarMRMat<El::Device::CPU>(*dist.grid);
      m_lgamma_alpha_sums             = new StarMRMat<El::Device::CPU>(*dist.grid);
      m_lgamma_alpha_level_count_sums = new StarMRMat<El::Device::CPU>(*dist.grid);
      break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      m_counts                        = new StarMRMat<El::Device::GPU>(*dist.grid);
      m_alpha_sums                    = new StarMRMat<El::Device::GPU>(*dist.grid);
      m_lgamma_alpha_sums             = new StarMRMat<El::Device::GPU>(*dist.grid);
      m_lgamma_alpha_level_count_sums = new StarMRMat<El::Device::GPU>(*dist.grid);
      break;
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
      m_counts                        = new StarVCMat<El::Device::CPU>(*dist.grid);
      m_alpha_sums                    = new StarVCMat<El::Device::CPU>(*dist.grid);
      m_lgamma_alpha_sums             = new StarVCMat<El::Device::CPU>(*dist.grid);
      m_lgamma_alpha_level_count_sums = new StarVCMat<El::Device::CPU>(*dist.grid);
      break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      m_counts                        = new StarVCMat<El::Device::GPU>(*dist.grid);
      m_alpha_sums                    = new StarVCMat<El::Device::GPU>(*dist.grid);
      m_lgamma_alpha_sums             = new StarVCMat<El::Device::GPU>(*dist.grid);
      m_lgamma_alpha_level_count_sums = new StarVCMat<El::Device::GPU>(*dist.grid);
      break;
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

EvalType polya_negloglike::finish_evaluate_compute(
  const AbsDistMat& predictions, const AbsDistMat& ground_truth) {

  // Initialize workspace
  m_counts->Resize(1, predictions.Width());
  m_alpha_sums->Resize(1, predictions.Width());
  m_lgamma_alpha_sums->Resize(1, predictions.Width());
  m_lgamma_alpha_level_count_sums->Resize(1, predictions.Width());
  Mat& counts_local = m_counts->Matrix();
  Mat& alpha_sums_local = m_alpha_sums->Matrix();
  Mat& lgamma_alpha_sums_local = m_lgamma_alpha_sums->Matrix();
  Mat& lgamma_alpha_level_count_sums_local = m_lgamma_alpha_level_count_sums->Matrix();

  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();

  // Matrix parameters
  const int width = predictions.Width();
  const int local_height = predictions_local.Height();
  const int local_width = predictions_local.Width();

  #pragma omp parallel for
  for (int col = 0; col < local_width; ++col) {
    DataType count = DataType(0);
    DataType alpha_sum = DataType(0);
    DataType lgamma_alpha_sum = DataType(0);
    DataType lgamma_alpha_level_count_sum = DataType(0);
    for (int row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      const DataType pred_val = predictions_local(row, col);
      count += true_val;
      alpha_sum += pred_val;
      lgamma_alpha_sum += std::lgamma(pred_val);
      lgamma_alpha_level_count_sum += std::lgamma(pred_val + true_val);
    }
    counts_local(0, col) = count;
    alpha_sums_local(0, col) = alpha_sum;
    lgamma_alpha_sums_local(0, col) = lgamma_alpha_sum;
    lgamma_alpha_level_count_sums_local(0, col) = lgamma_alpha_level_count_sum;
  }
  get_comm().allreduce(*m_counts, m_counts->RedundantComm());
  get_comm().allreduce(*m_alpha_sums, m_alpha_sums->RedundantComm());
  get_comm().allreduce(*m_lgamma_alpha_sums, m_lgamma_alpha_sums->RedundantComm());
  get_comm().allreduce(*m_lgamma_alpha_level_count_sums,
                        m_lgamma_alpha_level_count_sums->RedundantComm());

  // Compute mean objective function value across mini-batch
  EvalType local_sum = EvalType(0);
  for (int col = 0; col < local_width; ++col) {
    local_sum += (- std::lgamma(alpha_sums_local(0, col))
                  + std::lgamma(counts_local(0, col) + alpha_sums_local(0, col))
                  - lgamma_alpha_level_count_sums_local(0, col)
                  + lgamma_alpha_sums_local(0, col));
  }
  return get_comm().allreduce(local_sum / width, m_counts->DistComm());

}

void polya_negloglike::differentiate_compute(const AbsDistMat& predictions,
                                             const AbsDistMat& ground_truth,
                                             AbsDistMat& gradient) {

  // Local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  const Mat& counts_local = m_counts->LockedMatrix();
  const Mat& alpha_sums_local = m_alpha_sums->LockedMatrix();
  Mat& gradient_local = gradient.Matrix();

  // Matrix parameters
  const IntType local_height = gradient_local.Height();
  const IntType local_width = gradient_local.Width();

  // Compute gradient
  #pragma omp parallel for collapse(2)
  for (IntType col = 0; col < local_width; ++col) {
    for (IntType row = 0; row < local_height; ++row) {
      const DataType true_val = ground_truth_local(row, col);
      const DataType pred_val = predictions_local(row, col);
      const DataType alpha_sum = alpha_sums_local(0, col);
      const DataType count = counts_local(0, col);
      gradient_local(row, col) = (- digamma(alpha_sum)
                                  + digamma(count + alpha_sum)
                                  - digamma(true_val + pred_val)
                                  + digamma(pred_val));
    }
  }

}

}  // namespace lbann
