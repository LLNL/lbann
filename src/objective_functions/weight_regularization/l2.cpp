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

#include "lbann/objective_functions/weight_regularization/l2.hpp"
#include "lbann/models/model.hpp"
#ifdef LBANN_HAS_CUDNN
#include "lbann/utils/cublas_wrapper.hpp"
#endif // LBANN_HAS_CUDNN

namespace {

  /** Compute the entry-wise sum of squares of a local matrix. */
  EvalType sum_of_squares(const Mat& mat) {
    const El::Int height = mat.Height();
    const El::Int width = mat.Width();
    const El::Int ldim = mat.LDim();
    const auto& __restrict__ buf = mat.LockedBuffer();
    EvalType sqsum = EvalType(0);
    if (ldim == height) {
      // Parallelize single loop if data is contiguous
      const El::Int size = height*width;
      #pragma omp parallel for reduction(+:sqsum)
      for (El::Int i = 0; i < size; ++i) {
        const EvalType val = buf[i];
        sqsum += val * val;
      }
    } else {
      // Parallelize double loop if data is not contiguous
      #pragma omp parallel for reduction(+:sqsum) collapse(2)
      for (El::Int j = 0; j < width; ++j) {
        for (El::Int i = 0; i < height; ++i) {
          const EvalType val = buf[i + j*ldim];
          sqsum += val * val;
        }
      }
    }
    return sqsum;
  }

} // namespace

namespace lbann {

void l2_weight_regularization::setup(model& m) {
  objective_function_term::setup(m);

  // Check that term has no layer pointers
  if (!m_layers.empty()) {
    LBANN_ERROR("attempted to setup L2 weight regularization with layer pointers");
  }

  // Add all weights in model if no weights pointers are provided
  if (m_weights.empty()) {
    for (weights* w : m.get_weights()) {
      if (w->get_optimizer() != nullptr) {
        m_weights.push_back(w);
      }
    }
  }
  m_sqsums.resize(m_weights.size(), EvalType(0));
  m_allreduce_started.resize(m_weights.size(), false);
  for (size_t i = 0; i < m_weights.size(); ++i) {
    m_allreduce_reqs.emplace_back();
  }

#ifdef LBANN_HAS_CUDNN
  // Get cuDNN manager
  m_cudnn = nullptr;
  for (auto&& w : m_weights) {
    m_cudnn = w->get_cudnn_manager();
    if (m_cudnn != nullptr) { break; }
  }
#endif // LBANN_HAS_CUDNN

}

void l2_weight_regularization::start_evaluation() {
  if (m_scale_factor == EvalType(0)) { return; }

  // Reset terms for each weights
  const int num_weights = m_weights.size();
  std::fill(m_sqsums.begin(), m_sqsums.end(), EvalType(0));

#ifdef LBANN_HAS_CUDNN
  // Compute terms for GPU weights
  if (m_cudnn != nullptr) {
    const auto& num_gpus = m_cudnn->get_num_gpus();
    const auto& weights_per_gpu = (num_weights + num_gpus - 1) / num_gpus;
    cudnn::matrix sqsums_d(m_cudnn);
    sqsums_d.attach_to_work_spaces(1, weights_per_gpu);
    sqsums_d.zero();
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      CHECK_CUDA(cudaSetDevice(m_cudnn->get_gpu(gpu)));
      auto&& handle = m_cudnn->get_cublas_handle(gpu);
      CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
      const int weights_start = std::min(gpu * weights_per_gpu, num_weights);
      const int weights_end   = std::min((gpu+1) * weights_per_gpu, num_weights);
      for (int i = weights_start; i < weights_end; ++i) {
        const auto& w = m_weights[i];
        if (w->get_cudnn_manager() != nullptr) {
          const auto values_d = w->get_values_gpu()[gpu];
          auto sqsum_d = sqsums_d.get_data(gpu) + i - weights_start;
          cublas::dot(handle, w->get_size(), values_d, 1, values_d, 1, sqsum_d);
        }
      }
      CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    }
    Mat sqsums(1, num_weights);
    m_cudnn->gather_from_gpus(sqsums, sqsums_d.get_locked_data(), weights_per_gpu);
    m_cudnn->synchronize();
    for (int i = 0; i < num_weights; ++i) {
      m_sqsums[i] = EvalType(sqsums(0, i));
    }
  }
#endif // LBANN_HAS_CUDNN

  // Compute terms for CPU weights
  std::fill(m_allreduce_started.begin(), m_allreduce_started.end(), false);
  for (int i = 0; i < num_weights; ++i) {
    const auto& w = m_weights[i];
    if (w->get_cudnn_manager() == nullptr) {
      const auto& values = w->get_values();
      m_sqsums[i] = sum_of_squares(values.LockedMatrix());
      get_comm().nb_allreduce(&(m_sqsums[i]), 1, values.DistComm(),
                              m_allreduce_reqs[i], El::mpi::SUM);
      m_allreduce_started[i] = true;
    }
  }

}

EvalType l2_weight_regularization::finish_evaluation() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  for (size_t i = 0; i < m_weights.size(); ++i) {
    if (m_allreduce_started[i]) {
      get_comm().wait(m_allreduce_reqs[i]);
    }
  }
  const auto& sqsum = std::accumulate(m_sqsums.begin(),
                                      m_sqsums.end(),
                                      EvalType(0));
  return m_scale_factor * sqsum / 2;
}

void l2_weight_regularization::compute_weight_regularization() {
  if (m_scale_factor == EvalType(0)) { return; }

#ifdef LBANN_HAS_CUDNN
  // Compute gradient of L2 regularization term for GPU weights
  for (auto&& w : m_weights) {
    if (w->get_cudnn_manager() != nullptr) {
      cudnn::matrix values_d(m_cudnn);
      values_d.locked_attach(w->get_values_gpu(), w->get_size());
      w->get_optimizer()->add_to_gradient(values_d, m_scale_factor);
    }
  }
#endif // LBANN_HAS_CUDNN

  // Compute gradient of L2 regularization term for CPU weights
  for (auto&& w : m_weights) {
    if (w->get_cudnn_manager() == nullptr) {
      w->get_optimizer()->add_to_gradient(w->get_values(), m_scale_factor);
    }
  }

}

} // namespace lbann
