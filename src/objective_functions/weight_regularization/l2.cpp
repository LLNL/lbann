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
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to setup L2 weight regularization with layer pointers";
    throw lbann_exception(err.str());
  }

  // Add all weights in model if no weights pointers are provided
  if (m_weights.empty()) {
    for (weights* w : m.get_weights()) {
      if (w->get_optimizer() != nullptr) {
        m_weights.push_back(w);
      }
    }
  }

}

EvalType l2_weight_regularization::evaluate() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  EvalType sqsum = EvalType(0);
  for (const auto& w : m_weights) {
    cudnn::cudnn_manager* cudnn = w->get_cudnn_manager();
    if (cudnn != nullptr) {
    #ifndef LBANN_HAS_CUDNN
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: cuDNN not detected";
      throw lbann_exception(err.str());
    #else
      CHECK_CUDA(cudaSetDevice(cudnn->get_gpu(0)));
      const EvalType norm = cublas::nrm2(cudnn->get_cublas_handle(0),
                                         w->get_size(),
                                         w->get_values_gpu()[0], 1);
      sqsum += norm * norm;
    #endif // LBANN_HAS_CUDNN
    } else {
      // Further optimization: Can batch allreduces on the same communicator.
      const auto& values = w->get_values();
      const EvalType local_sqsum = sum_of_squares(values.LockedMatrix());
      sqsum += get_comm().allreduce(local_sqsum, values.DistComm());
    }
  }
  return m_scale_factor * sqsum / 2;
}

void l2_weight_regularization::compute_weight_regularization() {
  if (m_scale_factor == EvalType(0)) { return; }
  for (auto&& w : m_weights) {
    auto&& opt = w->get_optimizer();
    if (w->get_cudnn_manager() != nullptr) {
    #ifndef LBANN_HAS_CUDNN
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: cuDNN not detected";
      throw lbann_exception(err.str());
    #else
      cudnn::matrix values_d(w->get_cudnn_manager());
      auto&& values_ptrs = w->get_values_gpu();
      values_d.attach(values_ptrs, w->get_size());
      opt->add_to_gradient(values_d, m_scale_factor);
    #endif // LBANN_HAS_CUDNN
    } else {
      opt->add_to_gradient(w->get_values(), m_scale_factor);
    }
  }
}

} // namespace lbann
