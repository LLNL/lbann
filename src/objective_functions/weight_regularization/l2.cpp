////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
#ifdef LBANN_HAS_GPU
#include "lbann/utils/cublas.hpp"
#endif // LBANN_HAS_GPU

namespace lbann {

template <>
void l2_weight_regularization::accumulate_contribution<El::Device::CPU>(const CPUMat& vals,
                                                                        CPUMat& contribution) {
  auto& sqsum = contribution(0, 0);
  if (vals.IsEmpty()) {
  } else if (vals.Contiguous()) {
    const size_t size = vals.Height() * vals.Width();
    const auto& __restrict__ vals_buf = vals.LockedBuffer();
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:sqsum))
    for (size_t i = 0; i < size; ++i) {
      const auto& val = vals_buf[i];
      sqsum += val * val;
    }
  } else {
    const El::Int height = vals.Height();
    const El::Int width = vals.Width();
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:sqsum) collapse(2))
    for (El::Int col = 0; col < width; ++col) {
      for (El::Int row = 0; row < height; ++row) {
        const EvalType val = vals(row, col);
        sqsum += val * val;
      }
    }
  }
}

l2_weight_regularization::l2_weight_regularization(EvalType scale_factor)
  : objective_function_term(scale_factor) {}

void l2_weight_regularization::setup(model& m) {
  objective_function_term::setup(m);

  // Check that term has no layer pointers
  if (!m_layers.empty()) {
    LBANN_ERROR("attempted to setup L2 weight regularization with layer pointers");
  }

  // Add all weights in model if no weights pointers are provided
  if (m_weights.empty()) {
    m_weights = m.get_weights();
  }

  // Construct accumulation variables for each device
  for (auto* w : m_weights) {
    const auto& device = w->get_values().GetLocalDevice();
    if (m_contributions.count(device) == 0) {
#ifdef LBANN_HAS_GPU
      m_contributions[device].SetMemoryMode(1); // Pinned memory
#endif // LBANN_HAS_GPU
      m_contributions[device].Resize(1, 1);
    }
  }

}

void l2_weight_regularization::start_evaluation() {
  if (m_scale_factor == EvalType(0)) { return; }
  const El::Int num_weights = m_weights.size();

  // Compute contributions from CPU weights
  if (m_contributions.count(El::Device::CPU) > 0) {
    auto& contribution = m_contributions[El::Device::CPU];
    contribution(0, 0) = DataType(0);
    for (El::Int i = 0; i < num_weights; ++i) {
      const auto& vals = m_weights[i]->get_values();
      if (vals.GetLocalDevice() == El::Device::CPU
          && vals.Participating()
          && vals.RedundantRank() == i % vals.RedundantSize()) {
        accumulate_contribution<El::Device::CPU>(
          static_cast<const CPUMat&>(vals.LockedMatrix()),
          contribution);
      }
    }
    get_comm().nb_allreduce(static_cast<AbsMat&>(contribution),
                            get_comm().get_trainer_comm(),
                            m_allreduce_req);
  }

#ifdef LBANN_HAS_GPU
  // Compute contributions from GPU weights
  if (m_contributions.count(El::Device::GPU) > 0) {
    auto&& stream = El::GPUManager::Stream();
    GPUMat contribution;
#ifdef HYDROGEN_HAVE_CUB
    contribution.SetMemoryMode(1); // CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
    El::Zeros(contribution, 1, 1);
    for (El::Int i = 0; i < num_weights; ++i) {
      const auto& vals = m_weights[i]->get_values();
      if (vals.GetLocalDevice() == El::Device::GPU
          && vals.Participating()
          && vals.RedundantRank() == i % vals.RedundantSize()) {
        accumulate_contribution<El::Device::GPU>(
          static_cast<const GPUMat&>(vals.LockedMatrix()),
          contribution);
      }
    }
    get_comm().allreduce(static_cast<AbsMat&>(contribution),
                         get_comm().get_trainer_comm());
    CHECK_CUDA(cudaMemcpyAsync(m_contributions[El::Device::GPU].Buffer(),
                               contribution.LockedBuffer(),
                               sizeof(DataType),
                               cudaMemcpyDeviceToHost,
                               stream));
    m_copy_event.record(stream);
  }
#endif // LBANN_HAS_GPU

}

EvalType l2_weight_regularization::finish_evaluation() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  EvalType sqsum = 0;
  if (m_contributions.count(El::Device::CPU) > 0) {
    get_comm().wait(m_allreduce_req);
    sqsum += m_contributions[El::Device::CPU](0, 0);
  }
#ifdef LBANN_HAS_GPU
  if (m_contributions.count(El::Device::GPU) > 0) {
    m_copy_event.synchronize();
    sqsum += m_contributions[El::Device::GPU](0, 0);
  }
#endif // LBANN_HAS_GPU
  return m_scale_factor * sqsum / 2;
}

void l2_weight_regularization::compute_weight_regularization() {
  if (m_scale_factor == EvalType(0)) { return; }
  for (auto&& w : m_weights) {
    auto&& opt = w->get_optimizer();
    if (opt != nullptr) {
      opt->add_to_gradient(w->get_values(), m_scale_factor);
    }
  }
}

} // namespace lbann
