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
#include "lbann/utils/gpu/helpers.hpp"
#endif // LBANN_HAS_GPU
#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/weights/data_type_weights.hpp"
#include "lbann/utils/h2_tmp.hpp"

namespace lbann {

template <>
void l2_weight_regularization::accumulate_contribution<El::Device::CPU>(const CPUMatType& vals,
                                                                        CPUMatType& contribution) {
  auto& sqsum = contribution(0, 0);
  if (!vals.IsEmpty()) {
    if (vals.Contiguous()) {
      const size_t size = vals.Height() * vals.Width();
      const auto& __restrict__ vals_buf = vals.LockedBuffer();
      LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:sqsum))
      for (size_t i = 0; i < size; ++i) {
        const auto& val = vals_buf[i];
        sqsum += val * val;
      }
    }
    else {
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
}

l2_weight_regularization::l2_weight_regularization(EvalType scale_factor)
  : objective_function_term(scale_factor) {}

void l2_weight_regularization::setup(model& m) {
  objective_function_term::setup(m);

  // Check that term has no layer pointers
  if (!m_layers.empty()) {
    LBANN_ERROR("attempted to setup L2 weight regularization "
                "with no layer pointers");
  }

  // Add all weights in model if no weights pointers are provided
  if (m_weights.empty()) {
    m_weights = m.get_weights();
  }

  // Construct accumulation variables for each device
  for (auto* w : m_weights) {
    const auto& device = dynamic_cast<WeightsType*>(w)->get_values().GetLocalDevice();
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
      const auto& vals = dynamic_cast<WeightsType*>(m_weights[i])->get_values();
      if (vals.GetLocalDevice() == El::Device::CPU
          && vals.Participating()
          && vals.RedundantRank() == i % vals.RedundantSize()) {
        accumulate_contribution<El::Device::CPU>(
          static_cast<const CPUMatType&>(vals.LockedMatrix()),
          contribution);
      }
    }
    get_comm().nb_allreduce(static_cast<El::AbstractMatrix<AccumulateDataType>&>(contribution),
                            get_comm().get_trainer_comm(),
                            m_allreduce_req);
  }

#ifdef LBANN_HAS_GPU
  // Compute contributions from GPU weights
  if (m_contributions.count(El::Device::GPU) > 0) {
    DMatType<El::Device::GPU> contribution;
#ifdef HYDROGEN_HAVE_CUB
    contribution.SetMemoryMode(1); // CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
    auto sync_info = gpu::get_sync_info(contribution);
    El::Zeros(contribution, 1, 1);
    for (El::Int i = 0; i < num_weights; ++i) {
      const auto& vals = dynamic_cast<WeightsType*>(m_weights[i])->get_values();
      if (vals.GetLocalDevice() == El::Device::GPU
          && vals.Participating()
          && vals.RedundantRank() == i % vals.RedundantSize()) {
        accumulate_contribution<El::Device::GPU>(
          static_cast<const DMatType<El::Device::GPU>&>(vals.LockedMatrix()),
          contribution);
      }
    }
    get_comm().allreduce(static_cast<El::AbstractMatrix<AccumulateDataType>&>(contribution),
                         get_comm().get_trainer_comm());
    ::hydrogen::gpu::Copy1DToHost(contribution.LockedBuffer(),
                                  m_contributions[El::Device::GPU].Buffer(),
                                  1,
                                  sync_info);
    m_copy_event.record(sync_info.Stream());
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

// Somewhat hacky approach to avoid a BaseDistMat implementation of
// optimizer::add_to_gradient, since this is literally the only
// use-case. Given the line count, this was clearly the way to go... :/
namespace {
struct AddToGrad {
  AddToGrad(optimizer& opt, EvalType scale_factor)
    : opt_{&opt},
      scale_{scale_factor}
  {}
  void DispatchError(El::BaseDistMatrix const&) {
    LBANN_ERROR("Unable to dispatch!");
  }
  void DeductionError(El::BaseDistMatrix const&) {
    LBANN_ERROR("Unable to deduce type!");
  }
  template <typename T>
  void operator()(El::AbstractDistMatrix<T> const& contrib) {
    opt_->add_to_gradient(contrib, El::To<T>(scale_));
  }
  optimizer* opt_;
  EvalType scale_;
};// struct AddToGrad

using ValidDataTypes = h2::meta::TL<
#ifdef LBANN_HAS_GPU_FP16
  fp16,
#endif
#ifdef LBANN_HAS_HALF
  cpu_fp16,
#endif
  float, double>;

using MatTypes =
  h2::meta::tlist::ExpandTL<El::AbstractDistMatrix, ValidDataTypes>;

using DispatcherType =
  h2::multimethods::SwitchDispatcher<AddToGrad,
                                     void,
                                     El::BaseDistMatrix, MatTypes>;
}

void l2_weight_regularization::compute_weight_regularization() {
  if (m_scale_factor == EvalType(0)) { return; }
  for (auto* w : m_weights) {
    auto* opt = w->get_optimizer();
    if (opt != nullptr) {
      DispatcherType::Exec(AddToGrad(*opt, m_scale_factor), w->get_values());
    }
  }
}

} // namespace lbann
