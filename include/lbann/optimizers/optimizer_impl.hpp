////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_OPTIMIZERS_OPTIMIZER_IMPL_HPP_INCLUDED
#define LBANN_OPTIMIZERS_OPTIMIZER_IMPL_HPP_INCLUDED

#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/profiling.hpp"

namespace lbann {

template <typename TensorDataType>
class GradientHelperImpl : public optimizer::GradientHelper
{
public:
  using AbsDistMatType = El::AbstractDistMatrix<TensorDataType>;

public:
  GradientHelperImpl(El::Int height,
                     El::Int width,
                     El::DistData dist_data,
                     El::DistData grad_dist_data,
                     bool sharded_weights)
    : local_gradient_contrib_{AbsDistMatType::Instantiate(dist_data)},
      local_contrib_dist_{dist_data},
      global_gradient_{AbsDistMatType::Instantiate(grad_dist_data)},
      global_dist_{grad_dist_data},
      sharded_weights_{sharded_weights}
  {
    ensure_gradient_memory(height, width);
    El::Zeros(*local_gradient_contrib_, height, width);
    if (grad_dist_data != dist_data) {
      El::Zeros(*global_gradient_, height, width);
    }
  }

  void ensure_gradient_memory(El::Int height, El::Int width) override
  {
#if defined(LBANN_HAS_GPU)
    local_gradient_contrib_->Matrix().SetMemoryMode(1);
#endif // LBANN_HAS_GPU

    if (local_gradient_contrib_->Width() == 0) {
      local_gradient_contrib_->Resize(height, width);
      // If distribution is the same, have global gradient matrix view the
      // local contributions.
      if (local_contrib_dist_ == global_dist_) {
        El::View(*global_gradient_, *local_gradient_contrib_);
      }
    }
  }

  AbsDistMatType& local_gradient() noexcept override
  {
    return *local_gradient_contrib_;
  }

  AbsDistMatType const& local_gradient() const noexcept override
  {
    return *local_gradient_contrib_;
  }

  AbsDistMatType& global_gradient() noexcept override
  {
    return *global_gradient_;
  }

  AbsDistMatType const& global_gradient() const noexcept override
  {
    return *global_gradient_;
  }

  void start_sync(lbann_comm& comm) override
  {
    switch (this->get_status()) {
    case optimizer_gradient_status::sync_needed:
      // Sharded gradients are produced from a reduce-scatter on the local
      // contributions, non-sharded gradients use allreduce
      if (!sharded_weights_) {
        comm.nb_allreduce(*global_gradient_,
                          global_gradient_->RedundantComm(),
                          sync_req_);
      }
      else {
        // Reduce the local contributions and scatter the results
        // to the global gradient. The implementation below uses
        // allreduce temporarily.
        comm.nb_allreduce(*local_gradient_contrib_,
                          local_gradient_contrib_->RedundantComm(),
                          sync_req_);

        // TODO: Call reduce-scatter directly (currently fails)
        /*
        comm.nb_reduce_scatter(*local_gradient_contrib_,
                               *global_gradient_,
                               global_gradient_->RedundantComm(),
                               sync_req_);
        */
      }
      this->set_status(optimizer_gradient_status::sync_started);
      break;
    case optimizer_gradient_status::ready:
    case optimizer_gradient_status::cleared:
    case optimizer_gradient_status::sync_started:
      break;
    default:
      LBANN_ERROR("unexpected gradient status "
                  "(" +
                  to_string(this->get_status()) + ")");
    }
  }

  void complete_sync(lbann_comm& comm) override
  {
    switch (this->get_status()) {
    case optimizer_gradient_status::sync_started:
      comm.wait(sync_req_);
      if (sharded_weights_) {
        // TODO: When reduce-scatter is called in start_sync, remove this copy
        El::Copy(*local_gradient_contrib_, *global_gradient_);

        // Free up memory
        local_gradient_contrib_->Empty();
      }

      this->set_status(optimizer_gradient_status::ready);
      break;
    case optimizer_gradient_status::ready:
    case optimizer_gradient_status::cleared:
      break;
    case optimizer_gradient_status::sync_needed:
      LBANN_ERROR("attempted to finish gradient sync "
                  "before starting it");
      break;
    default:
      LBANN_ERROR("unexpected gradient status "
                  "(" +
                  to_string(this->get_status()) + ")");
    }
  }

  void clear() override
  {
    this->set_status(optimizer_gradient_status::cleared);
  }

private:
  /** Matches the distribution of gathered (unsharded) weights in backprop. */
  std::unique_ptr<AbsDistMatType> local_gradient_contrib_;
  El::DistData local_contrib_dist_;

  /** Matches the distribution of data_type_optimizer<T>::m_gradient (i.e.,
   *  post synchronization). Will view said matrix if only one data type
   *  exists.
   */
  std::unique_ptr<AbsDistMatType> global_gradient_;
  El::DistData global_dist_;

  Al::request sync_req_;
  bool sharded_weights_;
}; // class GradientHelperImpl

template <typename TensorDataType>
void optimizer::add_to_gradient(
  El::AbstractDistMatrix<TensorDataType> const& contrib,
  TensorDataType scale,
  bool sync_needed)
{
  TensorDataType buf_scale, in_scale;
  auto& grad = get_gradient_buffer(buf_scale, in_scale, sync_needed);
  El::Scale(buf_scale, grad);
  El::Axpy(in_scale * scale, contrib, grad);
}

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>&
optimizer::get_gradient_buffer(TensorDataType& buf_scale,
                               TensorDataType& in_scale,
                               bool sync_needed)
{

  // Anon enum to clarify "get<#>" calls below.
  enum
  {
    HEIGHT = 0,
    WIDTH,
    DISTDATA_L,
    DISTDATA_G,
  };
  using GradMgrType = GradientHelperImpl<TensorDataType>;

  auto& grad_mgr_ptr =
    m_local_gradient_contributions[std::type_index(typeid(TensorDataType))];
  // If the manager hasn't been created, let's make it.
  auto mat_info = this->get_matrix_info();
  if (!grad_mgr_ptr) {
    grad_mgr_ptr = std::make_unique<GradMgrType>(std::get<HEIGHT>(mat_info),
                                                 std::get<WIDTH>(mat_info),
                                                 std::get<DISTDATA_L>(mat_info),
                                                 std::get<DISTDATA_G>(mat_info),
                                                 this->is_sharded());
    grad_mgr_ptr->set_status(optimizer_gradient_status::cleared);
  }
  grad_mgr_ptr->ensure_gradient_memory(std::get<HEIGHT>(mat_info),
                                       std::get<WIDTH>(mat_info));

  // Get the underlying matrix back out.
  auto& grad_mgr = static_cast<GradMgrType&>(*grad_mgr_ptr);
  // Complete outstanding sync, if needed.
  if (grad_mgr.get_status() == optimizer_gradient_status::sync_started) {
    grad_mgr.complete_sync(*(this->m_comm));
  }
  auto& buffer = grad_mgr.local_gradient();

  // Determine scaling factor and transition state.
  switch (grad_mgr.get_status()) {
  case optimizer_gradient_status::ready:
    buf_scale = DataType(1);
    in_scale = DataType(1);
    if (sync_needed) {
      buf_scale /= buffer.RedundantSize();
      grad_mgr.set_status(optimizer_gradient_status::sync_needed);
    }
    break;
  case optimizer_gradient_status::cleared:
    buf_scale = DataType(0);
    in_scale = DataType(1);
    grad_mgr.set_status(sync_needed ? optimizer_gradient_status::sync_needed
                                    : optimizer_gradient_status::ready);
    break;
  case optimizer_gradient_status::sync_needed:
    buf_scale = DataType(1);
    // Properly scale data that does not need to be synchronized.
    in_scale =
      (sync_needed ? DataType(1) : DataType(1) / buffer.RedundantSize());
    break;
  case optimizer_gradient_status::sync_started:
  default:
    LBANN_ERROR("unexpected gradient status (" +
                to_string(grad_mgr.get_status()) + ")");
  }
  return buffer;
}

template <typename TensorDataType>
void optimizer::accumulate_all_gradient_contributions(
  El::AbstractDistMatrix<TensorDataType>& gradient)
{
  using AbsDistMatType = El::AbstractDistMatrix<TensorDataType>;
  static const TensorDataType one = TensorDataType(1.f);

  // There are a few cases to note here:
  //   1. One update of the same type.
  //   2. One update of a different type.
  //   3. Multiple updates of multiple types. In this case, some work
  //      can be saved if one of the updates has the same type as
  //      "gradient".

  // Some general information
  auto num_updates = this->m_local_gradient_contributions.size();
  auto const this_type_idx = std::type_index(typeid(TensorDataType));

  if (num_updates == 0UL)
    return;

  // Handle the case that one of the updates is TensorDataType. In
  // this case, the input gradients matrix can be made to "view" the
  // update, rather than requiring a copy.
  auto this_type_contrib =
    this->m_local_gradient_contributions.find(this_type_idx);
  if (this_type_contrib != this->m_local_gradient_contributions.end()) {
    // Check for invariant consistency.
    auto const& grad_mgr = *(this_type_contrib->second);
    if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
      LBANN_ERROR("Expected ready status. Got: ",
                  to_string(grad_mgr.get_status()));
    }
    // Sync the input gradient with the contribution, one way or another.
    auto const& contrib =
      dynamic_cast<AbsDistMatType const&>(grad_mgr.global_gradient());
    if (contrib.DistData() == gradient.DistData()) {
      El::LockedView(gradient, contrib);
    }
    else {
      LBANN_ERROR("Should never need this copy.");
      El::Copy(contrib, gradient);
    }
    --num_updates;
  }
  else {
    // No sync possible; zero out the matrix instead
    El::Zero(gradient);
  }

  // Handle the case that only 1 update of a different type is needed.
  if (num_updates == 1UL &&
      this->m_local_gradient_contributions.size() == 1UL) {
    auto const& grad_mgr =
      *(this->m_local_gradient_contributions.begin()->second);
    if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
      LBANN_ERROR("Expected ready status. Got: ",
                  to_string(grad_mgr.get_status()));
    }
    El::Copy(grad_mgr.global_gradient(), gradient);
  }
  else if (this->m_local_gradient_contributions.size() > 1UL) {
    // Need a temporary matrix for the type-casted copy.
    auto tmp = std::unique_ptr<AbsDistMatType>{
      gradient.Construct(gradient.Grid(), gradient.Root())};

    for (auto const& grad_mgr_v : this->m_local_gradient_contributions) {
      if (grad_mgr_v.first == this_type_idx)
        continue;
      auto const& grad_mgr = *(grad_mgr_v.second);
      if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
        LBANN_ERROR("Expected ready status. Got: ",
                    to_string(grad_mgr.get_status()));
      }
      auto const& grad_base = grad_mgr.global_gradient();
      El::Copy(grad_base, *tmp);
      El::Axpy(one, *tmp, gradient);
    }
  }
}

} // namespace lbann

#endif // LBANN_OPTIMIZERS_OPTIMIZER_IMPL_HPP_INCLUDED
