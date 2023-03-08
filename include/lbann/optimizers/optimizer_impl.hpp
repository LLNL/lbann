////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

namespace lbann {

template <typename TensorDataType>
void optimizer::add_to_gradient(
  El::AbstractDistMatrix<TensorDataType> const& contrib,
  TensorDataType scale,
  bool allreduce_needed)
{
  TensorDataType buf_scale, in_scale;
  auto& grad = get_gradient_buffer(buf_scale, in_scale, allreduce_needed);
  El::Scale(buf_scale, grad);
  El::Axpy(in_scale * scale, contrib, grad);
}

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>&
optimizer::get_gradient_buffer(TensorDataType& buf_scale,
                               TensorDataType& in_scale,
                               bool allreduce_needed)
{

  // Anon enum to clarify "get<#>" calls below.
  enum
  {
    HEIGHT = 0,
    WIDTH,
    DISTDATA
  };
  using GradMgrType = GradientHelperImpl<TensorDataType>;

  auto& grad_mgr_ptr = gradients_[std::type_index(typeid(TensorDataType))];
  // If the manager hasn't been created, let's make it.
  if (!grad_mgr_ptr) {
    auto mat_info = this->get_matrix_info();
    grad_mgr_ptr = std::make_unique<GradMgrType>(std::get<HEIGHT>(mat_info),
                                                 std::get<WIDTH>(mat_info),
                                                 std::get<DISTDATA>(mat_info));
    grad_mgr_ptr->set_status(optimizer_gradient_status::cleared);
  }
  // Get the underlying matrix back out.
  auto& grad_mgr = static_cast<GradMgrType&>(*grad_mgr_ptr);
  // Complete outstanding allreduce, if needed.
  if (grad_mgr.get_status() == optimizer_gradient_status::allreduce_started) {
    grad_mgr.complete_allreduce(*(this->m_comm));
  }
  auto& buffer = grad_mgr.gradient();

  // Determine scaling factor and transition state.
  switch (grad_mgr.get_status()) {
  case optimizer_gradient_status::ready:
    buf_scale = DataType(1);
    in_scale = DataType(1);
    if (allreduce_needed) {
      buf_scale /= buffer.RedundantSize();
      grad_mgr.set_status(optimizer_gradient_status::allreduce_needed);
    }
    break;
  case optimizer_gradient_status::cleared:
    buf_scale = DataType(0);
    in_scale = DataType(1);
    grad_mgr.set_status(allreduce_needed
                          ? optimizer_gradient_status::allreduce_needed
                          : optimizer_gradient_status::ready);
    break;
  case optimizer_gradient_status::allreduce_needed:
    buf_scale = DataType(1);
    // Properly scale data that does not need to be allreduced.
    in_scale =
      (allreduce_needed ? DataType(1) : DataType(1) / buffer.RedundantSize());
    break;
  case optimizer_gradient_status::allreduce_started:
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
  auto num_updates = this->gradients_.size();
  auto const this_type_idx = std::type_index(typeid(TensorDataType));

  if (num_updates == 0UL)
    return;

  // Handle the case that one of the updates is TensorDataType. In
  // this case, the input gradients matrix can be made to "view" the
  // update, rather than requiring a copy.
  auto this_type_contrib = this->gradients_.find(this_type_idx);
  if (this_type_contrib != this->gradients_.end()) {
    // Check for invariant consistency.
    auto const& grad_mgr = *(this_type_contrib->second);
    if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
      LBANN_ERROR("Expected ready status. Got: ",
                  to_string(grad_mgr.get_status()));
    }
    // Sync the input gradient with the contribution, one way or another.
    auto const& contrib =
      dynamic_cast<AbsDistMatType const&>(grad_mgr.gradient());
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
  if (num_updates == 1UL && this->gradients_.size() == 1UL) {
    auto const& grad_mgr = *(this->gradients_.begin()->second);
    if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
      LBANN_ERROR("Expected ready status. Got: ",
                  to_string(grad_mgr.get_status()));
    }
    El::Copy(grad_mgr.gradient(), gradient);
  }
  else if (this->gradients_.size() > 1UL) {
    // Need a temporary matrix for the type-casted copy.
    auto tmp = std::unique_ptr<AbsDistMatType>{
      gradient.Construct(gradient.Grid(), gradient.Root())};

    for (auto const& grad_mgr_v : this->gradients_) {
      if (grad_mgr_v.first == this_type_idx)
        continue;
      auto const& grad_mgr = *(grad_mgr_v.second);
      if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
        LBANN_ERROR("Expected ready status. Got: ",
                    to_string(grad_mgr.get_status()));
      }
      auto const& grad_base = grad_mgr.gradient();
      El::Copy(grad_base, *tmp);
      El::Axpy(one, *tmp, gradient);
    }
  }
}

template <typename TensorDataType>
void optimizer::GradientHelperImpl<TensorDataType>::start_allreduce(
  lbann_comm& comm)
{
  switch (this->get_status()) {
  case optimizer_gradient_status::allreduce_needed:
    comm.nb_allreduce(*gradient_, gradient_->RedundantComm(), allreduce_req_);
    this->set_status(optimizer_gradient_status::allreduce_started);
    break;
  case optimizer_gradient_status::ready:
  case optimizer_gradient_status::cleared:
  case optimizer_gradient_status::allreduce_started:
    break;
  default:
    LBANN_ERROR("unexpected gradient status "
                "(" +
                to_string(this->get_status()) + ")");
  }
}

template <typename TensorDataType>
void optimizer::GradientHelperImpl<TensorDataType>::complete_allreduce(
  lbann_comm& comm)
{
  switch (this->get_status()) {
  case optimizer_gradient_status::allreduce_started:
    comm.wait(allreduce_req_);
    this->set_status(optimizer_gradient_status::ready);
    break;
  case optimizer_gradient_status::ready:
  case optimizer_gradient_status::cleared:
    break;
  case optimizer_gradient_status::allreduce_needed:
    LBANN_ERROR("attempted to finish gradient allreduce "
                "before starting it");
    break;
  default:
    LBANN_ERROR("unexpected gradient status "
                "(" +
                to_string(this->get_status()) + ")");
  }
}

template <typename TensorDataType>
void optimizer::GradientHelperImpl<TensorDataType>::clear()
{
  this->set_status(optimizer_gradient_status::cleared);
}

} // namespace lbann

#endif // LBANN_OPTIMIZERS_OPTIMIZER_IMPL_HPP_INCLUDED
