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
#ifndef LBANN_SRC_LAYERS_TRANSFORM_CUTT_PERMUTEIMPL_HPP_INCLUDED
#define LBANN_SRC_LAYERS_TRANSFORM_CUTT_PERMUTEIMPL_HPP_INCLUDED

#include "lbann/base.hpp" // Elemental support.
#include "lbann/utils/exception.hpp"
#include "lbann/utils/typename.hpp"

#include "tensor_dims_utils.hpp"

#include <cutt.h>

#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

static char const* cutt_err_string(cuttResult err) noexcept
{
  switch (err) {
  case CUTT_SUCCESS:
    return "Success";
  case CUTT_INVALID_PLAN:
    return "Invalid plan handle";
  case CUTT_INVALID_PARAMETER:
    return "Invalid input parameter";
  case CUTT_INVALID_DEVICE:
    return "Execution tried on device different than where plan was created";
  case CUTT_INTERNAL_ERROR:
    return "Internal error";
  case CUTT_UNDEFINED_ERROR:
    return "Undefined error";
  default:
    return "<Unknown error value>";
  }
}

#define CHECK_CUTT(cmd)                                                        \
  do {                                                                         \
    cuttResult _check_cutt_err_result = (cmd);                                 \
    if (CUTT_SUCCESS != _check_cutt_err_result) {                              \
      LBANN_ERROR("cuTT operation \"" #cmd "\" FAILED (",                      \
                  cutt_err_string(_check_cutt_err_result),                     \
                  ")");                                                        \
    }                                                                          \
  } while (0)

namespace {

/** @brief cuTT-based implementation of tensor permute.
 *
 *  cuTT only supports packed tensors. LBANN currently assumes that
 *  all sample tensors are packed, so this is generally fine. However,
 *  LBANN allows the minibatch storage matrices to have a leading
 *  dimension that exceeds the height. This class handles this
 *  allowance by branching the code based on the relationship of the
 *  leading dimension and the height of the input/output
 *  matrices. When possible, it will create a transpose plan for the
 *  entire minibatch and launch a single kernel. When this is not
 *  possible, it will create a plan to transpose each sample and apply
 *  it sample-wise.
 *
 *  When a plan for the entire minibatch is possible, it will be
 *  required to create a new plan for any different minibatch size
 *  encountered.
 */
class cuTT_PermuteImpl
{
public:
  using DimsType = ColMajorDims<int>;

public:
  /** @name Lifecycle */
  ///@{

  cuTT_PermuteImpl(ColMajorPerm perm);
  ~cuTT_PermuteImpl() noexcept = default;
  ///@}
  /** @name Read-only Accessors (for testing) */
  ///@{

  ColMajorPerm const& perm() const noexcept;

  DimsType const& input_dims() const noexcept;
  DimsType const& output_dims() const noexcept;

  ///@}
  /** @name Permute interface */
  ///@{

  /** @brief Setup the dimensions.
   *
   *  Must be compatible with the provided perm vector.
   */
  void set_dims(DimsType input_dims);

  /** @brief Permute the tensor.
   *
   *  Applies the permutation to the tensor represented by "in". In
   *  line with the rest of LBANN, the permutation is applied to each
   *  column, which is treated as a packed tensor with the dimensions
   *  stored in this object.
   */
  template <typename DataT>
  void permute(El::Matrix<DataT, El::Device::GPU> const& in,
               El::Matrix<DataT, El::Device::GPU>& out) const;

  /** @brief Apply the inverse permutation to the tensor.
   *
   *  Applies the inverse permutation to the tensor represented by
   *  "in". In line with the rest of LBANN, the permutation is applied
   *  to each column, which is treated as a packed tensor with the
   *  dimensions stored in this object.
   */
  template <typename DataT>
  void inverse_permute(El::Matrix<DataT, El::Device::GPU> const& in,
                       El::Matrix<DataT, El::Device::GPU>& out) const;

  ///@}
  /** @name Modifiers */
  ///@{
  void swap(cuTT_PermuteImpl& other);
  ///@}

private:
  template <typename DataT>
  cuttHandle get_mb_plan(ColMajorPerm const& perm,
                         DimsType const& in_dims,
                         DimsType const& out_dims,
                         El::Matrix<DataT, El::Device::GPU> const& in,
                         El::Matrix<DataT, El::Device::GPU> const& out) const;

  template <typename DataT>
  cuttHandle
  get_sample_plan(ColMajorPerm const& perm,
                  DimsType const& in_dims,
                  DimsType const& out_dims,
                  El::Matrix<DataT, El::Device::GPU> const& in,
                  El::Matrix<DataT, El::Device::GPU> const& out) const;

  template <typename DataT>
  void do_mb_permute(ColMajorPerm const& perm,
                     DimsType const& in_dims,
                     DimsType const& out_dims,
                     El::Matrix<DataT, El::Device::GPU> const& in,
                     El::Matrix<DataT, El::Device::GPU>& out) const;

  template <typename DataT>
  void do_sample_permute(ColMajorPerm const& perm,
                         DimsType const& in_dims,
                         DimsType const& out_dims,
                         El::Matrix<DataT, El::Device::GPU> const& in,
                         El::Matrix<DataT, El::Device::GPU>& out) const;

private:
  ColMajorPerm m_perm;
  ColMajorPerm m_inv_perm;
  DimsType m_input_dims;
  DimsType m_output_dims;
}; // class cuTT_PermuteImpl

inline cuTT_PermuteImpl::cuTT_PermuteImpl(ColMajorPerm perm)
  : m_perm{std::move(perm)}, m_inv_perm{invert(m_perm)}
{
  LBANN_ASSERT_DEBUG(is_valid(m_perm));
  LBANN_ASSERT_DEBUG(is_valid(m_inv_perm));
}

inline auto cuTT_PermuteImpl::perm() const noexcept -> ColMajorPerm const&
{
  return m_perm;
}

inline auto cuTT_PermuteImpl::input_dims() const noexcept -> DimsType const&
{
  return m_input_dims;
}

inline auto cuTT_PermuteImpl::output_dims() const noexcept -> DimsType const&
{
  return m_output_dims;
}

inline void cuTT_PermuteImpl::set_dims(DimsType input_dims)
{
  m_input_dims = std::move(input_dims);
  m_output_dims = permute_dims(m_input_dims, m_perm);
}

template <typename DataT>
cuttHandle cuTT_PermuteImpl::get_mb_plan(
  ColMajorPerm const& perm,
  DimsType const& in_dims,
  DimsType const& out_dims,
  El::Matrix<DataT, El::Device::GPU> const& in,
  El::Matrix<DataT, El::Device::GPU> const& out) const
{
  LBANN_ASSERT_DEBUG(in.Width() == out.Width());
  LBANN_ASSERT_DEBUG(perm.size() == in_dims.size() &&
                     perm.size() == out_dims.size());

  std::vector<int> permutation(perm.get()), dimensions(in_dims.get());
  permutation.push_back(static_cast<int>(perm.size()));
  dimensions.push_back(in.Width());
  cuttHandle plan;
  CHECK_CUTT(cuttPlan(&plan,
                      dimensions.size(),
                      dimensions.data(),
                      permutation.data(),
                      sizeof(DataT),
                      out.GetSyncInfo().Stream()));
  return plan;
}

template <typename DataT>
cuttHandle cuTT_PermuteImpl::get_sample_plan(
  ColMajorPerm const& perm,
  DimsType const& in_dims,
  DimsType const& out_dims,
  El::Matrix<DataT, El::Device::GPU> const& in,
  El::Matrix<DataT, El::Device::GPU> const& out) const
{
  std::vector<int> permutation(perm.get()), dimensions(in_dims.get());
  cuttHandle sample_plan = 0U;
  CHECK_CUTT(cuttPlan(&sample_plan,
                      dimensions.size(),
                      dimensions.data(),
                      permutation.data(),
                      sizeof(DataT),
                      out.GetSyncInfo().Stream()));
  return sample_plan;
}

template <typename DataT>
void cuTT_PermuteImpl::permute(El::Matrix<DataT, El::Device::GPU> const& in,
                               El::Matrix<DataT, El::Device::GPU>& out) const
{
  if (in.Width() == El::Int{0})
    return;

  auto const& in_dims = m_input_dims;
  auto const& out_dims = m_output_dims;
  auto const& perm = m_perm;

  auto multisync =
    El::MakeMultiSync(El::SyncInfoFromMatrix(out), El::SyncInfoFromMatrix(in));

  if (in.LDim() == in.Height() && out.LDim() == out.Height() &&
      in.Width() > 1) {
    do_mb_permute(perm, in_dims, out_dims, in, out);
  }
  else {
    do_sample_permute(perm, in_dims, out_dims, in, out);
  }
}

template <typename DataT>
void cuTT_PermuteImpl::inverse_permute(
  El::Matrix<DataT, El::Device::GPU> const& in,
  El::Matrix<DataT, El::Device::GPU>& out) const
{
  if (in.Width() == El::Int{0})
    return;

  // Aliases to help this read better.
  auto const& in_dims = m_output_dims;
  auto const& out_dims = m_input_dims;
  auto const& perm = m_inv_perm;

  auto multisync =
    El::MakeMultiSync(El::SyncInfoFromMatrix(out), El::SyncInfoFromMatrix(in));

  if (in.LDim() == in.Height() && out.LDim() == out.Height() &&
      in.Width() > 1) {
    do_mb_permute(perm, in_dims, out_dims, in, out);
  }
  else {
    do_sample_permute(perm, in_dims, out_dims, in, out);
  }
}

template <typename DataT>
void cuTT_PermuteImpl::do_mb_permute(
  ColMajorPerm const& perm,
  DimsType const& in_dims,
  DimsType const& out_dims,
  El::Matrix<DataT, El::Device::GPU> const& in,
  El::Matrix<DataT, El::Device::GPU>& out) const
{
  auto const plan = get_mb_plan(perm, in_dims, out_dims, in, out);
  CHECK_CUTT(cuttExecute(plan,
                         const_cast<DataT*>(in.LockedBuffer()), // UGH
                         out.Buffer()));
  CHECK_CUTT(cuttDestroy(plan));
}

template <typename DataT>
void cuTT_PermuteImpl::do_sample_permute(
  ColMajorPerm const& perm,
  DimsType const& in_dims,
  DimsType const& out_dims,
  El::Matrix<DataT, El::Device::GPU> const& in,
  El::Matrix<DataT, El::Device::GPU>& out) const
{
  auto const plan = get_sample_plan(perm, in_dims, out_dims, in, out);
  DataT* const in_buf = const_cast<DataT*>(in.LockedBuffer()); // UGH
  DataT* const out_buf = out.Buffer();
  auto const batch_size = in.Width();
  auto const in_stride = in.LDim();
  auto const out_stride = out.LDim();
  for (El::Int sample = 0; sample < batch_size; ++sample) {
    CHECK_CUTT(cuttExecute(plan,
                           in_buf + sample * in_stride,
                           out_buf + sample * out_stride));
  }
  CHECK_CUTT(cuttDestroy(plan));
}

inline void cuTT_PermuteImpl::swap(cuTT_PermuteImpl& other)
{
  std::swap(m_perm, other.m_perm);
  std::swap(m_inv_perm, other.m_inv_perm);
  std::swap(m_input_dims, other.m_input_dims);
  std::swap(m_output_dims, other.m_output_dims);
}

} // namespace
#endif // LBANN_SRC_LAYERS_TRANSFORM_CUTT_PERMUTEIMPL_HPP_INCLUDED
