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
#ifndef LBANN_SRC_LAYERS_TRANSFORM_CUTENSOR_PERMUTEIMPL_HPP_INCLUDED
#define LBANN_SRC_LAYERS_TRANSFORM_CUTENSOR_PERMUTEIMPL_HPP_INCLUDED

#include "lbann/base.hpp" // Elemental support.
#include "lbann/utils/exception.hpp"
#include "lbann/utils/typename.hpp"

#include "tensor_dims_utils.hpp"

// This is only separated to make it easier if we "formally accept"
// cuTENSOR like we have cuDNN or cuBLAS, etc.
#include "cutensor_support.hpp"

#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lbann {

/** @brief cuTENSOR-based implementation of tensor permute.
 *
 *  Because cuTENSOR is primarily column-major-centric (I actually
 *  prefer it, too, but that's neither here nor there), this caches as
 *  much as possible to avoid the annoying recomputation of dims,
 *  strides, etc.
 */
class cuTENSOR_PermuteImpl
{
public:
  using DimsType = ColMajorDims<int64_t>;
  using StridesType = ColMajorStrides<int64_t>;
  using ModesType = std::vector<int32_t>;

public:
  /** @name Lifecycle */
  ///@{

  cuTENSOR_PermuteImpl(ColMajorPerm perm);

  ///@}
  /** @name Read-only Accessors (for testing) */
  ///@{

  ColMajorPerm const& perm() const noexcept;

  DimsType const& input_dims() const noexcept;
  DimsType const& output_dims() const noexcept;

  ModesType const& input_modes() const noexcept;
  ModesType const& output_modes() const noexcept;

  StridesType input_strides() const;
  StridesType output_strides() const;

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
   *
   *  @note (trb) I think this can be extended to support data type
   *  conversions during permutation, but I haven't tested this and
   *  there's currently not a known need.
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
  void swap(cuTENSOR_PermuteImpl& other);
  ///@}

private:
  /** @brief Keep track of descriptors so we don't have to repeatedly
   *         rebuild them.
   */
  inline static std::unordered_map<std::string, cutensorTensorDescriptor_t>
    m_desc_map;

private:
  static std::vector<int32_t> make_modes(size_t const ndims);
  template <typename DataT>
  static std::string get_desc_key(El::Matrix<DataT, El::Device::GPU> const& mat,
                                  DimsType const& dims);
  template <typename DataT>
  static cutensorTensorDescriptor_t
  get_descriptor(El::Matrix<DataT, El::Device::GPU> const& mat,
                 DimsType const& dims);

private:
  ColMajorPerm m_perm;
  DimsType m_input_dims;
  DimsType m_output_dims;
  ModesType m_input_modes;
  ModesType m_output_modes;
}; // class cuTENSOR_PermuteImpl

inline cuTENSOR_PermuteImpl::cuTENSOR_PermuteImpl(ColMajorPerm perm)
  : m_perm{std::move(perm)},
    m_input_modes{make_modes(m_perm.size())},
    m_output_modes{permute_impl(m_input_modes, m_perm.get())}
{
  LBANN_ASSERT_DEBUG(is_valid(m_perm));
}

inline auto cuTENSOR_PermuteImpl::perm() const noexcept -> ColMajorPerm const&
{
  return m_perm;
}

inline auto cuTENSOR_PermuteImpl::input_dims() const noexcept -> DimsType const&
{
  return m_input_dims;
}

inline auto cuTENSOR_PermuteImpl::output_dims() const noexcept
  -> DimsType const&
{
  return m_output_dims;
}

inline auto cuTENSOR_PermuteImpl::input_modes() const noexcept
  -> ModesType const&
{
  return m_input_modes;
}

inline auto cuTENSOR_PermuteImpl::output_modes() const noexcept
  -> ModesType const&
{
  return m_output_modes;
}

inline auto cuTENSOR_PermuteImpl::input_strides() const -> StridesType
{
  return get_strides(m_input_dims);
}

inline auto cuTENSOR_PermuteImpl::output_strides() const -> StridesType
{
  return get_strides(m_output_dims);
}

inline auto cuTENSOR_PermuteImpl::make_modes(size_t const ndims) -> ModesType
{
  std::vector<int32_t> modes(ndims + 1); // Add the sample dim.
  std::iota(begin(modes), end(modes), static_cast<int>('a'));
  return modes;
}

template <typename DataT>
std::string cuTENSOR_PermuteImpl::get_desc_key(
  El::Matrix<DataT, El::Device::GPU> const& mat,
  DimsType const& dims_in)
{
  auto const& dims = dims_in.get();
  std::ostringstream oss;
  oss << mat.Height() << "," << mat.Width() << "," << mat.LDim() << ";"
      << dims.front();
  for (size_t ii = 1; ii < dims.size(); ++ii)
    oss << "," << dims[ii];
  oss << ";" << lbann::TypeName<DataT>();
  return oss.str();
}

template <typename DataT>
cutensorTensorDescriptor_t cuTENSOR_PermuteImpl::get_descriptor(
  El::Matrix<DataT, El::Device::GPU> const& mat,
  DimsType const& dims)
{
  auto key = get_desc_key(mat, dims); // captures Width to account for
                                      // minibatch size and LDim to
                                      // account for stride.
  auto iter = m_desc_map.find(key);
  if (iter == end(m_desc_map)) {
    std::vector<int64_t> extents = dims.get();
    extents.push_back(mat.Width()); // Don't forget MB size

    auto strides = get_strides(dims);
    strides.get().push_back(mat.LDim()); // Don't forget sample stride.

    cutensorTensorDescriptor_t desc;
    CHECK_CUTENSOR(cutensorInitTensorDescriptor(get_handle_ptr(),
                                                &desc,
                                                extents.size(),
                                                extents.data(),
                                                strides.get().data(),
                                                CUDAType<DataT>,
                                                CUTENSOR_OP_IDENTITY));
    m_desc_map.emplace(std::move(key), desc);
    return desc;
  }
  return iter->second;
}

inline void cuTENSOR_PermuteImpl::set_dims(DimsType input_dims)
{
  m_input_dims = std::move(input_dims);
  m_output_dims = permute_dims(m_input_dims, m_perm);
}

template <typename DataT>
void cuTENSOR_PermuteImpl::permute(
  El::Matrix<DataT, El::Device::GPU> const& in,
  El::Matrix<DataT, El::Device::GPU>& out) const
{
  auto const in_desc = get_descriptor(in, m_input_dims);
  auto const out_desc = get_descriptor(out, m_output_dims);

  auto const one = El::To<CUDAScalar<DataT>>(1.f);
  auto multisync =
    El::MakeMultiSync(El::SyncInfoFromMatrix(out), El::SyncInfoFromMatrix(in));

  // This permutation is input_modes -> output_modes.
  CHECK_CUTENSOR(cutensorPermutation(
    get_handle_ptr(),
    &one,
    in.LockedBuffer(),
    &in_desc,
    m_input_modes.data(),
    out.Buffer(),
    &out_desc,
    m_output_modes.data(),
    CUDAScalarType<DataT>,
    static_cast<El::SyncInfo<El::Device::GPU>>(multisync).Stream()));
}

template <typename DataT>
void cuTENSOR_PermuteImpl::inverse_permute(
  El::Matrix<DataT, El::Device::GPU> const& in,
  El::Matrix<DataT, El::Device::GPU>& out) const
{
  // This permutation is output_modes -> input_modes. Use some aliases
  // to help.
  auto const& in_dims = m_output_dims;
  auto const& out_dims = m_input_dims;

  auto const& in_modes = m_output_modes;
  auto const& out_modes = m_input_modes;

  // This part matches the regular "permute" function
  auto const in_desc = get_descriptor(in, in_dims);
  auto const out_desc = get_descriptor(out, out_dims);

  auto const one = El::To<CUDAScalar<DataT>>(1.f);
  auto multisync =
    El::MakeMultiSync(El::SyncInfoFromMatrix(out), El::SyncInfoFromMatrix(in));

  CHECK_CUTENSOR(cutensorPermutation(
    get_handle_ptr(),
    &one,
    in.LockedBuffer(),
    &in_desc,
    in_modes.data(),
    out.Buffer(),
    &out_desc,
    out_modes.data(),
    CUDAScalarType<DataT>,
    static_cast<El::SyncInfo<El::Device::GPU>>(multisync).Stream()));
}

inline void cuTENSOR_PermuteImpl::swap(cuTENSOR_PermuteImpl& other)
{
  std::swap(m_perm, other.m_perm);
  std::swap(m_input_dims, other.m_input_dims);
  std::swap(m_output_dims, other.m_output_dims);
  std::swap(m_input_modes, other.m_input_modes);
  std::swap(m_output_modes, other.m_output_modes);
}

} // namespace lbann
#endif // LBANN_SRC_LAYERS_TRANSFORM_CUTENSOR_PERMUTEIMPL_HPP_INCLUDED
