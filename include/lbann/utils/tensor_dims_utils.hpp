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
#ifndef LBANN_UTILS_TENSOR_DIMS_UTILS_HPP_INCLUDED
#define LBANN_UTILS_TENSOR_DIMS_UTILS_HPP_INCLUDED

#include "lbann/utils/exception.hpp"

#include <algorithm>
#include <utility>
#include <vector>

// This stuff shouldn't be publicly accessible.
namespace lbann {

template <typename T, typename Tag>
class NamedVector
{
  std::vector<T> m_data;

public:
  using vector_type = std::vector<T>;
  using value_type = typename vector_type::value_type;

public:
  NamedVector() = default;
  explicit NamedVector(std::vector<T> const& v) : m_data{v} {}
  explicit NamedVector(std::vector<T>&& v) : m_data{std::move(v)} {}
  template <typename U>
  explicit NamedVector(std::vector<U> const& v) : m_data(v.begin(), v.end()) {}


  NamedVector(NamedVector const& other) = default;
  NamedVector(NamedVector&& other) = default;
  NamedVector& operator=(NamedVector const& other) = default;
  NamedVector& operator=(NamedVector&& other) = default;

  template <typename U, typename UTag>
  NamedVector(NamedVector<U, UTag> const& other)
  {
    convert(other, *this);
  }

  template <typename U, typename UTag>
  NamedVector& operator=(NamedVector<U, UTag> const& other)
  {
    convert(other, *this);
    return *this;
  }

  std::vector<T>& get() noexcept { return m_data; }
  std::vector<T> const& get() const noexcept { return m_data; }

  // Convenience
  auto size() const noexcept { return m_data.size(); }
  void swap(NamedVector& other) { swap(m_data, other.m_data); }
};

template <typename IndexT>
using RowMajorDims = NamedVector<IndexT, struct RowMajorDimsTag>;

template <typename IndexT>
using ColMajorDims = NamedVector<IndexT, struct ColMajorDimsTag>;

template <typename IndexT>
using RowMajorStrides = NamedVector<IndexT, struct RowMajorStridesTag>;

template <typename IndexT>
using ColMajorStrides = NamedVector<IndexT, struct ColMajorStridesTag>;

using RowMajorPerm = NamedVector<int, struct RowMajorPermTag>;

using ColMajorPerm = NamedVector<int, struct ColMajorPermTag>;

// Conversion functions RowMajor <-> ColMajor

// For dimensions, this is a simple reversal.
template <typename IndexT>
void convert(RowMajorDims<IndexT> const& src, ColMajorDims<IndexT>& tgt)
{
  tgt.get().assign(crbegin(src.get()), crend(src.get()));
}

template <typename IndexT>
void convert(ColMajorDims<IndexT> const& src, RowMajorDims<IndexT>& tgt)
{
  tgt.get().assign(crbegin(src.get()), crend(src.get()));
}

// For dimensions, this is a simple reversal.
template <typename IndexT>
void convert(RowMajorStrides<IndexT> const& src, ColMajorStrides<IndexT>& tgt)
{
  tgt.get().assign(crbegin(src.get()), crend(src.get()));
}

template <typename IndexT>
void convert(ColMajorStrides<IndexT> const& src, RowMajorStrides<IndexT>& tgt)
{
  tgt.get().assign(crbegin(src.get()), crend(src.get()));
}

// For permutation arrays, it's a reversal with a complement with
// respect to the total number of dimensions.
inline void switch_perm_majorness(std::vector<int> const& in,
                                  std::vector<int>& out)
{
  int const ndims = static_cast<int>(in.size());
  out.resize(ndims);
  std::transform(crbegin(in), crend(in), begin(out), [ndims](int const& a) {
    return ndims - a - 1;
  });
}

inline void convert(RowMajorPerm const& src, ColMajorPerm& tgt)
{
  switch_perm_majorness(src.get(), tgt.get());
}

inline void convert(ColMajorPerm const& src, RowMajorPerm& tgt)
{
  switch_perm_majorness(src.get(), tgt.get());
}

/** @brief Copy the input vector to a new type.
 *
 *  The types must implicitly convert.
 */
template <typename OutT, typename InT>
auto vec_convert(std::vector<InT> const& in)
{
  return std::vector<OutT>{cbegin(in), cend(in)};
}

/** @name Factory functions with type deduction */
///@{

template <typename IndexT>
auto RowMajor(std::vector<IndexT>&& ds)
{
  return RowMajorDims<IndexT>{std::move(ds)};
}

template <typename IndexT>
auto RowMajor(std::vector<IndexT> const& ds)
{
  return RowMajorDims<IndexT>{ds};
}

template <typename IndexT>
auto RowMajor(ColMajorDims<IndexT> const& dims)
{
  return RowMajorDims<IndexT>(dims);
}

template <typename IndexT>
auto RowMajor(RowMajorDims<IndexT> const& dims)
{
  return RowMajorDims<IndexT>(dims);
}

template <typename IndexT>
auto RowMajor(RowMajorDims<IndexT>&& dims)
{
  return RowMajorDims<IndexT>(std::move(dims));
}

template <typename IndexT>
auto ColMajor(std::vector<IndexT>&& ds)
{
  return ColMajorDims<IndexT>{std::move(ds)};
}

template <typename IndexT>
auto ColMajor(std::vector<IndexT> const& ds)
{
  return ColMajorDims<IndexT>{ds};
}

template <typename IndexT>
auto ColMajor(RowMajorDims<IndexT> const& dims)
{
  return ColMajorDims<IndexT>(dims);
}

template <typename IndexT>
auto ColMajor(ColMajorDims<IndexT> const& dims)
{
  return ColMajorDims<IndexT>(dims);
}

template <typename IndexT>
auto ColMajor(ColMajorDims<IndexT>&& dims)
{
  return ColMajorDims<IndexT>(std::move(dims));
}

///@}
/** @name Computing (packed) stride information. */
///@{

/** @brief Compute packed strides of the given dimensions.
 *
 *  This assumes that the tensor in question is packed with its
 *  dimensions represented in a column-major ordering. This allows for
 *  type-converting the accumulating stride.
 */
template <typename StrideT, typename DimT>
auto get_strides_as(ColMajorDims<DimT> const& dims)
{
  std::vector<DimT> const& dim_vec = dims.get();
  size_t const ndims = dim_vec.size();

  std::vector<StrideT> strides;
  strides.reserve(ndims);
  strides.push_back(StrideT{1});
  for (size_t ii = 0UL; ii < ndims - 1; ++ii)
    strides.push_back(strides[ii] * static_cast<StrideT>(dim_vec[ii]));
  return ColMajorStrides<StrideT>(strides);
}

/** @brief Compute packed strides of the given dimensions.
 *
 *  This assumes that the tensor in question is packed with its
 *  dimensions represented in a column-major ordering. The strides are
 *  represented in the same type as the input dimensions.
 */
template <typename DimT>
auto get_strides(ColMajorDims<DimT> const& dims)
{
  return get_strides_as<DimT>(dims);
}

// TODO: RowMajorDims impl for get_strides -- not needed for now.

///@}
/** @name Permutation helper functions */
///@{

/** @brief Checks that the permutation is valid.
 *
 *  A valid permutation uses every index in [0, ndims) exactly once.
 */
template <typename T>
bool check_perm_impl(std::vector<T> perm)
{
  size_t const ndims = perm.size();
  std::sort(begin(perm), end(perm));
  for (size_t ii = 0; ii < ndims; ++ii)
    if (static_cast<size_t>(perm[ii]) != ii)
      return false;
  return true;
}

/** @brief Returns the inverse of the given permutation. */
template <typename T>
auto invert_perm_impl(std::vector<T> const& perm)
{
  size_t const size = perm.size();
  std::vector<T> out(size);
  for (size_t ii = 0; ii < size; ++ii)
    out[perm[ii]] = ii;
  return out;
}

// This does the raw vector permute. The assumption is that "in" and
// "perm" are with respect to the same ordering (either row-major or
// column-major). output[i] = input[perm[i]]. This is enforced via the
// public interfaces below. If perm.size() < in.size(), the remaining
// elements in "in" are copied without permutation (e.g.,
// permute_impl({1,2,3,4}, {1,0}) -> {2,1,3,4}). This is useful for
// the "modes", which might include the non-permutable sample
// dimension.
template <typename IndexT, typename PermT>
auto permute_impl(std::vector<IndexT> const& in, std::vector<PermT> const& perm)
{
  if (perm.size() == 0UL)
    return in;

  size_t const ndims = in.size();
  size_t const nperm = perm.size();

  LBANN_ASSERT_DEBUG(nperm <= ndims);
  std::vector<IndexT> out;
  out.reserve(ndims);
  for (size_t ii = 0UL; ii < nperm; ++ii)
    out.push_back(in[perm[ii]]);
  for (size_t ii = nperm; ii < ndims; ++ii)
    out.push_back(in[ii]);
  return out;
}

///@}
/** @name Public interface for permutation arrays */
///@{

inline bool is_valid(RowMajorPerm const& perm)
{
  return perm.size() == 0UL || check_perm_impl(perm.get());
}

inline bool is_valid(ColMajorPerm const& perm)
{
  return perm.size() == 0UL || check_perm_impl(perm.get());
}

inline RowMajorPerm invert(RowMajorPerm const& in)
{
  return RowMajorPerm{invert_perm_impl(in.get())};
}

inline ColMajorPerm invert(ColMajorPerm const& in)
{
  return ColMajorPerm{invert_perm_impl(in.get())};
}

///@}
/** @name Permuting dimensions */
///@{

template <typename IndexT>
auto permute_dims(RowMajorDims<IndexT> const& in, RowMajorPerm const& perm)
{
  return RowMajor(permute_impl(in.get(), perm.get()));
}

template <typename IndexT>
auto permute_dims(ColMajorDims<IndexT> const& in, ColMajorPerm const& perm)
{
  return ColMajor(permute_impl(in.get(), perm.get()));
}

///@}

} // namespace lbann
#endif // LBANN_UTILS_TENSOR_DIMS_UTILS_HPP_INCLUDED
