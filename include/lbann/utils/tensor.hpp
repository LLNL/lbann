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

#ifndef LBANN_UTILS_TENSOR_HPP
#define LBANN_UTILS_TENSOR_HPP

#include "lbann/base.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/typename.hpp"

#include <El/core/DistMatrix/AbstractDistMatrix.hpp>
#include <iterator>
#include <type_traits>

namespace lbann {

/// @brief Function to efficiently select the best method for copying between
/// two distributed tensors. Enable selection between synchronous and
/// asynchronous copies based on tensor distribution and
/// pre-processing macros
template <typename TDT>
void do_tensor_copy(const BaseDistMat& src, El::AbstractDistMatrix<TDT>& tgt);

/// @brief If distributed tensors have the same distribution setup the
/// target to use a view to the source tensor, otherwise copy the src
/// to target.
template <typename TDT>
void view_or_copy_tensor(const BaseDistMat& src,
                         El::AbstractDistMatrix<TDT>& tgt,
                         bool locked_view = true);

namespace utils {
namespace details {

/** @brief Interpret the matrix as a tensor and return the tensor-ized
 *         dimensions.
 *  @param[in] A The matrix.
 *  @returns The dimensions of the matrix interpreted as a tensor.
 */
template <typename MatrixT>
std::vector<size_t> get_tensor_dims(MatrixT const& A)
{
  return {El::To<size_t>(A.Width()), El::To<size_t>(A.Height())};
}

/** @brief Attempt to compute the tensor dimensions of the local
 *         portion of the matrix, given the global tensor dimensions.
 *
 *  This is only valid in two cases. Either the matrix must be
 *  column-distributed or it must logically represent a collection of
 *  1D arrays.
 */
template <typename T>
std::vector<size_t> localize_dims(El::AbstractDistMatrix<T> const& A,
                                  std::vector<size_t> const& global_dims)
{
  if (A.ColDist() == El::Dist::STAR) {
    std::vector<size_t> out = global_dims;
    out.front() = A.LocalWidth();
    return out;
  }
  else if (global_dims.size() == 2UL) {
    return get_tensor_dims(A.LockedMatrix());
  }
  else {
    std::ostringstream oss;
    oss << "{";
    for (auto const& d : global_dims)
      oss << " " << d;
    oss << " }";
    LBANN_WARNING("Dimension localization is ill-posed. Dims=", oss.str());

    return global_dims;
  }
}

template <typename MatrixOutT, typename MatrixInT>
struct SafeMatrixCaster
{
  static MatrixOutT& cast(MatrixInT& in) { return in; }
  static MatrixOutT const& cast(MatrixInT const& in) { return in; }
};

template <typename OutDataType, El::Device D, typename InDataType>
struct SafeMatrixCaster<El::Matrix<OutDataType, D>,
                        El::AbstractMatrix<InDataType>>
{
  using OutType = El::Matrix<OutDataType, D>;
  using InType = El::AbstractMatrix<InDataType>;
  static OutType& cast(InType& in)
  {
    LBANN_ASSERT(in.GetDevice() == D);
    return static_cast<OutType&>(in);
  }
  static OutType const& cast(InType const& in)
  {
    LBANN_ASSERT(in.GetDevice() == D);
    return static_cast<OutType const&>(in);
  }
};

template <typename MatrixOutT, typename MatrixInT>
MatrixOutT& SafeMatrixCast(MatrixInT& in)
{
  using OutType = std::decay_t<MatrixOutT>;
  using InType = std::decay_t<MatrixInT>;
  return SafeMatrixCaster<OutType, InType>::cast(in);
}

/** @brief Manage a reference to a (possibly const) matrix */
template <typename MatrixT>
class MatrixReferenceWrapper
{
public:
  using matrix_type = MatrixT;

public:
  template <typename MatT>
  MatrixReferenceWrapper(MatT&& x)
    : m_data{SafeMatrixCast<matrix_type&>(std::forward<MatT>(x))}
  {}
  operator matrix_type&() const noexcept { return m_data; }
  matrix_type& data() const noexcept { return m_data; }

private:
  std::reference_wrapper<matrix_type> m_data;
};

/** @brief Interpret a matrix as a tensor. */
template <typename MatrixT>
class MatrixAsTensorView : public MatrixReferenceWrapper<MatrixT>
{
public:
  template <typename MatT>
  MatrixAsTensorView(MatT&& mat, std::vector<size_t> const& dims)
    : MatrixReferenceWrapper<MatrixT>{std::forward<MatT>(mat)}, m_dims{dims}
  {}

  std::vector<size_t> const& dims() const noexcept { return m_dims; }
  size_t rank() const noexcept { return m_dims.size(); };

private:
  std::vector<size_t> m_dims;
}; // MatrixAsTensorView

/** @brief Copy between two tensors on different process grids */
template <typename TDT>
void do_tensor_copy_between_grids(const BaseDistMat& src,
                                  El::AbstractDistMatrix<TDT>& tgt);

/** @brief Copy between two tensors on different process grids */
template <typename TDT,
          El::Dist ColDist,
          El::Dist RowDist,
          El::DistWrap Wrap,
          El::Device Device>
void do_tensor_copy_between_grids(
  const BaseDistMat& src,
  El::DistMatrix<TDT, ColDist, RowDist, Wrap, Device>& tgt);

} // namespace details

template <typename T, El::Device D>
class TensorView : public details::MatrixAsTensorView<El::Matrix<T, D>>
{
  using base_type = details::MatrixAsTensorView<El::Matrix<T, D>>;

public:
  template <typename MatT>
  TensorView(MatT&& mat)
    : TensorView{std::forward<MatT>(mat), details::get_tensor_dims(mat)}
  {}
  template <typename MatT>
  TensorView(MatT&& mat, std::vector<size_t> const& dims)
    : base_type{std::forward<MatT>(mat), dims}
  {}
};

template <typename T, El::Device D>
class ConstTensorView
  : public details::MatrixAsTensorView<El::Matrix<T, D> const>
{
  using base_type = details::MatrixAsTensorView<El::Matrix<T, D> const>;

public:
  template <typename MatT>
  ConstTensorView(MatT&& mat)
    : ConstTensorView{std::forward<MatT>(mat), details::get_tensor_dims(mat)}
  {}
  template <typename MatT>
  ConstTensorView(MatT&& mat, std::vector<size_t> const& dims)
    : base_type{std::forward<MatT>(mat), dims}
  {}
};

// There's a clever laziness here -- the dist matrix will never be
// directly checked for device allocation sanity. However, when the
// local tensor view is constructed around the local matrix, _that_
// matrix will be checked for device allocation sanity, thereby
// guaranteeing the sanity of the whole object.
//
// FIXME (trb 06/25/21): If we wanted to be VERY rigorous, we'd adjust
// the local tensor dimension vector in a distribution-specific
// way. But we shouldn't be doing tensor-y things with MC,MR matrices,
// just matrix-y things (i.e., we can expect that the samples are 1-D
// when we encounter MC,MR matrices).

template <typename T, El::Device D>
class DistTensorView
  : public details::MatrixAsTensorView<El::AbstractDistMatrix<T>>
{
  using base_type = details::MatrixAsTensorView<El::AbstractDistMatrix<T>>;

public:
  template <typename MatT>
  DistTensorView(MatT&& mat)
    : DistTensorView{std::forward<MatT>(mat), details::get_tensor_dims(mat)}
  {}
  template <typename MatT>
  DistTensorView(MatT&& mat, std::vector<size_t> const& dims)
    : base_type{std::forward<MatT>(mat), dims},
      m_local_data{mat.Matrix(), details::localize_dims(mat, this->dims())}
  {}

  /** @brief Access the local tensor data. */
  TensorView<T, D> const& local_data() const noexcept { return m_local_data; }

private:
  TensorView<T, D> m_local_data;

}; // DistTensorView<T,D>

template <typename T, El::Device D>
class ConstDistTensorView
  : public details::MatrixAsTensorView<El::AbstractDistMatrix<T> const>
{
  using base_type =
    details::MatrixAsTensorView<El::AbstractDistMatrix<T> const>;

public:
  template <typename MatT>
  ConstDistTensorView(MatT&& mat)
    : ConstDistTensorView{std::forward<MatT>(mat),
                          details::get_tensor_dims(mat)}
  {}
  template <typename MatT>
  ConstDistTensorView(MatT&& mat, std::vector<size_t> const& dims)
    : base_type{std::forward<MatT>(mat), dims},
      m_local_data{mat.LockedMatrix(),
                   details::localize_dims(mat, this->dims())}
  {}

  ConstTensorView<T, D> const& local_data() const noexcept
  {
    return m_local_data;
  }

private:
  ConstTensorView<T, D> m_local_data;

}; // ConstDistTensorView

} // namespace utils

} // namespace lbann
#endif // LBANN_UTILS_TENSOR_HPP
