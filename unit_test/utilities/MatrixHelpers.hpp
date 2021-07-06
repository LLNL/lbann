////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_UNIT_TEST_UTILITIES_MATRIX_HELPERS_HPP_INCLUDED
#define LBANN_UNIT_TEST_UTILITIES_MATRIX_HELPERS_HPP_INCLUDED

#include <lbann/base.hpp>

namespace El {

template <typename T>
bool compare_values(Matrix<T, Device::CPU> const& A,
                    Matrix<T, Device::CPU> const& B)
#ifdef HYDROGEN_RELEASE_BUILD
  noexcept
#endif
{
  using IdxT = decltype(A.Height());

  // Short-circuit for compare-to-self.
  if (&A == &B)
    return true;

  // Short-circuit for dimension mis-match
  auto const A_height = A.Height(), A_width = A.Width();
  if ((A_height != B.Height()) || (A_width != B.Width()))
    return false;

  for (IdxT col = To<IdxT>(0); col < A_width; ++col)
    for (IdxT row = To<IdxT>(0); row < A_height; ++row)
      if (A.CRef(row, col) != B.CRef(row, col))
        return false;
  return true;
}

// The "operator==" equivalence relation for non-distributed matrices
// is defined as follows.
//
// Two matrices, A and B, are considered equivalent if all of the
// following conditions are met:
//
//   1. They represent their data with the same type.
//   2. Their memory resides on the same device.
//   3. They have the same shape (LDim *NOT* included).
//   4. A_ij == B_ij for each i,j. Comparison is `operator==` for the
//      appropriate data type.

template <typename T, Device D>
bool operator==(Matrix<T, D> const& A, Matrix<T, D> const& B)
#ifdef HYDROGEN_RELEASE_BUILD
  noexcept
#endif
{
  return compare_values(A, B);
}

template <typename T, El::Device D>
bool operator==(AbstractMatrix<T> const& A, Matrix<T, D> const& B)
{
  switch (A.GetDevice()) {
  case Device::CPU:
    return static_cast<Matrix<T, Device::CPU> const&>(A) == B;
#ifdef LBANN_HAS_GPU
  case Device::GPU:
    return static_cast<Matrix<T, Device::GPU> const&>(A) == B;
#endif // LBANN_HAS_GPU
  default:
    throw "Invalid device";
  }
  return false;
}

template <typename S, typename T>
bool operator==(AbstractMatrix<S> const& A, AbstractMatrix<T> const& B) noexcept
{
  return false;
}

template <typename T>
bool operator==(AbstractMatrix<T> const& A, AbstractMatrix<T> const& B)
#ifdef HYDROGEN_RELEASE_BUILD
  noexcept
#endif
{
  switch (B.GetDevice()) {
  case Device::CPU:
    return A == static_cast<Matrix<T, Device::CPU> const&>(B);
#ifdef LBANN_HAS_GPU
  case Device::GPU:
    return A == static_cast<Matrix<T, Device::GPU> const&>(B);
#endif // LBANN_HAS_GPU
  }
  return false;
}

// Two DistMatrices are equivalent under `operator==` iff their
// distributions match and their local matrices are equivalent under
// `operator==`.

template <typename S, typename T>
bool operator==(AbstractDistMatrix<S> const& A,
                AbstractDistMatrix<T> const& B) noexcept
{
  return false;
}

template <typename T>
bool operator==(AbstractDistMatrix<T> const& A,
                AbstractDistMatrix<T> const& B) noexcept
{
  return ((A.DistData() == B.DistData()) &&
          (A.LockedMatrix() == B.LockedMatrix()));
}

} // namespace El
#endif // LBANN_UNIT_TEST_UTILITIES_MATRIX_HELPERS_HPP_INCLUDED
