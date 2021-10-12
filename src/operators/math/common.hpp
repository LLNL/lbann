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
#ifndef LBANN_SRC_OPERATORS_MATH_COMMON_HPP_INCLUDED
#define LBANN_SRC_OPERATORS_MATH_COMMON_HPP_INCLUDED

#include "lbann/base.hpp"

namespace lbann {
namespace internal {

/** @brief A binary entrywise map c <- f(a,b).
 */
template <typename S, typename T, typename U, typename F>
void EntrywiseZipInto(El::Matrix<S, El::Device::CPU> const& A,
                      El::Matrix<T, El::Device::CPU> const& B,
                      El::Matrix<U, El::Device::CPU>& C,
                      F func)
{
  EL_DEBUG_CSE;
  auto const m = A.Height();
  auto const n = A.Width();

  LBANN_ASSERT_DEBUG(B.Height() == m);
  LBANN_ASSERT_DEBUG(B.Width() == n);

  LBANN_ASSERT_DEBUG(C.Height() == m);
  LBANN_ASSERT_DEBUG(C.Width() == n);

  S const* ABuf = A.LockedBuffer();
  T const* BBuf = B.LockedBuffer();
  U* CBuf = C.Buffer();

  auto const ALDim = A.LDim();
  auto const BLDim = B.LDim();
  auto const CLDim = C.LDim();

  // Use entry-wise parallelization for column vectors. Otherwise
  // use column-wise parallelization.
  if (n == 1) {
    EL_PARALLEL_FOR
    for (El::Int i = 0; i < m; ++i) {
      CBuf[i] = func(ABuf[i], BBuf[i]);
    }
  }
  else {
    EL_PARALLEL_FOR_COLLAPSE2
    for (El::Int j = 0; j < n; ++j) {
      for (El::Int i = 0; i < m; ++i) {
        CBuf[i + j * CLDim] = func(ABuf[i + j * ALDim], BBuf[i + j * BLDim]);
      }
    }
  }
}

/** Apply a binary backprop operator to CPU data.
 *  The input and output data must be on CPU and must have the same
 *  dimensions. Given a binary function \f$ y = f(x_1,x_2) \f$, the
 *  corresponding BinaryBackPropOperator is a 5-ary function with the
 *  arguments \f$ x_1 \f$, \f$ x_2 \f$, \f$ dL/dy \f$, \f$ dL/dx_1\f$,
 *  \f$ dL/dx_2 \f$. The last two arguments should be overwritten when
 *  the BinaryBackPropOperator is called.
 */
template <typename DataT, typename F>
void apply_binary_backprop_operator(
  El::Matrix<DataT, El::Device::CPU> const& x1,
  El::Matrix<DataT, El::Device::CPU> const& x2,
  El::Matrix<DataT, El::Device::CPU> const& dy,
  El::Matrix<DataT, El::Device::CPU>& dx1,
  El::Matrix<DataT, El::Device::CPU>& dx2,
  F f)
{
  LBANN_CALIPER_MARK_FUNCTION;
  if (x1.Contiguous() && x2.Contiguous() && dy.Contiguous() &&
      dx1.Contiguous() && dx2.Contiguous()) {
    const auto* x1_buffer = x1.LockedBuffer();
    const auto* x2_buffer = x2.LockedBuffer();
    const auto* dy_buffer = dy.LockedBuffer();
    auto* dx1_buffer = dx1.Buffer();
    auto* dx2_buffer = dx2.Buffer();
    const size_t size = x1.Height() * x1.Width();
    LBANN_OMP_PARALLEL_FOR
    for (size_t i = 0; i < size; ++i) {
      f(x1_buffer[i], x2_buffer[i], dy_buffer[i], dx1_buffer[i], dx2_buffer[i]);
    }
  }
  else {
    auto const width = x1.Width();
    auto const height = x1.Height();
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (El::Int jj = 0; jj < width; ++jj) {
      for (El::Int ii = 0; ii < height; ++ii) {
        f(x1(ii, jj), x2(ii, jj), dy(ii, jj), dx1(ii, jj), dx2(ii, jj));
      }
    }
  }
}

} // namespace internal
} // namespace lbann
#endif // LBANN_SRC_OPERATORS_MATH_COMMON_HPP_INCLUDED
