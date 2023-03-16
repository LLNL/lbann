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

#define LBANN_ROWWISE_WEIGHTS_NORMS_LAYER_INSTANTIATE
#include "lbann/layers/misc/rowwise_weights_norms_impl.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::row_sqsums(
  const El::Matrix<TensorDataType, Device>& mat,
  El::Matrix<TensorDataType, Device>& row_sqsums)
{

  // Matrix data
  const size_t height = mat.Height();
  const size_t width = mat.Width();
  const TensorDataType* __restrict__ mat_buf = mat.LockedBuffer();
  const size_t mat_ldim = mat.LDim();
  TensorDataType* __restrict__ row_sqsums_buf = row_sqsums.Buffer();

  // Block size for loops
  // Note: x86 cache lines are 64B
  constexpr size_t _bsize = 64 / sizeof(TensorDataType);
  constexpr size_t bsize = _bsize > 1 ? _bsize : 1;

  // Compute sums of squares for each row
  El::Zero(row_sqsums);
  LBANN_OMP_PARALLEL_FOR
  for (size_t row_start = 0; row_start < height; row_start += bsize) {
    const size_t row_end = std::min(row_start + bsize, height);
    for (size_t col = 0; col < width; ++col) {
      for (size_t row = row_start; row < row_end; ++row) {
        const auto& x = mat_buf[row + col * mat_ldim];
        auto& y = row_sqsums_buf[row];
        y += x * x;
      }
    }
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::sqrt(
  El::Matrix<TensorDataType, Device>& mat)
{
  El::EntrywiseMap(mat, {[](TensorDataType const& a) { return El::Sqrt(a); }});
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::divide(
  El::Matrix<TensorDataType, Device>& numer,
  const El::Matrix<TensorDataType, Device>& denom)
{

  // Check that matrices are valid
  if (numer.Height() != denom.Height() || numer.Width() != denom.Width()) {
    LBANN_ERROR("numerator and denominator do not have same dims");
  }
  if (!numer.Contiguous() || !denom.Contiguous()) {
    LBANN_ERROR("matrices are not contiguous");
  }

  // Divide entries and store in numerator buffer
  const size_t size = numer.Height() * numer.Width();
  TensorDataType* __restrict__ numer_buf = numer.Buffer();
  const TensorDataType* __restrict__ denom_buf = denom.LockedBuffer();
  LBANN_OMP_PARALLEL_FOR
  for (size_t i = 0; i < size; ++i) {
    auto& x = numer_buf[i];
    const auto& y = denom_buf[i];
    const auto& z = x / y;
    x = std::isfinite(z) ? z : El::TypeTraits<TensorDataType>::Zero();
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::row_axpy(
  TensorDataType alpha,
  const El::Matrix<TensorDataType, Device>& a_vec,
  const El::Matrix<TensorDataType, Device>& x_mat,
  TensorDataType beta,
  El::Matrix<TensorDataType, Device>& y_mat)
{

  // Check that matrices are valid
  if (x_mat.Height() != y_mat.Height() || x_mat.Width() != y_mat.Width() ||
      a_vec.Height() != y_mat.Height() || a_vec.Width() != 1) {
    LBANN_ERROR("matrix dims do not match");
  }

  // Matrix data
  const size_t height = y_mat.Height();
  const size_t width = y_mat.Width();
  const TensorDataType* __restrict__ a_buf = a_vec.LockedBuffer();
  const TensorDataType* __restrict__ x_buf = x_mat.LockedBuffer();
  const size_t x_ldim = x_mat.LDim();
  TensorDataType* __restrict__ y_buf = y_mat.Buffer();
  const size_t y_ldim = y_mat.LDim();

  // Compute sums of squares for each row
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (size_t col = 0; col < width; ++col) {
    for (size_t row = 0; row < height; ++row) {
      const auto& a = a_buf[row];
      const auto& x = x_buf[row + col * x_ldim];
      auto& y = y_buf[row + col * y_ldim];
      y = alpha * a * x + beta * y;
    }
  }
}

#define PROTO(T)                                                               \
  template class rowwise_weights_norms_layer<T,                                \
                                             data_layout::DATA_PARALLEL,       \
                                             El::Device::CPU>;                 \
  template class rowwise_weights_norms_layer<T,                                \
                                             data_layout::MODEL_PARALLEL,      \
                                             El::Device::CPU>
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
