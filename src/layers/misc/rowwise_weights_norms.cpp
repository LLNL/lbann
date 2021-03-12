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

#define LBANN_ROWWISE_WEIGHTS_NORMS_LAYER_INSTANTIATE
#include "lbann/layers/misc/rowwise_weights_norms.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::row_sqsums(
  const El::Matrix<TensorDataType, Device>& mat,
  El::Matrix<TensorDataType, Device>& row_sqsums) {

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
        const auto& x = mat_buf[row+col*mat_ldim];
        auto& y = row_sqsums_buf[row];
        y += x*x;
      }
    }
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::sqrt(
  El::Matrix<TensorDataType, Device>& mat) {
  // auto func = [](const TensorDataType& x) -> TensorDataType {
  //   return El::Sqrt(x);
  // };
  // El::EntrywiseMap(mat, El::MakeFunction(func));
  El::EntrywiseMap(mat, El::Sqrt<TensorDataType>);
}

#define PROTO(T)                                            \
  template class rowwise_weights_norms_layer<               \
    T, data_layout::DATA_PARALLEL, El::Device::CPU>;        \
  template class rowwise_weights_norms_layer<               \
    T, data_layout::MODEL_PARALLEL, El::Device::CPU>
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
