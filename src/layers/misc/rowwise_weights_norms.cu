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
  LBANN_ERROR("Not implemented"); /// @todo Implement
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::sqrt(
  El::Matrix<TensorDataType, Device>& mat) {
  LBANN_ERROR("Not implemented"); /// @todo Implement
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::divide(
  El::Matrix<TensorDataType, Device>& numer,
  const El::Matrix<TensorDataType, Device>& denom) {
  LBANN_ERROR("Not implemented"); /// @todo Implement
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType, Layout, Device>::row_axpy(
  TensorDataType alpha,
  const El::Matrix<TensorDataType, Device>& a_vec,
  const El::Matrix<TensorDataType, Device>& x_mat,
  TensorDataType beta,
  El::Matrix<TensorDataType, Device>& y_mat) {
  LBANN_ERROR("Not implemented"); /// @todo Implement
}

#define PROTO(T)                                            \
  template class rowwise_weights_norms_layer<               \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;        \
  template class rowwise_weights_norms_layer<               \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
