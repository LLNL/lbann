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

#define LBANN_ENTRYWISE_SCALE_BIAS_LAYER_INSTANTIATE
#include "lbann/layers/learning/entrywise_scale_bias.hpp"
#include "lbann/optimizers/optimizer_impl.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
void fp_impl(
  const El::Matrix<TensorDataType, El::Device::CPU>& local_input,
  El::Matrix<TensorDataType, El::Device::CPU>& local_output,
  El::Matrix<TensorDataType, El::Device::CPU> const& local_scale_bias)
{

  // Local matrices
  const auto local_scale = El::LockedView(local_scale_bias, El::ALL, El::IR(0));
  const auto local_bias = El::LockedView(local_scale_bias, El::ALL, El::IR(1));

  // Apply entry-wise scale and bias
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& a = local_scale(row, 0);
      const auto& b = local_bias(row, 0);
      const auto& x = local_input(row, col);
      auto& y = local_output(row, col);
      y = a * x + b;
    }
  }
}

template <typename TensorDataType>
void bp_impl(
  const El::Matrix<TensorDataType, El::Device::CPU>& local_input,
  const El::Matrix<TensorDataType, El::Device::CPU>& local_gradient_wrt_output,
  El::Matrix<TensorDataType, El::Device::CPU>& local_gradient_wrt_input,
  El::Matrix<TensorDataType, El::Device::CPU> const& local_scale_bias,
  El::AbstractDistMatrix<TensorDataType>& gradient_wrt_scale_bias)
{

  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local matrices
  auto& local_gradient_wrt_scale_bias =
    dynamic_cast<CPUMatType&>(gradient_wrt_scale_bias.Matrix());
  const auto local_scale = El::LockedView(local_scale_bias, El::ALL, El::IR(0));
  auto local_gradient_wrt_scale =
    El::View(local_gradient_wrt_scale_bias, El::ALL, El::IR(0));
  auto local_gradient_wrt_bias =
    El::View(local_gradient_wrt_scale_bias, El::ALL, El::IR(1));

  // Dimensions
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();

  // Iterate through row blocks
  // Note: Block size is chosen to match cache line size.
  El::Zero(local_gradient_wrt_scale_bias);
  constexpr El::Int _bsize = 64 / sizeof(DataType);
  constexpr El::Int bsize = _bsize > 1 ? _bsize : 1;
  LBANN_OMP_PARALLEL_FOR
  for (El::Int row_start = 0; row_start < local_height; row_start += bsize) {
    const El::Int row_end = std::min(row_start + bsize, local_height);
    const El::Int col_start = 0;
    const El::Int col_end = local_width;

    // Compute gradient contributions for row block
    for (El::Int col = col_start; col < col_end; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& a = local_scale(row, 0);
        const auto& x = local_input(row, col);
        const auto& dy = local_gradient_wrt_output(row, col);
        auto& dx = local_gradient_wrt_input(row, col);
        auto& da = local_gradient_wrt_scale(row, 0);
        auto& db = local_gradient_wrt_bias(row, 0);
        dx = a * dy;
        da += x * dy;
        db += dy;
      }
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void entrywise_scale_bias_layer<TensorDataType, Layout, Device>::fp_compute()
{
  using LocalMatType = El::Matrix<TensorDataType, Device>;
  fp_impl(
    dynamic_cast<LocalMatType const&>(this->get_local_prev_activations()),
    dynamic_cast<LocalMatType&>(this->get_local_activations()),
    dynamic_cast<LocalMatType const&>(this->weights_values(0).LockedMatrix()));
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void entrywise_scale_bias_layer<TensorDataType, Layout, Device>::bp_compute()
{
  using LocalMatType = El::Matrix<TensorDataType, Device>;

  bp_impl(
    dynamic_cast<LocalMatType const&>(this->get_local_prev_activations()),
    dynamic_cast<LocalMatType const&>(this->get_local_prev_error_signals()),
    dynamic_cast<LocalMatType&>(this->get_local_error_signals()),
    dynamic_cast<LocalMatType const&>(this->weights_values(0).LockedMatrix()),
    *this->m_weights_gradient);

  // Update optimizer with gradient
  auto* opt = this->get_weights(0).get_optimizer();
  if (opt != nullptr) {
    opt->add_to_gradient(*(this->m_weights_gradient),
                         El::TypeTraits<TensorDataType>::One(),
                         true);
  }
}

LBANN_LAYER_DEFAULT_BUILDER(entrywise_scale_bias)

#define PROTO(T)                                                               \
  template class entrywise_scale_bias_layer<T,                                 \
                                            data_layout::DATA_PARALLEL,        \
                                            El::Device::CPU>;                  \
  template class entrywise_scale_bias_layer<T,                                 \
                                            data_layout::MODEL_PARALLEL,       \
                                            El::Device::CPU>;                  \
  LBANN_LAYER_BUILDER_ETI(entrywise_scale_bias, T, El::Device::CPU)

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
