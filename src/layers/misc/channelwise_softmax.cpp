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

#include <memory>
#include <type_traits>
#define LBANN_CHANNELWISE_SOFTMAX_LAYER_INSTANTIATE
#include "lbann/layers/misc/channelwise_softmax_impl.hpp"
#include "lbann/utils/memory.hpp"

namespace lbann {

// =========================================================
// Forward prop
// =========================================================

namespace {

template <typename TensorDataType>
void fp_impl(El::Int num_channels,
             El::Int channel_size,
             El::Int channel_stride,
             const El::AbstractDistMatrix<TensorDataType>& input,
             El::AbstractDistMatrix<TensorDataType>& output)
{

  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<LocalMat&>(output.Matrix());

  // Dimensions
  const El::Int local_mini_batch_size = local_input.Width();

  // Compute softmax shifts
  //   shift = max(x_i)
  LocalMat local_shifts(num_channels, local_mini_batch_size);
  El::Fill(local_shifts, std::numeric_limits<TensorDataType>::lowest());
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int k = 0; k < local_mini_batch_size; ++k) {
    for (El::Int j = 0; j < num_channels; ++j) {
      auto& maxval = local_shifts(j, k);
      for (El::Int i = 0; i < channel_size; ++i) {
        maxval = std::max(maxval, local_input(i + j * channel_stride, k));
      }
    }
  }

  // Compute softmax denominators
  //   denom = sum( exp(x_i-shift) )
  LocalMat local_denoms(num_channels, local_mini_batch_size);
  El::Zero(local_denoms);
  for (El::Int k = 0; k < local_mini_batch_size; ++k) {
    for (El::Int j = 0; j < num_channels; ++j) {
      const auto& shift = local_shifts(j, k);
      auto& denom = local_denoms(j, k);
      for (El::Int i = 0; i < channel_size; ++i) {
        const auto& x = local_input(i + j * channel_stride, k);
        denom += std::exp(x - shift);
      }
    }
  }

  // Compute softmax
  //   y_i = exp(x_i-shift) / denom
  for (El::Int k = 0; k < local_mini_batch_size; ++k) {
    for (El::Int j = 0; j < num_channels; ++j) {
      const auto& shift = local_shifts(j, k);
      const auto& denom = local_denoms(j, k);
      for (El::Int i = 0; i < channel_size; ++i) {
        const auto& x = local_input(i + j * channel_stride, k);
        auto& y = local_output(i + j * channel_stride, k);
        y = std::exp(x - shift) / denom;
      }
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_softmax_layer<TensorDataType, Layout, Device>::fp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("channelwise_softmax_layer::fp_compute");
  El::Int num_channels, channel_size, channel_stride;
  this->get_channel_size_and_stride(channel_size, channel_stride, num_channels);
  fp_impl(num_channels,
          channel_size,
          channel_stride,
          this->get_prev_activations(),
          this->get_activations());
}

// =========================================================
// Backprop
// =========================================================

namespace {

template <typename TensorDataType>
void bp_impl(El::Int num_channels,
             El::Int channel_size,
             El::Int channel_stride,
             const El::AbstractDistMatrix<TensorDataType>& output,
             const El::AbstractDistMatrix<TensorDataType>& output_grad,
             El::AbstractDistMatrix<TensorDataType>& input_grad)
{

  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& local_output =
    dynamic_cast<const LocalMat&>(output.LockedMatrix());
  const auto& local_output_grad =
    dynamic_cast<const LocalMat&>(output_grad.LockedMatrix());
  auto& local_input_grad = dynamic_cast<LocalMat&>(input_grad.Matrix());

  // Dimensions
  const El::Int local_mini_batch_size = local_output.Width();

  // dot(y,dL/dy)
  LocalMat local_y_dot_dy(num_channels, local_mini_batch_size);
  El::Zero(local_y_dot_dy);
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int k = 0; k < local_mini_batch_size; ++k) {
    for (El::Int j = 0; j < num_channels; ++j) {
      auto& y_dot_dy = local_y_dot_dy(j, k);
      for (El::Int i = 0; i < channel_size; ++i) {
        const auto& y = local_output(i + j * channel_stride, k);
        const auto& dy = local_output_grad(i + j * channel_stride, k);
        y_dot_dy += y * dy;
      }
    }
  }

  // dL/dx_i = y_i * ( dL/dy_i - dot(y,dL/dy) )
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int k = 0; k < local_mini_batch_size; ++k) {
    for (El::Int j = 0; j < num_channels; ++j) {
      const auto& y_dot_dy = local_y_dot_dy(j, k);
      for (El::Int i = 0; i < channel_size; ++i) {
        const auto& y = local_output(i + j * channel_stride, k);
        const auto& dy = local_output_grad(i + j * channel_stride, k);
        auto& dx = local_input_grad(i + j * channel_stride, k);
        dx = y * (dy - y_dot_dy);
      }
    }
  }
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_softmax_layer<TensorDataType, Layout, Device>::bp_compute()
{
  LBANN_CALIPER_MARK_SCOPE("channelwise_softmax_layer::bp_compute");
  El::Int num_channels, channel_size, channel_stride;
  this->get_channel_size_and_stride(channel_size, channel_stride, num_channels);
  bp_impl(num_channels,
          channel_size,
          channel_stride,
          this->get_activations(),
          this->get_prev_error_signals(),
          this->get_error_signals());
}

// =========================================================
// Explicit template instantiation
// =========================================================

#define PROTO(T)                                                               \
  template class channelwise_softmax_layer<T,                                  \
                                           data_layout::DATA_PARALLEL,         \
                                           El::Device::CPU>
#include "lbann/macros/instantiate.hpp"
#undef PROTO

#ifdef LBANN_HAS_GPU
#define PROTO(T)                                                               \
  extern template class channelwise_softmax_layer<T,                           \
                                                  data_layout::DATA_PARALLEL,  \
                                                  El::Device::GPU>
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_HAS_GPU

} // namespace lbann
