////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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

#define LBANN_CHANNELWISE_SOFTMAX_LAYER_INSTANTIATE
#include "channelwise_softmax_kernels.cuh"
#include "lbann/layers/misc/channelwise_softmax_impl.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_softmax_layer<TensorDataType, Layout, Device>::fp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    this->get_distconv_adapter().fp_compute();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  // Local matrices
  const size_t num_channels = this->get_output_dims().front();
  const size_t channel_size = this->get_output_size() / num_channels;
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input =
    dynamic_cast<const LocalMat&>(this->get_prev_activations().LockedMatrix());
  auto& local_output =
    dynamic_cast<LocalMat&>(this->get_activations().Matrix());

  channelwise_softmax_fp_impl(num_channels,
                              channel_size,
                              local_input,
                              local_output);
}

// =========================================================
// Backprop
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void channelwise_softmax_layer<TensorDataType, Layout, Device>::bp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    this->get_distconv_adapter().bp_compute();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  const size_t num_channels = this->get_output_dims().front();
  const size_t channel_size = this->get_output_size() / num_channels;

  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_output =
    dynamic_cast<const LocalMat&>(this->get_activations().LockedMatrix());
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(
    this->get_prev_error_signals().LockedMatrix());
  auto& local_input_grad =
    dynamic_cast<LocalMat&>(this->get_error_signals().Matrix());

  channelwise_softmax_bp_impl(num_channels,
                              channel_size,
                              local_output,
                              local_output_grad,
                              local_input_grad);
}

// =========================================================
// Explicit template instantiation
// =========================================================

#define PROTO(T)                                                               \
  template class channelwise_softmax_layer<T,                                  \
                                           data_layout::DATA_PARALLEL,         \
                                           El::Device::GPU>;
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
