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

#include "lbann/layers/learning/channelwise_scale_bias.hpp"

namespace lbann {

template <>
void channelwise_scale_bias_layer<data_layout::DATA_PARALLEL,El::Device::CPU>
     ::fp_compute() {

  // Local matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  const auto& local_weights = m_weights[0]->get_values().LockedMatrix();

  // Dimensions
  // Note: channel_size is the number of input entries per channel and
  // local_width is the number of local mini-batch samples.
  const auto dims = get_output_dims();
  const El::Int num_channels = dims[0];
  const El::Int channel_size = std::accumulate(dims.begin() + 1,
                                               dims.end(),
                                               1, std::multiplies<int>());
  const El::Int local_width = local_input.Width();

  // Apply channel-wise scale and bias
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      const auto s = local_weights(channel, 0);
      const auto b = local_weights(channel, 1);
      const auto* x = local_input.LockedBuffer(channel * channel_size, col);
      auto* y = local_output.Buffer(channel * channel_size, col);
      for (El::Int i = 0; i < channel_size; ++i) {
        y[i] = s * x[i] + b;
      }
    }
  }

}

template <>
void channelwise_scale_bias_layer<data_layout::DATA_PARALLEL,El::Device::CPU>
     ::bp_compute() {

  // Local matrices
  const auto& local_input = get_local_prev_activations();
  const auto& local_weights = m_weights[0]->get_values().LockedMatrix();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  auto& local_gradient_wrt_weights = m_weights_gradient->Matrix();

  // Dimensions
  // Note: channel_size is the number of input entries per channel and
  // local_width is the number of local mini-batch samples.
  const auto dims = get_output_dims();
  const El::Int num_channels = dims[0];
  const El::Int channel_size = std::accumulate(dims.begin() + 1,
                                               dims.end(),
                                               1, std::multiplies<int>());
  const El::Int local_width = local_input.Width();

  // Compute gradients
  LBANN_OMP_PARALLEL_FOR
  for (El::Int channel = 0; channel < num_channels; ++channel) {
    const auto s = local_weights(channel, 0);
    auto& ds = local_gradient_wrt_weights(channel, 0);
    auto& db = local_gradient_wrt_weights(channel, 1);
    ds = DataType{0};
    db = DataType{0};
    for (El::Int col = 0; col < local_width; ++col) {
      const El::Int offset = channel * channel_size;
      const auto* x = local_input.LockedBuffer(offset, col);
      const auto* dy = local_gradient_wrt_output.LockedBuffer(offset, col);
      auto* dx = local_gradient_wrt_input.Buffer(offset, col);
      for (El::Int i = 0; i < channel_size; ++i) {
        ds += x[i] * dy[i];
        db += dy[i];
        dx[i] = s * dy[i];
      }
    }
  }

  // Update optimizer with gradient
  auto* opt = m_weights[0]->get_optimizer();
  if (opt != nullptr) {
    const El::Int mini_batch_size = this->m_model->get_effective_mini_batch_size();
    opt->add_to_gradient(*m_weights_gradient,
                         DataType{1} / mini_batch_size,
                         true);
  }

}

} // namespace lbann
