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
  const auto& local_input = dynamic_cast<const CPUMat&>(get_local_prev_activations());
  auto& local_output = dynamic_cast<CPUMat&>(get_local_activations());
  const auto& local_weights = dynamic_cast<const CPUMat&>(m_weights[0]->get_values().LockedMatrix());
  const auto local_scale = El::LockedView(local_weights,
                                          El::ALL, El::IR(0));
  const auto local_bias = El::LockedView(local_weights,
                                         El::ALL, El::IR(1));

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
  LBANN_OMP_PARALLEL_FOR
  for (El::Int channel = 0; channel < num_channels; ++channel) {
    const auto a = local_scale(channel, 0);
    const auto b = local_bias(channel, 0);
    const El::Int row_start = channel * channel_size;
    const El::Int row_end = (channel + 1) * channel_size;
    const El::Int col_start = 0;
    const El::Int col_end = local_width;
    for (El::Int col = col_start; col < col_end; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& x = local_input(row, col);
        auto& y = local_output(row, col);
        y = a * x + b;
      }
    }
  }

}

template <>
void channelwise_scale_bias_layer<data_layout::DATA_PARALLEL,El::Device::CPU>
     ::bp_compute() {

  // Local matrices
  const auto& local_input = dynamic_cast<const CPUMat&>(get_local_prev_activations());
  const auto& local_gradient_wrt_output = dynamic_cast<const CPUMat&>(get_local_prev_error_signals());
  auto& local_gradient_wrt_input = dynamic_cast<CPUMat&>(get_local_error_signals());
  const auto& local_weights = dynamic_cast<const CPUMat&>(m_weights[0]->get_values().LockedMatrix());
  auto& local_gradient_wrt_weights = dynamic_cast<CPUMat&>(m_weights_gradient->Matrix());
  const auto local_scale = El::LockedView(local_weights,
                                          El::ALL, El::IR(0));
  auto local_gradient_wrt_scale = El::View(local_gradient_wrt_weights,
                                           El::ALL, El::IR(0));
  auto local_gradient_wrt_bias = El::View(local_gradient_wrt_weights,
                                          El::ALL, El::IR(1));

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
    const auto a = local_scale(channel, 0);
    auto& da = local_gradient_wrt_scale(channel, 0);
    auto& db = local_gradient_wrt_bias(channel, 0);
    da = 0;
    db = 0;
    const El::Int row_start = channel * channel_size;
    const El::Int row_end = (channel + 1) * channel_size;
    const El::Int col_start = 0;
    const El::Int col_end = local_width;
    for (El::Int col = col_start; col < col_end; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& x = local_input(row, col);
        const auto& dy = local_gradient_wrt_output(row, col);
        auto& dx = local_gradient_wrt_input(row, col);
        da += x * dy;
        db += dy;
        dx = a * dy;
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
