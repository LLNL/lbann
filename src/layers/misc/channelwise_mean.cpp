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

#include "lbann/layers/misc/channelwise_mean.hpp"

namespace lbann {

template <>
void channelwise_mean_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::fp_compute() {

  // Local matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();

  // Dimensions
  // Note: channel_size is the number of input entries per channel and
  // local_width is the number of local mini-batch samples.
  const auto& input_dims = get_input_dims();
  const El::Int num_channels = input_dims[0];
  const El::Int channel_size = std::accumulate(input_dims.begin() + 1,
                                               input_dims.end(),
                                               1, std::multiplies<int>());
  const auto& local_width = local_input.Width();

  // Compute channel-wise mean
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      DataType sum = 0;
      for (El::Int i = 0; i < channel_size; ++i) {
        sum += local_input(i + channel * channel_size, col);
      }
      local_output(channel, col) = sum / channel_size;
    }
  }

}

template <>
void channelwise_mean_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::bp_compute() {

  // Local matrices
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();

  // Dimensions
  // Note: channel_size is the number of input entries per channel and
  // local_width is the number of local mini-batch samples.
  const auto& input_dims = get_input_dims();
  const El::Int num_channels = input_dims[0];
  const El::Int channel_size = std::accumulate(input_dims.begin() + 1,
                                               input_dims.end(),
                                               1, std::multiplies<int>());
  const auto& local_width = local_gradient_wrt_input.Width();

  // Compute gradients
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      const auto& dy = local_gradient_wrt_output(channel, col);
      const auto& dx = dy / channel_size;
      for (El::Int i = 0; i < channel_size; ++i) {
        local_gradient_wrt_input(i + channel * channel_size, col) = dx;
      }
    }
  }

}

} // namespace lbann
