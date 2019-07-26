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

#include "lbann/layers/learning/entrywise_scale_bias.hpp"

namespace lbann {

namespace {

void fp_impl(const CPUMat& local_input,
             CPUMat& local_output,
             const weights& scale_bias) {

  // Local matrices
  const auto& local_scale_bias
    = dynamic_cast<const CPUMat&>(scale_bias.get_values().LockedMatrix());
  const auto local_scale = El::LockedView(local_scale_bias,
                                          El::ALL, El::IR(0));
  const auto local_bias = El::LockedView(local_scale_bias,
                                         El::ALL, El::IR(1));

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

void bp_impl(const CPUMat& local_input,
             const CPUMat& local_gradient_wrt_output,
             CPUMat& local_gradient_wrt_input,
             weights& scale_bias,
             AbsDistMat& gradient_wrt_scale_bias,
             El::Int mini_batch_size) {

  // Local matrices
  const auto& local_scale_bias
    = dynamic_cast<const CPUMat&>(scale_bias.get_values().LockedMatrix());
  auto& local_gradient_wrt_scale_bias
    = dynamic_cast<CPUMat&>(gradient_wrt_scale_bias.Matrix());
  const auto local_scale = El::LockedView(local_scale_bias,
                                          El::ALL, El::IR(0));
  auto local_gradient_wrt_scale = El::View(local_gradient_wrt_scale_bias,
                                           El::ALL, El::IR(0));
  auto local_gradient_wrt_bias = El::View(local_gradient_wrt_scale_bias,
                                          El::ALL, El::IR(1));

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

  // Update optimizer with gradient
  auto* opt = scale_bias.get_optimizer();
  if (opt != nullptr) {
    opt->add_to_gradient(gradient_wrt_scale_bias,
                         DataType{1} / mini_batch_size,
                         true);
  }

}

} // namespace

// Template instantiation
template <>
void entrywise_scale_bias_layer<data_layout::DATA_PARALLEL,El::Device::CPU>
     ::fp_compute() {
  fp_impl(get_local_prev_activations(),
          get_local_activations(),
          *m_weights[0]);
}
template <>
void entrywise_scale_bias_layer<data_layout::MODEL_PARALLEL,El::Device::CPU>
     ::fp_compute() {
  fp_impl(get_local_prev_activations(),
          get_local_activations(),
          *m_weights[0]);
}
template <>
void entrywise_scale_bias_layer<data_layout::DATA_PARALLEL,El::Device::CPU>
     ::bp_compute() {
  bp_impl(get_local_prev_activations(),
          get_local_prev_error_signals(),
          get_local_error_signals(),
          *this->m_weights[0],
          *m_weights_gradient,
          this->m_model->get_effective_mini_batch_size());
}
template <>
void entrywise_scale_bias_layer<data_layout::MODEL_PARALLEL,El::Device::CPU>
     ::bp_compute() {
  bp_impl(get_local_prev_activations(),
          get_local_prev_error_signals(),
          get_local_error_signals(),
          *this->m_weights[0],
          *m_weights_gradient,
          this->m_model->get_effective_mini_batch_size());
}

} // namespace lbann
