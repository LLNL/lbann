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
#ifndef LBANN_UTILS_DNN_LIB_OPENMP_SOFTMAX_HPP_
#define LBANN_UTILS_DNN_LIB_OPENMP_SOFTMAX_HPP_

#include "lbann/utils/dnn_enums.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "lbann/utils/dnn_lib/openmp.hpp"
#include "lbann/utils/profiling.hpp"

namespace lbann {

// This is simply the "fp_model_parallel" implementation, without the
// communication.
template <typename DataT, typename ScalarT>
void openmp_backend::softmax_forward(
  ScalarT const& alpha_in,
  TensorDescriptor const& inputDesc,
  El::Matrix<DataT, El::Device::CPU> const& local_input,
  ScalarT const& beta_in,
  TensorDescriptor const& outputDesc,
  El::Matrix<DataT, El::Device::CPU>& local_output,
  El::SyncInfo<El::Device::CPU> const& si,
  softmax_mode mode,
  softmax_alg alg)
{
  if (alg == softmax_alg::LOG)
    return logsoftmax_forward(alpha_in,
                              inputDesc,
                              local_input,
                              beta_in,
                              outputDesc,
                              local_output,
                              si,
                              mode);

  if (mode != softmax_mode::INSTANCE) {
    LBANN_ERROR("Unsupported softmax mode");
  }

  // Local matrices
  using workspace_type = hydrogen::simple_buffer<DataT, El::Device::CPU>;
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();
  workspace_type local_workspace(local_width,
                                 std::numeric_limits<DataT>::lowest(),
                                 si,
                                 /*memory_mode=*/0);
  auto* const workspace = local_workspace.data();

  // This uses a caching allocator, so it should be very low overhead
  // (at least after the zeroth iteration). We don't need pinned
  // memory, so use memory mode "0".

  // Find column-wise maximum entries
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    auto& max_entry = workspace[col];
    for (El::Int row = 0; row < local_height; ++row) {
      max_entry = std::max(max_entry, local_input(row, col));
    }
  }

  // Exponentiate outputs and compute column sums
  // Note: Subtracting by the column max prevents output from blowing
  // up. Large negative values underflow to 0.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto shift = workspace[col];
    DataT sum = El::TypeTraits<DataT>::Zero();
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& x = local_input(row, col);
      auto& y = local_output(row, col);
      y = std::exp(x - shift);
      sum += y;
    }
    workspace[col] = sum;
  }

  // Divide outputs by column sums
  // Note: Small values can be rounded to minimum output value to
  // avoid denormalized floats.
  auto const zero = El::TypeTraits<DataT>::Zero();
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& scale = 1 / workspace[col];
    for (El::Int row = 0; row < local_height; ++row) {
      auto& y = local_output(row, col);
      auto const tmp = scale * y;
      y = std::isnormal(tmp) ? tmp : zero;
    }
  }
}

template <typename DataT, typename ScalarT>
void openmp_backend::softmax_backward(
  ScalarT const& alpha_in,
  TensorDescriptor const& outputDesc,
  El::Matrix<DataT, El::Device::CPU> const& local_output,
  TensorDescriptor const& outputGradDesc,
  El::Matrix<DataT, El::Device::CPU> const& local_gradient_wrt_output,
  ScalarT const& beta_in,
  TensorDescriptor const& inputGradDesc,
  El::Matrix<DataT, El::Device::CPU>& local_gradient_wrt_input,
  El::SyncInfo<El::Device::CPU> const& si,
  softmax_mode mode,
  softmax_alg alg)
{
  if (alg == softmax_alg::LOG)
    return logsoftmax_backward(alpha_in,
                               outputDesc,
                               local_output,
                               outputGradDesc,
                               local_gradient_wrt_output,
                               beta_in,
                               inputGradDesc,
                               local_gradient_wrt_input,
                               si,
                               mode);

  if (mode != softmax_mode::INSTANCE) {
    LBANN_ERROR("Unsupported softmax mode");
  }

  // Local matrices
  using workspace_type = hydrogen::simple_buffer<DataT, El::Device::CPU>;
  const auto& local_height = local_output.Height();
  const auto& local_width = local_output.Width();
  workspace_type local_workspace(local_width,
                                 /*value=*/El::TypeTraits<DataT>::Zero(),
                                 si,
                                 /*memory_mode=*/0);
  auto* const workspace = local_workspace.data();

  // Compute dot products between output and gradient w.r.t. output
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    auto& y_dot_dy = workspace[col];
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& y = local_output(row, col);
      const auto& dy = local_gradient_wrt_output(row, col);
      y_dot_dy += y * dy;
    }
  }

  // Compute gradient w.r.t. input
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& y_dot_dy = workspace[col];
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& y = local_output(row, col);
      const auto& dy = local_gradient_wrt_output(row, col);
      auto& dx = local_gradient_wrt_input(row, col);
      dx = y * (dy - y_dot_dy);
    }
  }
}
} // namespace lbann
#endif // LBANN_UTILS_DNN_LIB_OPENMP_SOFTMAX_HPP_
