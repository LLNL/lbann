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

#include "lbann/layers/misc/variance.hpp"

namespace lbann {

namespace {

/** CPU forward prop implementation.
 *  We use a two-pass algorithm since it is more numerically stable
 *  than the naive single-pass algorithm.
 */
void fp_cpu(const AbsDistMat& input,
            AbsDistMat& output,
            AbsDistMat& means,
            AbsDistMat& workspace,
            bool biased) {

  // Local matrices
  const auto& local_input = static_cast<const CPUMat&>(input.LockedMatrix());
  auto& local_means = static_cast<CPUMat&>(means.Matrix());
  auto& local_workspace = static_cast<CPUMat&>(workspace.Matrix());

  // Dimensions
  const auto& height = input.Height();
  const auto& width = input.Width();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // Compute column-wise mean
  means.Empty(false);
  means.AlignWith(input);
  means.Resize(1, width);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    DataType sum = 0;
    for (El::Int row = 0; row < local_height; ++row) {
      sum += local_input(row, col);
    }
    local_means(0, col) = sum / height;
  }
  El::AllReduce(means, means.RedundantComm());

  // Compute column-wise variance
  workspace.Empty(false);
  workspace.AlignWith(input);
  workspace.Resize(1, width);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& mean = local_means(0, col);
    DataType sum = 0;
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& diff = local_input(row, col) - mean;
      sum += diff * diff;
    }
    local_workspace(0, col) = sum / (biased ? height : height - 1);
  }
  El::AllReduce(workspace, workspace.RedundantComm());
  El::Copy(workspace, output);

}

/** CPU backprop implementation.
 *  Means have already been computed in forward prop.
 */
void bp_cpu(const AbsDistMat& input,
            const AbsDistMat& gradient_wrt_output,
            AbsDistMat& gradient_wrt_input,
            const AbsDistMat& means,
            AbsDistMat& workspace,
            bool biased) {

  // Local matrices
  const auto& local_input = static_cast<const CPUMat&>(input.LockedMatrix());
  auto& local_gradient_wrt_input = static_cast<CPUMat&>(gradient_wrt_input.Matrix());
  const auto& local_means = static_cast<const CPUMat&>(means.LockedMatrix());
  auto& local_workspace = static_cast<CPUMat&>(workspace.Matrix());

  // Dimensions
  const auto& height = input.Height();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // Initialize workspace with gradients w.r.t. output
  El::Copy(gradient_wrt_output, workspace);

  // Compute gradients w.r.t. input
  const DataType scale = DataType(2) / (biased? height : height - 1);
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& dy = local_workspace(0, col);
      const auto& x = local_input(row, col);
      const auto& mean = local_means(0, col);
      auto& dx = local_gradient_wrt_input(row, col);
      dx = dy * scale * (x - mean);
    }
  }

}

} // namespace

template <>
void variance_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::fp_compute() {
  fp_cpu(get_prev_activations(),
         get_activations(),
         *m_means,
         *m_workspace,
         m_biased);
}

template <>
void variance_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::bp_compute() {
  bp_cpu(get_prev_activations(),
         get_prev_error_signals(),
         get_error_signals(),
         *m_means,
         *m_workspace,
         m_biased);
}

template <>
void variance_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::fp_compute() {
  fp_cpu(get_prev_activations(),
         get_activations(),
         *m_means,
         *m_workspace,
         m_biased);
}

template <>
void variance_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::bp_compute() {
  bp_cpu(get_prev_activations(),
         get_prev_error_signals(),
         get_error_signals(),
         *m_means,
         *m_workspace,
         m_biased);
}

} // namespace lbann
