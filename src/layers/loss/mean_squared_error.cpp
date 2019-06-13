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

#include "lbann/layers/loss/mean_squared_error.hpp"

namespace lbann {

namespace {

void local_fp_cpu(El::Int height,
                  const AbsMat& local_prediction,
                  const AbsMat& local_ground_truth,
                  AbsMat& local_contribution) {

  // Useful constants
  const auto& local_height = local_prediction.Height();
  const auto& local_width = local_prediction.Width();

  // Compute local contribution to mean squared error
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    DataType sum = 0;
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& err = (local_prediction(row, col)
                         - local_ground_truth(row, col));
      sum += err * err;
    }
    local_contribution(0, col) = sum / height;
  }

}

void local_bp_cpu(El::Int height,
                  const AbsMat& local_prediction,
                  const AbsMat& local_ground_truth,
                  const AbsMat& local_gradient_wrt_output,
                  AbsMat& local_gradient_wrt_prediction,
                  AbsMat& local_gradient_wrt_ground_truth) {

  // Useful constants
  const DataType scale = DataType(2) / height;
  const El::Int local_height = local_prediction.Height();
  const El::Int local_width = local_prediction.Width();

  // Compute gradients
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& err = (local_prediction(row, col)
                         - local_ground_truth(row, col));
      const auto& dy = local_gradient_wrt_output(0, col);
      local_gradient_wrt_prediction(row, col) = scale * err * dy;
      local_gradient_wrt_ground_truth(row, col) = - scale * err * dy;
    }
  }

}

} // namespace

template <>
void mean_squared_error_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::local_fp_compute(El::Int height,
                        const AbsMat& local_prediction,
                        const AbsMat& local_ground_truth,
                        AbsMat& local_contribution) {
  local_fp_cpu(height, local_prediction, local_ground_truth,
               local_contribution);
}

template <>
void mean_squared_error_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::local_bp_compute(El::Int height,
                        const AbsMat& local_prediction,
                        const AbsMat& local_ground_truth,
                        const AbsMat& local_gradient_wrt_output,
                        AbsMat& local_gradient_wrt_prediction,
                        AbsMat& local_gradient_wrt_ground_truth) {
  local_bp_cpu(height,
               local_prediction,
               local_ground_truth,
               local_gradient_wrt_output,
               local_gradient_wrt_prediction,
               local_gradient_wrt_ground_truth);
}

template <>
void mean_squared_error_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::local_fp_compute(El::Int height,
                        const AbsMat& local_prediction,
                        const AbsMat& local_ground_truth,
                        AbsMat& local_contribution) {
  local_fp_cpu(height, local_prediction, local_ground_truth,
               local_contribution);
}

template <>
void mean_squared_error_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::local_bp_compute(El::Int height,
                        const AbsMat& local_prediction,
                        const AbsMat& local_ground_truth,
                        const AbsMat& local_gradient_wrt_output,
                        AbsMat& local_gradient_wrt_prediction,
                        AbsMat& local_gradient_wrt_ground_truth) {
  local_bp_cpu(height,
               local_prediction,
               local_ground_truth,
               local_gradient_wrt_output,
               local_gradient_wrt_prediction,
               local_gradient_wrt_ground_truth);
}

} // namespace lbann
