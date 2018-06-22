////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/loss/cross_entropy.hpp"

namespace lbann {

namespace {

/** CPU implementation of cross entropy.
 *  If all other matrices have colDist,rowDist data distribution, then
 *  workspace is assumed to have El::STAR,rowDist data distribution.
 */
void fp_cpu(lbann_comm& comm,
            const AbsDistMat& prediction,
            const AbsDistMat& ground_truth,
            AbsDistMat& output,
            AbsDistMat& workspace) {
  
  // Initialize matrices
  workspace.AlignWith(prediction.DistData());
  workspace.Resize(1, prediction.Width());
  auto& local_workspace = workspace.Matrix();
  const auto& local_prediction = prediction.LockedMatrix();
  const auto& local_ground_truth = ground_truth.LockedMatrix();

  // Useful constants
  const DataType zero = DataType(0);
  const El::Int local_height = local_prediction.Height();
  const El::Int local_width = local_prediction.Width();

  // Compute local contribution to cross entropy
#pragma omp parallel for
  for (El::Int col = 0; col < local_width; ++col) {
    DataType sum = zero;
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& true_val = local_ground_truth(row, col);
      const auto& pred_val = local_prediction(row, col);
      if (true_val > zero) {
#ifdef LBANN_DEBUG
        if (pred_val <= zero) { LBANN_ERROR("non-positive prediction"); }
#endif // LBANN_DEBUG
        sum += - true_val * std::log(pred_val);
      }
    }
    local_workspace(0, col) = sum;
  }

  // Accumulate local contributions
  /// @todo Consider reduce rather than allreduce
  comm.allreduce(workspace, workspace.RedundantComm());
  El::Copy(workspace, output);

}

/** CPU implementation of cross entropy backprop.
 *  If all other matrices have colDist,rowDist data distribution, then
 *  workspace is assumed to have El::STAR,rowDist data distribution.
 */
void bp_cpu(const AbsDistMat& prediction,
            const AbsDistMat& ground_truth,
            const AbsDistMat& gradient_wrt_output,
            AbsDistMat& gradient_wrt_prediction,
            AbsDistMat& gradient_wrt_ground_truth,
            AbsDistMat& workspace) {
  
  // Initialize matrices
  El::Copy(gradient_wrt_output, workspace);
  const auto& local_workspace = workspace.LockedMatrix();
  const auto& local_prediction = prediction.LockedMatrix();
  const auto& local_ground_truth = ground_truth.LockedMatrix();
  auto& local_gradient_wrt_prediction = gradient_wrt_prediction.Matrix();
  auto& local_gradient_wrt_ground_truth = gradient_wrt_ground_truth.Matrix();

  // Useful constants
  const DataType zero = DataType(0);
  const El::Int local_height = local_prediction.Height();
  const El::Int local_width = local_prediction.Width();

  // Compute gradients
#pragma omp parallel for
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& x = local_prediction(row, col);
      const auto& xhat = local_ground_truth(row, col);
      const auto& dy = local_workspace(0, col);
      auto& dx = local_gradient_wrt_prediction(row, col);
      auto& dxhat = local_gradient_wrt_ground_truth(row, col);
      dx = (xhat > zero) ? - dy * xhat / x : zero;
      dxhat = - dy * std::log(x);
    }
  }

}

} // namespace

template <>
void cross_entropy_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  El::DistMatrix<DataType, El::STAR, El::MR, El::ELEMENT, El::Device::CPU> workspace;
  fp_cpu(*this->m_comm,
         get_prev_activations(0),
         get_prev_activations(1),
         get_activations(),
         workspace);
}

template <>
void cross_entropy_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  El::DistMatrix<DataType, El::STAR, El::MR, El::ELEMENT, El::Device::CPU> workspace;
  bp_cpu(get_prev_activations(0),
         get_prev_activations(1),
         get_prev_error_signals(),
         get_error_signals(0),
         get_error_signals(1),
         workspace);
}

template <>
void cross_entropy_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, El::Device::CPU> workspace;
  fp_cpu(*this->m_comm,
         get_prev_activations(0),
         get_prev_activations(1),
         get_activations(),
         workspace);
}

template <>
void cross_entropy_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, El::Device::CPU> workspace;
  bp_cpu(get_prev_activations(0),
         get_prev_activations(1),
         get_prev_error_signals(),
         get_error_signals(0),
         get_error_signals(1),
         workspace);
}

} // namespace lbann
