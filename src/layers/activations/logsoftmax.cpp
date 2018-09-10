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

#include "lbann/layers/activations/logsoftmax.hpp"
#ifdef LBANN_HAS_CUDNN
#include "lbann/utils/cublas.hpp"
#endif  // LBANN_HAS_CUDNN

namespace lbann {

template <>
void logsoftmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarMRMat<El::Device::CPU>(grid);
}

template <>
void logsoftmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarVCMat<El::Device::CPU>(grid);
}

#ifdef LBANN_HAS_GPU
template <>
void logsoftmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarMRMat<El::Device::GPU>(grid);
}

template <>
void logsoftmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarVCMat<El::Device::GPU>(grid);
}
#endif // LBANN_HAS_GPU

template <>
void logsoftmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_compute_cpu();
}

template <>
void logsoftmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_compute_cpu();
}

#ifdef LBANN_HAS_GPU
template <>
void logsoftmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {

  // Local matrices.
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  auto& local_workspace = m_workspace->Matrix();

  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();

  // Find the maximum entry in each local column.
  if (local_height == 0) {
    // When there's no local data, fill the workspace with a small value so the
    // maximum across processors is still computed correctly.
    El::Fill(local_workspace, std::numeric_limits<DataType>::lowest());
  } else {
    logsoftmax_cuda::max_local_col_entry(
      local_height, local_width, local_input.LockedBuffer(),
      local_input.LDim(), local_workspace.Buffer(), El::GPUManager::Stream());
  }
  // Find the global max entry in each column.
  m_comm->allreduce(*m_workspace, m_workspace->RedundantComm(), El::mpi::MAX);

  // Exponentiate activations and compute column sums.
  // This subtracts by the column max for stability.
  // Save the sum, shift values for the log-sum-exp trick.
  if (local_height == 0) {
    // Zero out so that we contribute nothing to the sum.
    El::Zero(local_workspace);
  } else {
    logsoftmax_cuda::exp_and_col_sum(
      local_height, local_width, 
      local_input.LockedBuffer(), local_input.LDim(),
      local_output.Buffer(), local_output.LDim(),
      local_workspace.Buffer(), 
      El::GPUManager::Stream());
  }
  // Compute the global sums for each column.
  m_comm->allreduce(*m_workspace, m_workspace->RedundantComm(), El::mpi::SUM);

  // Subtract from activations the log column sums.
  // Shift by the max column entry for the log-sum-exp trick.
  logsoftmax_cuda::sub_by_col_sums_and_shift(
    local_height, local_width, 
    local_input.LockedBuffer(), local_input.LDim(), 
    local_output.Buffer(), local_output.LDim(), 
    local_workspace.LockedBuffer(),
    El::GPUManager::Stream());

}

template <>
void logsoftmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {

  // Local matrices.
  const auto& local_output = get_local_activations();
  const auto& local_grad_wrt_output = get_local_prev_error_signals();
  auto& local_grad_wrt_input = get_local_error_signals();
  auto& local_workspace = m_workspace->Matrix();

  const El::Int local_height = local_output.Height();
  const El::Int local_width = local_output.Width();

  // Compute column sums for gradient w.r.t. output.
  logsoftmax_cuda::out_grad_col_sum(
    local_height, local_width,
    local_workspace.Buffer(),
    local_grad_wrt_output.LockedBuffer(), local_grad_wrt_output.LDim(),
    El::GPUManager::Stream());

  // Compute gradient w.r.t. input.
  logsoftmax_cuda::grad_wrt_input(
    local_height, local_width, local_output.LockedBuffer(),
    local_output.LDim(), local_workspace.LockedBuffer(),
    local_grad_wrt_output.LockedBuffer(), local_grad_wrt_output.LDim(),
    local_grad_wrt_input.Buffer(), local_grad_wrt_input.LDim(),
    El::GPUManager::Stream());

}
#endif // LBANN_HAS_GPU

template <>
void logsoftmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_compute_cpu();
}

template <>
void logsoftmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_compute_cpu();
}

#ifdef LBANN_HAS_GPU
template <>
void logsoftmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  if (local_input.Height() > 0 && local_input.Width() > 0) {

    // Useful constants
    const DataType zero = DataType(0);
    const DataType one = DataType(1);

    // Apply logsoftmax
    CHECK_CUDNN(cudnnSoftmaxForward(cudnn::get_handle(),
                                    CUDNN_SOFTMAX_LOG,
                                    CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &one,
                                    m_tensors_cudnn_desc.get_prev_activations(),
                                    local_input.LockedBuffer(),
                                    &zero,
                                    m_tensors_cudnn_desc.get_activations(),
                                    local_output.Buffer()));
  }
#endif // LBANN_HAS_CUDNN
}

template <>
void logsoftmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  const auto& local_output = get_local_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  if (local_output.Height() > 0 && local_output.Width() > 0) {

    // Useful constants
    const DataType zero = DataType(0);
    const DataType one = DataType(1);
    
    // Perform backprop
    CHECK_CUDNN(cudnnSoftmaxBackward(cudnn::get_handle(),
                                     CUDNN_SOFTMAX_LOG,
                                     CUDNN_SOFTMAX_MODE_INSTANCE,
                                     &one,
                                     m_tensors_cudnn_desc.get_activations(),
                                     local_output.LockedBuffer(),
                                     m_tensors_cudnn_desc.get_prev_error_signals(),
                                     local_gradient_wrt_output.LockedBuffer(),
                                     &zero,
                                     m_tensors_cudnn_desc.get_error_signals(),
                                     local_gradient_wrt_input.Buffer()));
  }
#endif // LBANN_HAS_CUDNN
}
#endif // LBANN_HAS_GPU

} // namespace lbann
