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

#include "lbann/layers/activations/softmax.hpp"

namespace lbann {

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarMRMat<El::Device::CPU>(grid);
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarVCMat<El::Device::CPU>(grid);
}

#ifdef LBANN_HAS_GPU
template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarMRMat<El::Device::GPU>(grid);
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarVCMat<El::Device::GPU>(grid);
}
#endif // LBANN_HAS_GPU

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_compute_cpu();
}

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_compute_cpu();
}

#ifdef LBANN_HAS_GPU
template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
  throw new lbann_exception("Unimplemented method");
}

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
  throw new lbann_exception("Unimplemented method");
}
#endif // LBANN_HAS_GPU

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_compute_cpu();
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_compute_cpu();
}

#ifdef LBANN_HAS_GPU
template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  throw lbann_exception("softmax_layer: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Matrices
  const auto& prev_activations = get_prev_activations();
  auto& activations = get_activations();

  // Apply softmax on the GPU
  CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
  CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(),
                             this->m_cudnn->get_stream()));
  CHECK_CUDNN(cudnnSoftmaxForward(this->m_cudnn->get_handle(),
                                  CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_INSTANCE,
                                  &one,
                                  this->m_prev_activations_cudnn_desc,
                                  prev_activations.LockedBuffer(),
                                  &zero,
                                  this->m_activations_cudnn_desc,
                                  activations.Buffer()));

#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
  // Round to minimum value to avoid denormalized floats
  softmax_cuda::fp_cutoff(*this->m_cudnn,
                          activations.Buffer(),
                          get_num_neurons(),
                          this->m_mini_batch_size_per_gpu,
                          m_min_output);
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF

#endif // LBANN_HAS_CUDNN
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  throw lbann_exception("softmax_layer: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;

  // Matrices
  const auto& activations = get_activations();
  const auto& prev_error_signals = get_prev_error_signals();
  auto& error_signals = get_error_signals();

  // Apply softmax on each GPU
  CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
  CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(),
                             this->m_cudnn->get_stream()));
  CHECK_CUDNN(cudnnSoftmaxBackward(this->m_cudnn->get_handle(),
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_INSTANCE,
                                   &one,
                                   this->m_activations_cudnn_desc,
                                   activations.LockedBuffer(),
                                   this->m_prev_error_signals_cudnn_desc,
                                   prev_error_signals.LockedBuffer(),
                                   &one,
                                   this->m_error_signals_cudnn_desc,
                                   error_signals.Buffer()));

#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
  // Round to minimum value to avoid denormalized floats
  softmax_cuda::bp_cutoff(*this->m_cudnn,
                          activations.LockedBuffer(),
                          error_signals.Buffer(),
                          get_num_neurons(),
                          this->m_mini_batch_size_per_gpu,
                          this->m_min_output);
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF

#endif // LBANN_HAS_CUDNN
}
#endif // LBANN_HAS_GPU

} // namespace lbann
