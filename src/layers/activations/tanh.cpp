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

#include "lbann/layers/activations/tanh.hpp"

namespace lbann {

// Model-parallel CPU forward/backward prop
template <>
void tanh_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  entrywise_activation_layer::fp_compute_cpu();
}
template <>
void tanh_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  entrywise_activation_layer::bp_compute_cpu();
}

// Data-parallel CPU forward/backward prop
template <>
void tanh_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  entrywise_activation_layer::fp_compute_cpu();
}
template <>
void tanh_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  entrywise_activation_layer::bp_compute_cpu();
}

#ifdef LBANN_HAS_GPU

// Model-parallel GPU forward/backward prop
template <>
void tanh_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  const DataType zero = DataType(0);
  const DataType one = DataType(1);
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
    CHECK_CUDNN(cudnnActivationForward(this->m_cudnn->get_handle(),
                                       m_activation_cudnn_desc,
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
void tanh_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  const DataType zero = DataType(0);
  const DataType one = DataType(1);
  const auto& local_input = get_local_prev_activations();
  const auto& local_output = get_local_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
      CHECK_CUDNN(cudnnActivationBackward(this->m_cudnn->get_handle(),
                                          m_activation_cudnn_desc,
                                          &one,
                                          m_tensors_cudnn_desc.get_activations(),
                                          local_output.LockedBuffer(),
                                          m_tensors_cudnn_desc.get_prev_error_signals(),
                                          local_gradient_wrt_output.LockedBuffer(),
                                          m_tensors_cudnn_desc.get_prev_activations(),
                                          local_input.LockedBuffer(),
                                          &zero,
                                          m_tensors_cudnn_desc.get_error_signals(),
                                          local_gradient_wrt_input.Buffer()));
  }
#endif // LBANN_HAS_CUDNN
}

// Data-parallel GPU forward/backward prop
template <>
void tanh_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  const DataType zero = DataType(0);
  const DataType one = DataType(1);
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
    CHECK_CUDNN(cudnnActivationForward(this->m_cudnn->get_handle(),
                                       m_activation_cudnn_desc,
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
void tanh_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  const DataType zero = DataType(0);
  const DataType one = DataType(1);
  const auto& local_input = get_local_prev_activations();
  const auto& local_output = get_local_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
      CHECK_CUDNN(cudnnActivationBackward(this->m_cudnn->get_handle(),
                                          m_activation_cudnn_desc,
                                          &one,
                                          m_tensors_cudnn_desc.get_activations(),
                                          local_output.LockedBuffer(),
                                          m_tensors_cudnn_desc.get_prev_error_signals(),
                                          local_gradient_wrt_output.LockedBuffer(),
                                          m_tensors_cudnn_desc.get_prev_activations(),
                                          local_input.LockedBuffer(),
                                          &zero,
                                          m_tensors_cudnn_desc.get_error_signals(),
                                          local_gradient_wrt_input.Buffer()));
  }
#endif // LBANN_HAS_CUDNN
}

#endif // LBANN_HAS_GPU

} // namespace lbann
