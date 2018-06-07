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

namespace {
#ifdef LBANN_HAS_CUDNN

/** Forward prop on GPU. */
  void fp_gpu(cudnnHandle_t& handle,
              cudnnActivationDescriptor_t& activation_desc,
              cudnnTensorDescriptor_t& input_desc,
              const AbsMat& input,
              cudnnTensorDescriptor_t& output_desc,
              AbsMat& output) {
  if (input.GetDevice() != El::Device::GPU
      || output.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("matrices are not resident on GPU");
  }
  const DataType zero = DataType(0);
  const DataType one = DataType(1);
  if (input.Height() > 0 && input.Width() > 0) {
      cudnn::set_tensor_desc(input_desc, input);
      cudnn::set_tensor_desc(output_desc, output);
      CHECK_CUDNN(cudnnActivationForward(handle,
                                         activation_desc,
                                         &one,
                                         input_desc,
                                         input.LockedBuffer(),
                                         &zero,
                                         output_desc,
                                         output.Buffer()));
  }
}

/** Backward prop on GPU. */
void bp_gpu(cudnnHandle_t& handle,
            cudnnActivationDescriptor_t& activation_desc,
            cudnnTensorDescriptor_t& input_desc,
            const AbsMat& input,
            cudnnTensorDescriptor_t& output_desc,
            const AbsMat& output,
            cudnnTensorDescriptor_t& gradient_wrt_output_desc,
            const AbsMat& gradient_wrt_output,
            cudnnTensorDescriptor_t& gradient_wrt_input_desc,
            AbsMat& gradient_wrt_input) {
  if (input.GetDevice() != El::Device::GPU
      || output.GetDevice() != El::Device::GPU
      || gradient_wrt_output.GetDevice() != El::Device::GPU
      || gradient_wrt_input.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("matrices are not resident on GPU");
  }
  const DataType one = DataType(1);
  if (input.Height() > 0 && input.Width() > 0) {
      cudnn::set_tensor_desc(input_desc, input);
      cudnn::set_tensor_desc(output_desc, output);
      cudnn::set_tensor_desc(gradient_wrt_output_desc, gradient_wrt_output);
      cudnn::set_tensor_desc(gradient_wrt_input_desc, gradient_wrt_input);
      CHECK_CUDNN(cudnnActivationBackward(handle,
                                          activation_desc,
                                          &one,
                                          output_desc,
                                          output.LockedBuffer(),
                                          gradient_wrt_output_desc,
                                          gradient_wrt_output.LockedBuffer(),
                                          input_desc,
                                          input.LockedBuffer(),
                                          &one,
                                          gradient_wrt_input_desc,
                                          gradient_wrt_input.Buffer()));
  }
}

#endif // LBANN_HAS_CUDNN
} // namespace

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
  fp_gpu(this->m_cudnn->get_handle(),
         m_activation_cudnn_desc,
         m_input_cudnn_desc, get_local_prev_activations(),
         m_output_cudnn_desc, get_local_activations());
#endif // LBANN_HAS_CUDNN
}
template <>
void tanh_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  bp_gpu(this->m_cudnn->get_handle(),
         m_activation_cudnn_desc,
         m_input_cudnn_desc, get_local_prev_activations(),
         m_output_cudnn_desc, get_local_activations(),
         m_gradient_wrt_output_cudnn_desc, get_local_prev_error_signals(),
         m_gradient_wrt_input_cudnn_desc, get_local_error_signals());
#endif // LBANN_HAS_CUDNN
}

// Data-parallel GPU forward/backward prop
template <>
void tanh_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  fp_gpu(this->m_cudnn->get_handle(),
         m_activation_cudnn_desc,
         m_input_cudnn_desc, get_local_prev_activations(),
         m_output_cudnn_desc, get_local_activations());
#endif // LBANN_HAS_CUDNN
}
template <>
void tanh_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  bp_gpu(this->m_cudnn->get_handle(),
         m_activation_cudnn_desc,
         m_input_cudnn_desc, get_local_prev_activations(),
         m_output_cudnn_desc, get_local_activations(),
         m_gradient_wrt_output_cudnn_desc, get_local_prev_error_signals(),
         m_gradient_wrt_input_cudnn_desc, get_local_error_signals());
#endif // LBANN_HAS_CUDNN
}

#endif // LBANN_HAS_GPU

} // namespace lbann
