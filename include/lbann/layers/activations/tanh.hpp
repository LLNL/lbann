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

#ifndef LBANN_LAYER_ACTIVATION_TANH_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_TANH_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

namespace lbann {

#ifdef LBANN_HAS_CUDNN
namespace tanh_cuda {
  void fp(cudnn::cudnn_manager& cudnn,
          int height,
          int width_per_gpu,
          const DataType* input,
          int input_leading_dim,
          DataType* output,
          int output_leading_dim);
  void bp(cudnn::cudnn_manager& cudnn,
          int height, int width_per_gpu,
          const DataType* input,
          int input_leading_dim,
          const DataType* gradient_wrt_output,
          int gradient_wrt_output_leading_dim,
          DataType* gradient_wrt_input,
          int gradient_wrt_input_leading_dim);
} // namespace tanh_cuda
#endif // LBANN_HAS_CUDNN

/** Hyperbolic tangent activation function. */
template <data_layout T_layout, El::Device Dev>
class tanh_layer : public entrywise_activation_layer {
 public:
  tanh_layer(lbann_comm *comm,
             cudnn::cudnn_manager *cudnn = nullptr)
    : entrywise_activation_layer(comm) {

  #ifdef LBANN_HAS_CUDNN
    // Activate GPU if needed
    if (cudnn != nullptr && T_layout == data_layout::DATA_PARALLEL) {
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  tanh_layer* copy() const override { return new tanh_layer(*this); }
  std::string get_type() const override { return "tanh"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:

  DataType activation(DataType x) const override {
    return std::tanh(x);
  }

  DataType activation_derivative(DataType x) const override {
    const DataType coshx = std::cosh(x);
    return 1 / (coshx * coshx);
  }

  void fp_compute_gpu() override {
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else
    tanh_cuda::fp(*m_cudnn,
                  get_num_neurons(),
                  m_mini_batch_size_per_gpu,
                  get_prev_activations().LockedBuffer(),
                  get_prev_activations().LDim(),
                  get_activations().Buffer(),
                  get_activations().LDim());
  #endif // LBANN_HAS_CUDNN
  }

  void bp_compute_gpu() override {
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else
    tanh_cuda::bp(*m_cudnn,
                  get_num_neurons(),
                  m_mini_batch_size_per_gpu,
                  get_prev_activations().LockedBuffer(),
                  get_prev_activations().LDim(),
                  get_prev_error_signals().LockedBuffer(),
                  get_prev_error_signals().LDim(),
                  get_error_signals().Buffer(),
                  get_error_signals().LDim());
  #endif // LBANN_HAS_CUDNN
  }

};

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_TANH_HPP_INCLUDED
