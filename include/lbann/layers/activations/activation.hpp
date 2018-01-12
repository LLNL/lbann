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

#ifndef LBANN_LAYER_ACTIVATION_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Abstract base class for activation layer.
 *  Activation layers implement the nonlinear activation functions
 *  common in neural networks.
 */
class activation_layer : public Layer {
 public:
  /** Constructor. */
  activation_layer(lbann_comm *comm) : Layer(comm) {}
};

/** Abstract base class for entry-wise activation layer.
 *  A nonlinear activation function is applied independently to each
 *  input entry.
 */
class entrywise_activation_layer : public activation_layer {

 public:

  /** Constructor. */
  entrywise_activation_layer(lbann_comm *comm)
    : activation_layer(comm) {}

 protected:

  /** Activation function.
   *  This function is applied independently to each input entry.
   */
  virtual DataType activation(DataType x) const = 0;

  /** Derivative of activation function. */
  virtual DataType activation_derivative(DataType x) const = 0;

  void fp_compute() override {
    if(this->m_using_gpus) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
      bp_compute_gpu();
    } else {
      bp_compute_cpu();
    }
  }

  virtual void fp_compute_gpu() {
    throw lbann_exception("entrywise_activation_layer: no forward propagation GPU implementation");
  }

  virtual void bp_compute_gpu() {
    throw lbann_exception("entrywise_activation_layer: no backward propagation GPU implementation");
  }

  virtual void fp_compute_cpu() {

    // Input and output matrices
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();

    // Matrix parameters
    const int local_height = local_input.Height();
    const int local_width = local_input.Width();
    const int input_ldim = local_input.LDim();
    const int output_ldim = local_output.LDim();
    const DataType* __restrict__ input_buffer = local_input.LockedBuffer();
    DataType* __restrict__ output_buffer = local_output.Buffer();

    // Apply activation function to each input entry
    if (input_ldim == local_height && output_ldim == local_height) {
      // Contiguous data
      const size_t buffer_size = local_height * local_width;
      #pragma omp parallel for
      for (size_t i = 0; i < buffer_size; ++i) {
        output_buffer[i] = activation(input_buffer[i]);
      }
    } else {
      // Non-contiguous data
      #pragma omp parallel for collapse(2)
      for(int col = 0; col < local_width; ++col) {
        for(int row = 0; row < local_height; ++row) {
          const auto& x = input_buffer[row + col * input_ldim];
          auto& y = output_buffer[row + col * output_ldim];
          y = activation(x);
        }
      }
    }

  }

  virtual void bp_compute_cpu() {

    // Input and output matrices
    const auto& local_input = get_local_prev_activations();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();

    // Matrix parameters
    const int local_height = local_input.Height();
    const int local_width = local_input.Width();
    const int input_ldim = local_input.LDim();
    const int gradient_wrt_output_ldim = local_gradient_wrt_output.LDim();
    const int gradient_wrt_input_ldim = local_gradient_wrt_input.LDim();
    const DataType* __restrict__ input_buffer = local_input.LockedBuffer();
    const DataType* __restrict__ gradient_wrt_output_buffer
      = local_gradient_wrt_output.LockedBuffer();
    DataType* __restrict__ gradient_wrt_input_buffer
      = local_gradient_wrt_input.Buffer();

    // Apply activation function to each input entry
    if (input_ldim == local_height
        && gradient_wrt_output_ldim == local_height
        && gradient_wrt_input_ldim == local_height) {
      // Contiguous data
      const size_t buffer_size = local_height * local_width;
      #pragma omp parallel for
      for (size_t i = 0; i < buffer_size; ++i) {
        const auto& x = input_buffer[i];
        const auto& dy = gradient_wrt_output_buffer[i];
        auto& dx = gradient_wrt_input_buffer[i];
        dx += dy * activation_derivative(x);
      }
    } else {
      // Non-contiguous data
      #pragma omp parallel for collapse(2)
      for(int col = 0; col < local_width; ++col) {
        for(int row = 0; row < local_height; ++row) {
          const auto& x = input_buffer[row + col * input_ldim];
          const auto& dy
            = gradient_wrt_output_buffer[row + col * gradient_wrt_output_ldim];
          auto& dx
            = gradient_wrt_input_buffer[row + col * gradient_wrt_input_ldim];
          dx += dy * activation_derivative(x);
        }
      }
    }

  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_ACTIVATION_HPP_INCLUDED
