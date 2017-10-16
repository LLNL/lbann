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

#ifndef LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

namespace lbann {

/**
 * Rectified linear unit activation function.
 * See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
template <data_layout T_layout>
class relu_layer : public entrywise_activation_layer {

 private:

#ifdef __LIB_CUDNN
  /// Activation descriptor
  cudnnActivationDescriptor_t m_activation_cudnn_desc;
#endif

 public:
  relu_layer(int index,
             lbann_comm *comm,
             cudnn::cudnn_manager *cudnn = nullptr) :
    entrywise_activation_layer(index, comm) {

    initialize_distributed_matrices();

  #ifdef __LIB_CUDNN

    m_activation_cudnn_desc = nullptr;

    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }

  #endif // #ifdef __LIB_CUDNN

  }

  relu_layer(const relu_layer& other) :
    entrywise_activation_layer(other) {
  #ifdef __LIB_CUDNN
    m_activation_cudnn_desc = nullptr;
    cudnn::copy_activation_cudnn_desc(other.m_activation_cudnn_desc,
                                      m_activation_cudnn_desc);
  #endif // __LIB_CUDNN
  }

  relu_layer& operator=(const relu_layer& other) {
    entrywise_activation_layer::operator=(other);
#ifdef __LIB_CUDNN
    cudnn::copy_activation_cudnn_desc(other.m_activation_cudnn_desc,
                                      m_activation_cudnn_desc);
#endif // __LIB_CUDNN
  }

  virtual ~relu_layer() {
  #ifdef __LIB_CUDNN
    if(m_activation_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyActivationDescriptor(m_activation_cudnn_desc));
    }
  #endif
  }

  relu_layer* copy() const { return new relu_layer(*this); }

  std::string get_type() const { return "relu"; }

  /** Returns description of ctor params */
  std::string get_description() const {
    return std::string {} +
     " relu" + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  virtual inline void initialize_distributed_matrices() {
    entrywise_activation_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_gpu() {
    entrywise_activation_layer::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("relu_layer: cuDNN not detected");
  #else

    // Initialize activation descriptor
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&m_activation_cudnn_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(m_activation_cudnn_desc,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             0.0));
  #endif
  }

 protected:

  DataType activation_function(DataType x) {
    return x > DataType(0) ? x : DataType(0);
  }

  DataType activation_function_gradient(DataType x) {
    return x > DataType(0) ? DataType(1) : DataType(0);
  }

  void fp_compute_gpu() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("relu_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Apply application on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnActivationForward(this->m_cudnn->get_handle(i),
                                         m_activation_cudnn_desc,
                                         &one,
                                         this->m_prev_neurons_cudnn_desc,
                                         this->m_prev_activations_d[i],
                                         &zero,
                                         this->m_neurons_cudnn_desc,
                                         this->m_activations_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

  void bp_compute_gpu() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("relu_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Apply application on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnActivationBackward(this->m_cudnn->get_handle(i),
                                          m_activation_cudnn_desc,
                                          &one,
                                          this->m_prev_neurons_cudnn_desc,
                                          this->m_prev_activations_d[i],
                                          this->m_neurons_cudnn_desc,
                                          this->m_prev_error_signal_d[i],
                                          this->m_neurons_cudnn_desc,
                                          this->m_activations_d[i],
                                          &zero,
                                          this->m_prev_neurons_cudnn_desc,
                                          this->m_error_signal_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

};


}  // namespace lbann

#endif  // LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
