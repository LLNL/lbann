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
template <class T_layout>
class relu_layer : public entrywise_activation_layer<T_layout> {

 private:

#ifdef __LIB_CUDNN
  /// Tensor descriptor
  cudnnTensorDescriptor_t m_tensor_desc;
  /// Activation descriptor
  cudnnActivationDescriptor_t m_activation_desc;
#endif

 public:
  relu_layer(data_layout data_dist,
             uint index,
             lbann_comm *comm,
             uint mini_batch_size,
             uint num_neurons,
             cudnn::cudnn_manager *cudnn = NULL) :
    entrywise_activation_layer<T_layout>(data_dist, index, comm,
                                         mini_batch_size, num_neurons) {

  #ifdef __LIB_CUDNN

    m_tensor_desc = NULL;
    m_activation_desc = NULL;

    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
      const int num_gpus = this->m_cudnn->get_num_gpus();
      const int num_processes = this->m_comm->get_procs_per_model();
      const int local_mini_batch_size = (mini_batch_size + num_processes - 1) / num_processes;
      this->m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;
    }

  #endif // #ifdef __LIB_CUDNN

  }

  virtual ~relu_layer() {

  #ifdef __LIB_CUDNN
    if(this->m_using_gpus) {

      // Destroy cuDNN objects
      if(m_tensor_desc) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_tensor_desc));
      }
      if(m_activation_desc) {
        CHECK_CUDNN(cudnnDestroyActivationDescriptor(m_activation_desc));
      }

      // Deallocate GPU memory
      this->m_cudnn->deallocate_on_gpus(this->m_activations_d);
      this->m_cudnn->deallocate_on_gpus(this->m_error_signal_d);
      if(!this->m_prev_layer_using_gpus) {
        this->m_cudnn->deallocate_on_gpus(this->m_prev_activations_d);
      }
      if(!this->m_next_layer_using_gpus) {
        this->m_cudnn->deallocate_on_gpus(this->m_prev_error_signal_d);
      }

    }
  #endif

  }

  void setup(int num_prev_neurons) {
    entrywise_activation_layer<T_layout>::setup(num_prev_neurons);

  #ifdef __LIB_CUDNN
    // Setup cuDNN objects
    if(this->m_using_gpus) {
      setup_gpu();
    }
  #endif // #ifdef __LIB_CUDNN

  }

void setup_gpu() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("relu_layer: cuDNN not detected");
  #else

    // Initialize descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_tensor_desc));
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&m_activation_desc));

    // Set tensor descriptor
    int tensor_dims = this->m_num_neurons * this->m_mini_batch_size_per_gpu;
    int one = 1;
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_tensor_desc,
                                           this->m_cudnn->get_cudnn_data_type(),
                                           1,
                                           &tensor_dims,
                                           &one));
    
    // Set activation descriptor
    CHECK_CUDNN(cudnnSetActivationDescriptor(m_activation_desc,
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
                                         m_activation_desc,
                                         &one,
                                         m_tensor_desc,
                                         this->m_activations_d[i],
                                         &zero,
                                         m_tensor_desc,
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
                                          m_activation_desc,
                                          &one,
                                          m_tensor_desc,
                                          this->m_prev_activations_d[i],
                                          m_tensor_desc,
                                          this->m_prev_error_signal_d[i],
                                          m_tensor_desc, 
                                          this->m_activations_d[i],
                                          &zero,
                                          m_tensor_desc,
                                          this->m_error_signal_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

};


}  // namespace lbann

#endif  // LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
