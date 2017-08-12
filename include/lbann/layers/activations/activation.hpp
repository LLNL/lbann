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

class activation_layer : public Layer {

 public:
  activation_layer(int index,
                   lbann_comm *comm) :
    Layer(index, comm) {
  }
  activation_layer(const activation_layer&) = default;
  activation_layer& operator=(const activation_layer&) = default;

  virtual ~activation_layer() {}

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    Layer::initialize_distributed_matrices<T_layout>();
  }

};

class entrywise_activation_layer : public activation_layer {

 public:
  entrywise_activation_layer(int index,
                             lbann_comm *comm) :
    activation_layer(index, comm) {
  }
  entrywise_activation_layer(const entrywise_activation_layer&) = default;
  entrywise_activation_layer& operator=(
    const entrywise_activation_layer&) = default;

  virtual ~entrywise_activation_layer() {}

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    activation_layer::initialize_distributed_matrices<T_layout>();
  }

 protected:
  
  virtual DataType activation_function(DataType x) = 0;
  virtual DataType activation_function_gradient(DataType x) = 0;

  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() {
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
    
    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();

    // Local matrix parameters
    const int local_height = prev_activations_local.Height();
    const int local_width = prev_activations_local.Width();
    const int prev_activations_ldim = prev_activations_local.LDim();
    const int activations_ldim = activations_local.LDim();
    const DataType* __restrict__ prev_activations_buffer 
      = prev_activations_local.LockedBuffer();
    DataType* __restrict__ activations_buffer = activations_local.Buffer();
    
    // Apply activation function
    if(prev_activations_ldim == local_height
       && activations_ldim == local_height) {
      // Contiguous data
      #pragma omp parallel for
      for(int i = 0; i < local_height * local_width; ++i) {
        const DataType prev_activations_entry = prev_activations_buffer[i];
        DataType& activations_entry = activations_buffer[i];
        activations_entry = activation_function(prev_activations_entry);
      }
    } else {
      // Non-contiguous data
      #pragma omp parallel for collapse(2)
      for(int col = 0; col < local_width; ++col) {
        for(int row = 0; row < local_height; ++row) {
          const DataType prev_activations_entry
            = prev_activations_buffer[row + col * prev_activations_ldim];
          DataType& activations_entry
            = activations_buffer[row + col * activations_ldim];
          activations_entry = activation_function(prev_activations_entry);
        }
      }
    }

  }

  virtual void bp_compute_cpu() {
    
    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal_v->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();

    // Local matrix parameters
    const int local_height = prev_activations_local.Height();
    const int local_width = prev_activations_local.Width();
    const int prev_activations_ldim = prev_activations_local.LDim();
    const int prev_error_signal_ldim = prev_error_signal_local.LDim();
    const int error_signal_ldim = error_signal_local.LDim();
    const DataType* __restrict__ prev_activations_buffer 
      = prev_activations_local.LockedBuffer();
    const DataType* __restrict__ prev_error_signal_buffer 
      = prev_error_signal_local.LockedBuffer();
    DataType* __restrict__ error_signal_buffer = error_signal_local.Buffer();
    
    // Apply activation function back propagation
    if(prev_activations_ldim == local_height
       && prev_error_signal_ldim == local_height
       && error_signal_ldim == local_height) {
      // Contiguous data
      #pragma omp parallel for
      for(int i = 0; i < local_height * local_width; ++i) {
        const DataType prev_activations_entry = prev_activations_buffer[i];
        const DataType prev_error_signal_entry = prev_error_signal_buffer[i];
        DataType& error_signal_entry = error_signal_buffer[i];
        error_signal_entry
          = activation_function_gradient(prev_activations_entry) * prev_error_signal_entry;
      }
    } else {
      // Non-contiguous data
      #pragma omp parallel for collapse(2)
      for(int col = 0; col < local_width; ++col) {
        for(int row = 0; row < local_height; ++row) {
          const DataType prev_activations_entry
            = prev_activations_buffer[row + col * prev_activations_ldim];
          const DataType prev_error_signal_entry
            = prev_error_signal_buffer[row + col * prev_error_signal_ldim];
          DataType& error_signal_entry
            = error_signal_buffer[row + col * error_signal_ldim];
          error_signal_entry
            = activation_function_gradient(prev_activations_entry) * prev_error_signal_entry;
        }
      }
    }

  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_ACTIVATION_HPP_INCLUDED
