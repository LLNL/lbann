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
//
// lbann_layer_pooling .hpp .cpp - Pooling Layer (max, average)
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_POOLING_HPP_INCLUDED
#define LBANN_LAYER_POOLING_HPP_INCLUDED

#include <vector>
#include "lbann/base.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {

/// Pooling layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class pooling_layer : public transform {
 private:

  /// Pooling mode
  const pool_mode m_pool_mode;

  /// Pooling window dimensions
  std::vector<int> m_pool_dims;
  /// Pooling padding
  std::vector<int> m_pool_pads;
  /// Pooling strides
  std::vector<int> m_pool_strides;
  /// Size of pooling window
  int m_pool_size;

#ifdef __LIB_CUDNN
  /// Pooling descriptor
  cudnnPoolingDescriptor_t m_pooling_desc;
#endif // __LIB_CUDNN

 public:
  /// Constructor
  pooling_layer(int index,
                lbann_comm *comm,
                int mini_batch_size,
                int num_data_dims,
                const int *pool_dims,
                const int *pool_pads,
                const int *pool_strides,
                pool_mode _pool_mode,
                cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm, mini_batch_size),
      m_pool_mode(_pool_mode) {
    // Setup the data distribution
    initialize_distributed_matrices();

    // Initialize input dimensions and pooling parameters
    m_pool_dims.assign(pool_dims, pool_dims+num_data_dims);
    m_pool_size = std::accumulate(m_pool_dims.begin(),
                                  m_pool_dims.end(),
                                  1,
                                  std::multiplies<int>());
    m_pool_pads.assign(pool_pads, pool_pads+num_data_dims);
    m_pool_strides.assign(pool_strides, pool_strides+num_data_dims);

  #ifdef __LIB_CUDNN

    // Initialize cuDNN objects
    m_pooling_desc = NULL;

    // Initialize GPU memory if using GPU
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  pooling_layer(const pooling_layer&) = default;
  pooling_layer& operator=(const pooling_layer&) = default;

  /// Destructor
  ~pooling_layer() {
  #ifdef __LIB_CUDNN
    // Destroy cuDNN objects
    if(m_pooling_desc) {
      CHECK_CUDNN(cudnnDestroyPoolingDescriptor(m_pooling_desc));
    }
  #endif // __LIB_CUDNN
  }

  pooling_layer* copy() const { return new pooling_layer(*this); }

  std::string get_name() const { return "pooling"; }

  virtual inline void initialize_distributed_matrices() {
    transform::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_dims() {
    transform::setup_dims();
    // Initialize neuron tensor dimensions
    this->m_num_neuron_dims = this->m_num_prev_neuron_dims;
    this->m_neuron_dims.resize(this->m_num_neuron_dims);
    this->m_neuron_dims[0] = this->m_prev_neuron_dims[0];
    for(int i=0; i<this->m_num_neuron_dims-1; ++i) {
      const int effective_dim = (this->m_prev_neuron_dims[i+1]
                                 + 2*m_pool_pads[i] - m_pool_dims[i] + 1);
      this->m_neuron_dims[i+1] = ((effective_dim + m_pool_strides[i] - 1)
                                  / m_pool_strides[i]);
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());
  }

  /// Initialize GPU objects
  void setup_gpu() {
    transform::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("lbann_layer_pooling: cuDNN not detected");
  #else

    // Set pooling descriptor
    cudnnPoolingMode_t cudnn_pool_mode;
    switch(m_pool_mode) {
    case pool_mode::max:
      cudnn_pool_mode = CUDNN_POOLING_MAX; break;
    case pool_mode::average:
      cudnn_pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; break;
    case pool_mode::average_no_pad:
      cudnn_pool_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING; break;
    default:
      throw lbann_exception("pooling_layer: no GPU implementation for pooling mode");
    }
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&m_pooling_desc));
    CHECK_CUDNN(cudnnSetPoolingNdDescriptor(m_pooling_desc,
                                            cudnn_pool_mode,
                                            CUDNN_PROPAGATE_NAN,
                                            m_pool_dims.size(),
                                            m_pool_dims.data(),
                                            m_pool_pads.data(),
                                            m_pool_strides.data()));

  #endif // #ifndef __LIB_CUDNN
  }

  protected:

  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_cudnn();
    } else {
      fp_compute_im2col();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      bp_compute_cudnn();
    } else {
      bp_compute_im2col();
    }
  }

 private:

  /// Pooling forward propagation with cuDNN
  void fp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("pooling_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Perform pooling with each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnPoolingForward(this->m_cudnn->get_handle(i),
                                      m_pooling_desc,
                                      &one,
                                      this->m_prev_neurons_cudnn_desc,
                                      this->m_prev_activations_d[i],
                                      &zero,
                                      this->m_neurons_cudnn_desc,
                                      this->m_activations_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

  /// Pooling backward propagation with cuDNN
  void bp_compute_cudnn() {    
  #ifndef __LIB_CUDNN
    throw lbann_exception("pooling_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Get number of GPUs
    const int num_gpus = this->m_cudnn->get_num_gpus();

    // Perform back propagation on each GPU
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnPoolingBackward(this->m_cudnn->get_handle(i),
                                       m_pooling_desc,
                                       &one,
                                       this->m_neurons_cudnn_desc,
                                       this->m_activations_d[i],
                                       this->m_neurons_cudnn_desc,
                                       this->m_prev_error_signal_d[i],
                                       this->m_prev_neurons_cudnn_desc,
                                       this->m_prev_activations_d[i],
                                       &zero,
                                       this->m_prev_neurons_cudnn_desc,
                                       this->m_error_signal_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

  /// Pooling forward propagation with im2col
  void fp_compute_im2col() {

    // Throw exception if pooling mode is not max or average pooling
    if(m_pool_mode != pool_mode::max
        && m_pool_mode != pool_mode::average) {
      throw lbann_exception("pooling_layer: CPU pooling layer only implements max and average pooling");
    }

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();

    // Output entries are divided amongst channels
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_channels;

    // Initialize im2col matrix
    Mat im2col_mat(m_pool_size * num_channels, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {

      // Construct im2col matrix from input
      const Mat input_mat = El::LockedView(prev_activations_local, El::ALL, El::IR(sample));
      im2col(input_mat,
             im2col_mat,
             num_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_pool_pads.data(),
             m_pool_dims.data(),
             m_pool_strides.data());

      // Apply max pooling
      if(m_pool_mode == pool_mode::max) {
        DataType *output_buffer = activations_local.Buffer(0, sample);
        #pragma omp parallel for collapse(2)
        for(int c = 0; c < num_channels; ++c) {
          for(int j = 0; j < num_per_output_channel; ++j) {
            DataType *im2col_buffer = im2col_mat.Buffer(c*m_pool_size, j);
            DataType output_entry = -INFINITY;
            for(int i = 0; i < m_pool_size; ++i) {
              output_entry = std::max(output_entry, im2col_buffer[i]);
            }
            const int output_index = j + c * num_per_output_channel;
            output_buffer[output_index] = output_entry;
          }
        }
      }

      // Apply average pooling
      if(m_pool_mode == pool_mode::average) {
        DataType *output_buffer = activations_local.Buffer(0, sample);
        #pragma omp parallel for collapse(2)
        for(int c = 0; c < num_channels; ++c) {
          for(int j = 0; j < num_per_output_channel; ++j) {
            DataType *im2col_buffer = im2col_mat.Buffer(c*m_pool_size, j);
            DataType output_entry = 0;
            for(int i = 0; i < m_pool_size; ++i) {
              output_entry += im2col_buffer[i];
            }
            output_entry /= m_pool_size;
            const int output_index = j + c * num_per_output_channel;
            output_buffer[output_index] = output_entry;
          }
        }
      }

    }

  }

  /// Pooling forward propagation with im2col
  void bp_compute_im2col() {

    // Throw exception if pooling mode is not max or average pooling
    if(m_pool_mode != pool_mode::max
        && m_pool_mode != pool_mode::average) {
      throw lbann_exception("pooling_layer: CPU pooling layer only implements max and average pooling");
    }

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal_v->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();

    // Output entries are divided amongst channels
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_channels;

    // Initialize im2col matrix
    Mat im2col_mat(m_pool_size * num_channels, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {

      // Compute gradient w.r.t. im2col matrix for max pooling
      if(m_pool_mode == pool_mode::max) {

        // Construct im2col matrix from input
        const Mat input_mat = El::LockedView(prev_activations_local, El::ALL, El::IR(sample));
        im2col(input_mat,
               im2col_mat,
               num_channels,
               this->m_num_prev_neuron_dims - 1,
               this->m_prev_neuron_dims.data() + 1,
               m_pool_pads.data(),
               m_pool_dims.data(),
               m_pool_strides.data());

        // Copy previous error signal to im2col matrix entries
        // corresponding to max
        const DataType *prev_error_signal_buffer
          = prev_error_signal_local.LockedBuffer(0, sample);
        #pragma omp parallel for collapse(2)
        for(int j = 0; j < num_per_output_channel; ++j) {
          for(int c = 0; c < num_channels; ++c) {
            DataType *im2col_buffer = im2col_mat.Buffer(c*m_pool_size, j);
            int max_index = 0;
            DataType max_entry = -INFINITY;
            for(int i = 0; i < m_pool_size; ++i) {
              const DataType current_entry = im2col_buffer[i];
              im2col_buffer[i] = 0;
              if(current_entry > max_entry) {
                max_index = i;
                max_entry = current_entry;
              }
            }
            const int prev_error_signal_index = j + c * num_per_output_channel;
            im2col_buffer[max_index]
              = prev_error_signal_buffer[prev_error_signal_index];
          }
        }

      }

      // Compute gradient w.r.t. im2col matrix for average pooling
      if(m_pool_mode == pool_mode::average) {
        #pragma omp parallel for collapse(2)
        for(int j = 0; j < num_per_output_channel; ++j) {
          for(int c = 0; c < num_channels; ++c) {
            DataType *im2col_buffer = im2col_mat.Buffer(c*m_pool_size, j);
            const int input_index = j + c * num_per_output_channel;
            const DataType output_entry
              = prev_error_signal_local(input_index, sample) / m_pool_size;
            for(int i = 0; i < m_pool_size; ++i) {
              im2col_buffer[i] = output_entry;
            }
          }
        }

      }

      // Compute error signal (i.e. gradient w.r.t. input)
      Mat output_mat = El::View(error_signal_local, El::ALL, El::IR(sample));
      col2im(im2col_mat,
             output_mat,
             num_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_pool_pads.data(),
             m_pool_dims.data(),
             m_pool_strides.data());

    }

  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_POOLING_HPP_INCLUDED
