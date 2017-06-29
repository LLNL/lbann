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
// lbann_layer_local_response_normalization .hpp .cpp - LRN layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED

#include <vector>
#include "lbann/lbann_base.hpp"
#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/lbann_exception.hpp"

namespace lbann {

/// Local Response Normalization layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class local_response_normalization_layer : public regularizer_layer {
 private:

  /// Normalization window width
  int m_window_width;
  /// LRN alpha scaling parameter
  DataType m_lrn_alpha;
  /// LRN beta power parameter
  DataType m_lrn_beta;
  /// LRN k parameter
  DataType m_lrn_k;

#ifdef __LIB_CUDNN
  /// Data tensor descriptor
  cudnnTensorDescriptor_t m_tensor_desc;
  /// Pooling descriptor
  cudnnLRNDescriptor_t m_lrn_desc;
#endif // __LIB_CUDNN

 public:
  local_response_normalization_layer
  (int index,
   lbann_comm *comm,
   int mini_batch_size,
   int window_width,
   DataType lrn_alpha,
   DataType lrn_beta,
   DataType lrn_k,
   cudnn::cudnn_manager *cudnn = NULL)
    : regularizer_layer(index, comm, mini_batch_size),
      m_window_width(window_width), m_lrn_alpha(lrn_alpha), m_lrn_beta(lrn_beta),
      m_lrn_k(lrn_k) {
    // Setup the data distribution
    initialize_distributed_matrices();

  #ifdef __LIB_CUDNN

    // Initialize cuDNN objects
    m_tensor_desc = NULL;
    m_lrn_desc = NULL;

    // Initialize GPU memory if using GPU
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;

      // Get number of GPUs
      const int num_gpus = this->m_cudnn->get_num_gpus();

      // Get number of columns per GPU
      const int num_processes = this->m_comm->get_procs_per_model();
      const int local_mini_batch_size = (mini_batch_size + num_processes - 1) / num_processes;
      this->m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

    }
  #endif // __LIB_CUDNN
  }

  ~local_response_normalization_layer() {
  #ifdef __LIB_CUDNN
    if(this->m_using_gpus) {

      // Destroy cuDNN objects
      if(m_tensor_desc) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_tensor_desc));
      }
      if(m_lrn_desc) {
        CHECK_CUDNN(cudnnDestroyLRNDescriptor(m_lrn_desc));
      }

      // Deallocate GPU memory
      this->m_cudnn->deallocate_on_gpus(this->m_activations_d);
      this->m_cudnn->deallocate_on_gpus(this->m_error_signal_d);
      if(!this->m_prev_layer->using_gpus()) {
        this->m_cudnn->deallocate_on_gpus(this->m_prev_activations_d);
      }
      if(!this->m_next_layer->using_gpus()) {
        this->m_cudnn->deallocate_on_gpus(this->m_prev_error_signal_d);
      }

    }
  #endif // __LIB_CUDNN
  }

  std::string get_name() const { return "local response normalization"; }

  virtual inline void initialize_distributed_matrices() {
    regularizer_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  virtual void setup(const Layer *prev_layer, const Layer *next_layer) {
    Layer::setup(prev_layer, next_layer);

    // Initialize neuron tensor dimensions
    this->m_num_neurons = this->m_num_prev_neurons;
    this->m_num_neuron_dims = this->m_num_prev_neuron_dims;
    this->m_neuron_dims = this->m_prev_neuron_dims;

  #ifdef __LIB_CUDNN
    // Setup cuDNN objects
    if(this->m_using_gpus) {
      setup_gpu();
    }
  #endif // __LIB_CUDNN

    // Initialize activations matrix
    El::Zeros(*this->m_activations, this->m_num_neurons, this->m_mini_batch_size);

  }

 private:
  /// Initialize GPU objects
  void setup_gpu() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
  #else

    // Initialize descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_tensor_desc));
    CHECK_CUDNN(cudnnCreateLRNDescriptor(&m_lrn_desc));

    // Set input tensor descriptor
    std::vector<int> dims = this->m_prev_neuron_dims;
    dims.insert(dims.begin(), this->m_mini_batch_size_per_gpu);
    std::vector<int> strides(this->m_num_prev_neuron_dims+1);
    strides[strides.size()-1]  = 1;
    for(int i=strides.size()-2; i>=0; --i) {
      strides[i] = strides[i+1] * dims[i+1];
    }
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_tensor_desc,
                                           this->m_cudnn->get_cudnn_data_type(),
                                           dims.size(),
                                           dims.data(),
                                           strides.data()));

    // Set local response normalization descriptor
    CHECK_CUDNN(cudnnSetLRNDescriptor(m_lrn_desc,
                                      (unsigned int) m_window_width,
                                      (double) m_lrn_alpha,
                                      (double) m_lrn_beta,
                                      (double) m_lrn_k));

    // Allocate GPU memory
    this->m_cudnn->allocate_on_gpus(this->m_activations_d,
                                    this->m_num_neurons,
                                    this->m_mini_batch_size_per_gpu);
    this->m_cudnn->allocate_on_gpus(this->m_error_signal_d,
                                    this->m_num_prev_neurons,
                                    this->m_mini_batch_size_per_gpu);
    if(!this->m_prev_layer->using_gpus()) {
      this->m_cudnn->allocate_on_gpus(this->m_prev_activations_d,
                                      this->m_num_prev_neurons,
                                      this->m_mini_batch_size_per_gpu);
    }
    if(!this->m_next_layer->using_gpus()) {
      this->m_cudnn->allocate_on_gpus(this->m_prev_error_signal_d,
                                      this->m_num_neurons,
                                      this->m_mini_batch_size_per_gpu);
    }

  #endif // #ifndef __LIB_CUDNN
  }

 protected:
  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_cudnn();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      bp_compute_cudnn();
    } else {
      bp_compute_cpu();
    }
  }

 private:
  /// GPU implementation of forward propagation
  void fp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Perform local response normalization with each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnLRNCrossChannelForward(this->m_cudnn->get_handle(i),
                                              m_lrn_desc,
                                              CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                              &one,
                                              m_tensor_desc,
                                              this->m_prev_activations_d[i],
                                              &zero,
                                              m_tensor_desc,
                                              this->m_activations_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

  /// GPU implementation of backward propagation
  void bp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
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
      CHECK_CUDNN(cudnnLRNCrossChannelBackward(this->m_cudnn->get_handle(i),
                                               m_lrn_desc,
                                               CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                               &one,
                                               m_tensor_desc,
                                               this->m_activations_d[i],
                                               m_tensor_desc,
                                               this->m_prev_error_signal_d[i],
                                               m_tensor_desc,
                                               this->m_prev_activations_d[i],
                                               &zero,
                                               m_tensor_desc,
                                               this->m_error_signal_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

  /// CPU implementation of forward propagation
  void fp_compute_cpu() {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();

    // Input and output entries are divided amongst channels
    const int num_channels = this->m_neuron_dims[0];
    const int num_per_channel = this->m_num_neurons / num_channels;

    ////////////////////////////////////////////////////////////////
    // activations(i) = prev_activations(i) / scale_factor(i) ^ beta
    // scale_factor(i)
    //   = k + alpha / window_width * sum( prev_activations(j) ^ 2 )
    // Note: The sum is over entries in the normalization window.
    ////////////////////////////////////////////////////////////////

    // Iterate through data samples in mini-batch
    #pragma omp parallel for collapse(2)
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {
      // Iterate through positions in sample
      for(int pos = 0; pos < num_per_channel; ++pos) {

        // Initialize normalization window
        int window_start = - m_window_width / 2;
        int window_end = m_window_width / 2;
        DataType window_sum = 0;
        for(int c = std::max(window_start, 0);
            c <= std::min(window_end, num_channels-1);
            ++c) {
          const DataType x
            = prev_activations_local(pos + num_per_channel*c, sample);
          window_sum += x * x;
        }

        // Iterate through channels at current position
        for(int channel = 0; channel < num_channels; ++channel) {
          const int index = pos + num_per_channel * channel;

          // Apply local response normalization to current entry
          const DataType input_entry = prev_activations_local.Get(index, sample);
          const DataType scale_factor = m_lrn_k + m_lrn_alpha / m_window_width * window_sum;
          const DataType output_entry = input_entry * Pow(scale_factor, -m_lrn_beta);
          activations_local(index, sample) = output_entry;

          // Shift normalization window by one entry
          if(window_start >= 0) {
            const int i = pos + num_per_channel*window_start;
            const DataType x = prev_activations_local(i, sample);
            window_sum -= x * x;
          }
          ++window_start;
          ++window_end;
          if(window_end < num_channels) {
            const int i = pos + num_per_channel*window_end;
            const DataType x = prev_activations_local(i, sample);
            window_sum += x * x;
          }

        }

      }

    }

  }

  /// CPU implementation of backward propagation
  void bp_compute_cpu() {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& activations_local = this->m_activations_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal_v->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();

    // Initialize error signal to zero
    Zero(error_signal_local);

    // Input and output entries are divided amongst channels
    const int num_channels = this->m_neuron_dims[0];
    const int num_per_channel = this->m_num_neurons / num_channels;

    ////////////////////////////////////////////////////////////////
    // error_signal(i)
    //   = prev_error_signal(i) / scale_factor(i) ^ beta
    //     - 2 * alpha * beta / window_width * prev_activations(i)
    //       * sum( prev_error_signal(j) * activations(j)
    //              / scale_factor(j) )
    // Note: See comments in fp_linearity_cpu for a definition of
    //   scale_factor. The sum is over entries in the normalization
    //   window.
    ////////////////////////////////////////////////////////////////

    // Iterate through data samples in mini-batch
    #pragma omp parallel for collapse(2)
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {
      // Iterate through positions in sample
      for(int pos = 0; pos < num_per_channel; ++pos) {

        // Initialize normalization window
        int window_start = - m_window_width / 2;
        int window_end = m_window_width / 2;
        DataType window_sum = 0;
        for(int c = std::max(window_start, 0);
            c <= std::min(window_end, num_channels-1);
            ++c) {
          const DataType x
            = prev_activations_local.Get(pos + num_per_channel*c, sample);
          window_sum += x * x;
        }

        // Iterate through channels at current position
        DataType error_signal_update;
        for(int channel = 0; channel < num_channels; ++channel) {
          const int index = pos + num_per_channel * channel;

          // Get data for current entry
          const DataType activations_entry = activations_local.Get(index, sample);
          const DataType prev_error_signal_entry = prev_error_signal_local.Get(index, sample);
          const DataType scale_factor = m_lrn_k + m_lrn_alpha / m_window_width * window_sum;

          // Update current error signal entry
          error_signal_update = prev_error_signal_entry * Pow(scale_factor, -m_lrn_beta);
          error_signal_local.Update(index, sample, error_signal_update);

          // Update error signal entries in normalization window
          for(int c = std::max(window_start, 0);
              c <= std::min(window_end, num_channels-1);
              ++c) {
            const int i = pos + num_per_channel * c;
            const DataType prev_activations_entry = prev_activations_local.Get(i, sample);
            error_signal_update
              = (-2 * m_lrn_alpha * m_lrn_beta / m_window_width * prev_activations_entry
                 * prev_error_signal_entry * activations_entry / scale_factor);
            error_signal_local.Update(i, sample, error_signal_update);
          }

          // Shift normalization window by one entry
          if(window_start >= 0) {
            const int i = pos + num_per_channel*window_start;
            const DataType x = prev_activations_local.Get(i, sample);
            window_sum -= x * x;
          }
          ++window_start;
          ++window_end;
          if(window_end < num_channels) {
            const int i = pos + num_per_channel*window_end;
            const DataType x = prev_activations_local.Get(i, sample);
            window_sum += x * x;
          }

        }

      }

    }

  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_POOLING_HPP_INCLUDED
