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

#ifndef LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Local Response Normalization layer. */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
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

#ifdef LBANN_HAS_CUDNN
  /// Pooling descriptor
  cudnnLRNDescriptor_t m_lrn_cudnn_desc;
#endif // LBANN_HAS_CUDNN

 public:
  local_response_normalization_layer
  (lbann_comm *comm,
   int window_width,
   DataType lrn_alpha,
   DataType lrn_beta,
   DataType lrn_k,
   cudnn::cudnn_manager *cudnn = nullptr)
    : regularizer_layer(comm),
      m_window_width(window_width), m_lrn_alpha(lrn_alpha), m_lrn_beta(lrn_beta),
      m_lrn_k(lrn_k) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "local_response_normalization only supports DATA_PARALLEL");
  #ifdef LBANN_HAS_CUDNN
    // Initialize cuDNN objects
    m_lrn_cudnn_desc = nullptr;
    if (cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN
  }

  local_response_normalization_layer(const local_response_normalization_layer& other) :
    regularizer_layer(other),
    m_window_width(other.m_window_width),
    m_lrn_alpha(other.m_lrn_alpha),
    m_lrn_beta(other.m_lrn_beta),
    m_lrn_k(other.m_lrn_k) {
  #ifdef LBANN_HAS_CUDNN
    m_lrn_cudnn_desc = nullptr;
    cudnn::copy_lrn_cudnn_desc(other.m_lrn_cudnn_desc, m_lrn_cudnn_desc);
  #endif // LBANN_HAS_CUDNN
  }

  local_response_normalization_layer& operator=(const local_response_normalization_layer& other) {
    regularizer_layer::operator=(other);
    m_window_width = other.m_window_width;
    m_lrn_alpha = other.m_lrn_alpha;
    m_lrn_beta = other.m_lrn_beta;
    m_lrn_k = other.m_lrn_k;
  #ifdef LBANN_HAS_CUDNN
    cudnn::copy_lrn_cudnn_desc(other.m_lrn_cudnn_desc, m_lrn_cudnn_desc);
  #endif // LBANN_HAS_CUDNN
  }

  ~local_response_normalization_layer() override {
  #ifdef LBANN_HAS_CUDNN
    // Destroy cuDNN objects
    if (m_lrn_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyLRNDescriptor(m_lrn_cudnn_desc));
    }
  #endif // LBANN_HAS_CUDNN
  }

  local_response_normalization_layer* copy() const override {
    return new local_response_normalization_layer(*this);
  }

  /// Use LRN rather than local response normalization for better formatting
  std::string get_type() const override { return "LRN"; }

  std::string get_description() const override {
    return " LRN window width: " + std::to_string(m_window_width) + " alpha: " +
      std::to_string(m_lrn_alpha) + " beta: " + std::to_string(m_lrn_beta)
      + " k: " + std::to_string(m_lrn_k)
      + " dataLayout: " + get_data_layout_string(get_data_layout());
  }

  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /// Initialize GPU objects
  void setup_gpu() override {
    regularizer_layer::setup_gpu();
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
  #else
    CHECK_CUDNN(cudnnCreateLRNDescriptor(&m_lrn_cudnn_desc));
    CHECK_CUDNN(cudnnSetLRNDescriptor(m_lrn_cudnn_desc,
                                      (unsigned int) m_window_width,
                                      (double) m_lrn_alpha,
                                      (double) m_lrn_beta,
                                      (double) m_lrn_k));
  #endif // #ifndef LBANN_HAS_CUDNN
  }

  void fp_compute() override {
    if (this->m_using_gpus) {
      fp_compute_cudnn();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if (this->m_using_gpus) {
      bp_compute_cudnn();
    } else {
      bp_compute_cpu();
    }
  }

 private:
  /// GPU implementation of forward propagation
  void fp_compute_cudnn() {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Perform local response normalization with each GPU
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
    CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(),
                               this->m_cudnn->get_stream()));
    CHECK_CUDNN(cudnnLRNCrossChannelForward(this->m_cudnn->get_handle(),
                                            m_lrn_cudnn_desc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &one,
                                            this->m_prev_activations_cudnn_desc,
                                            get_prev_activations().LockedBuffer(),
                                            &zero,
                                            this->m_activations_cudnn_desc,
                                            get_activations().Buffer()));

  #endif // #ifndef LBANN_HAS_CUDNN
  }

  /// GPU implementation of backward propagation
  void bp_compute_cudnn() {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;

    // Perform back propagation on each GPU
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
    CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(),
                               this->m_cudnn->get_stream()));
    CHECK_CUDNN(cudnnLRNCrossChannelBackward(this->m_cudnn->get_handle(),
                                             m_lrn_cudnn_desc,
                                             CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                             &one,
                                             this->m_activations_cudnn_desc,
                                             get_activations().LockedBuffer(),
                                             this->m_prev_error_signals_cudnn_desc,
                                             get_prev_error_signals().LockedBuffer(),
                                             this->m_prev_activations_cudnn_desc,
                                             get_prev_activations().LockedBuffer(),
                                             &one,
                                             this->m_error_signals_cudnn_desc,
                                             get_error_signals().Buffer()));

  #endif // #ifndef LBANN_HAS_CUDNN
  }

  /// CPU implementation of forward propagation
  void fp_compute_cpu() {

    // Local matrices
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();

    // Matrix parameters
    const int local_width = local_input.Width();
    const DataType* input_buffer = local_input.LockedBuffer();
    const int input_ldim = local_input.LDim();
    DataType* output_buffer = local_output.Buffer();
    const int output_ldim = local_output.LDim();

    // Get LRN parameters
    const int num_channels = this->m_neuron_dims[0];
    const int num_per_channel = this->m_num_neurons / num_channels;

    // Check if LRN is using default beta parameter
    const bool default_beta = (std::fabs((m_lrn_beta - 0.75) / 0.75)
                               < 2 * std::numeric_limits<DataType>::epsilon());

    ////////////////////////////////////////////////////////////////
    // activations(i) = prev_activations(i) / scale_factor(i) ^ beta
    // scale_factor(i)
    //   = k + alpha * sum( prev_activations(j) ^ 2 )
    // Note: The sum is over entries in the normalization window.
    ////////////////////////////////////////////////////////////////

    // Iterate through blocks in channels of each data sample
    const int max_block_size = 16;
    #pragma omp parallel for collapse(2)
    for (int sample = 0; sample < local_width; ++sample) {
      for (int block_start = 0;
          block_start < num_per_channel;
          block_start += max_block_size) {
        const int block_size = std::min(max_block_size,
                                        num_per_channel - block_start);
        DataType workspace[max_block_size];

        // Iterate through channels
        for (int channel = 0; channel < num_channels; ++channel) {
          const int window_start = std::max(channel - m_window_width / 2, 0);
          const int window_end = std::min(channel + m_window_width / 2, num_channels - 1);

          // Compute sum of squares in workspace
          std::fill(workspace, workspace + block_size, DataType(0));
          for (int window_pos = window_start; window_pos <= window_end; ++window_pos) {
            for (int block_pos = 0; block_pos < block_size; ++block_pos) {
              const int index = block_start + block_pos + window_pos * num_per_channel;
              const DataType input_entry = input_buffer[index + sample * input_ldim];
              workspace[block_pos] += input_entry * input_entry;
            }
          }

          // Compute 1 / (k + alpha * sum(x^2) ) in workspace
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            workspace[block_pos] = 1 / (m_lrn_k + m_lrn_alpha * workspace[block_pos]);
          }

          // Compute output
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            const int index = block_start + block_pos + channel * num_per_channel;
            const DataType scale_factor = workspace[block_pos];
            const DataType input_entry = input_buffer[index + sample * input_ldim];
            DataType& output_entry = output_buffer[index + sample * output_ldim];
            if (default_beta) { // Special case when beta = 0.75
              output_entry = (input_entry
                              * std::sqrt(scale_factor * std::sqrt(scale_factor)));
            }
            else {
              output_entry = input_entry * std::pow(scale_factor, m_lrn_beta);
            }
          }

        }

      }
    }

  }

  /// CPU implementation of backward propagation
  void bp_compute_cpu() {

    // Get local matrices
    const auto& local_input = get_local_prev_activations();
    const auto& local_output = get_local_activations();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();

    // Get matrix buffers
    const int local_width = local_input.Width();
    const DataType* input_buffer = local_input.LockedBuffer();
    const int input_ldim = local_input.LDim();
    const DataType* output_buffer = local_output.LockedBuffer();
    const int output_ldim = local_output.LDim();
    const DataType* gradient_wrt_output_buffer = local_gradient_wrt_output.LockedBuffer();
    const int gradient_wrt_output_ldim = local_gradient_wrt_output.LDim();
    DataType* gradient_wrt_input_buffer = local_gradient_wrt_input.Buffer();
    const int gradient_wrt_input_ldim = local_gradient_wrt_input.LDim();

    // Get LRN parameters
    const int num_channels = this->m_neuron_dims[0];
    const int num_per_channel = this->m_num_neurons / num_channels;

    // Check if LRN is using default beta parameter
    const bool default_beta = (std::fabs((m_lrn_beta - 0.75) / 0.75)
                               < 2 * std::numeric_limits<DataType>::epsilon());

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

    // Iterate through blocks in channels of each data sample
    const int max_block_size = 16;
    #pragma omp parallel for collapse(2)
    for (int sample = 0; sample < local_width; ++sample) {
      for (int block_start = 0;
          block_start < num_per_channel;
          block_start += max_block_size) {
        const int block_size = std::min(max_block_size,
                                        num_per_channel - block_start);
        DataType workspace[max_block_size];

        // Iterate through channels
        for (int channel = 0; channel < num_channels; ++channel) {
          const int window_start = std::max(channel - m_window_width / 2, 0);
          const int window_end = std::min(channel + m_window_width / 2, num_channels - 1);

          // Compute sum of squares in workspace
          std::fill(workspace, workspace + block_size, DataType(0));
          for (int window_pos = window_start; window_pos <= window_end; ++window_pos) {
            for (int block_pos = 0; block_pos < block_size; ++block_pos) {
              const int index = block_start + block_pos + window_pos * num_per_channel;
              const DataType input_entry = input_buffer[index + sample * input_ldim];
              workspace[block_pos] += input_entry * input_entry;
            }
          }

          // Compute 1 / (k + alpha * sum(x^2) ) in workspace
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            workspace[block_pos] = 1 / (m_lrn_k + m_lrn_alpha * workspace[block_pos]);
          }

          // Compute error signal contribution for current entry
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            const int index = block_start + block_pos + channel * num_per_channel;
            const DataType scale_factor = workspace[block_pos];
            const DataType gradient_wrt_output_entry
              = gradient_wrt_output_buffer[index + sample * gradient_wrt_output_ldim];
            DataType& gradient_wrt_input_entry
              = gradient_wrt_input_buffer[index + sample * gradient_wrt_input_ldim];
            if (default_beta) { // Special case when beta = 0.75
              gradient_wrt_input_entry
                += gradient_wrt_output_entry * std::sqrt(scale_factor * std::sqrt(scale_factor));
            }
            else {
              gradient_wrt_input_entry
                += gradient_wrt_output_entry * std::pow(scale_factor, m_lrn_beta);
            }
          }

          // Compute y * dy / (k + alpha * sum(x^2) ) in workspace
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            const int index = block_start + block_pos + channel * num_per_channel;
            const DataType output_entry = output_buffer[index + sample * output_ldim];
            const DataType gradient_wrt_output_entry
              = gradient_wrt_output_buffer[index + sample * gradient_wrt_output_ldim];
            workspace[block_pos] = (-2 * m_lrn_alpha * m_lrn_beta * workspace[block_pos]
                                    * output_entry * gradient_wrt_output_entry);
          }

          // Compute error signal contribution for entries in window
          for (int window_pos = window_start; window_pos <= window_end; ++window_pos) {
            for (int block_pos = 0; block_pos < block_size; ++block_pos) {
              const int index = block_start + block_pos + window_pos * num_per_channel;
              const DataType input_entry = input_buffer[index + sample * input_ldim];
              gradient_wrt_input_buffer[index + sample * gradient_wrt_input_ldim]
                += workspace[block_pos] * input_entry;
            }
          }

        }

      }
    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED
