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
// convolution .hpp .cpp - Convolution Layer
// 07/06/2016: changing distributed matrices to STAR,VC format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_CONVOLUTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/learning/learning.hpp"
#include "lbann/lbann_base.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/utils/lbann_random.hpp"
#include "lbann/utils/lbann_timer.hpp"
#include "lbann/utils/lbann_im2col.hpp"

namespace lbann {

// Forward declaration.
class lbann_callback_imcomm;

/// Convolution layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class convolution_layer : public learning {
 private:

  friend class lbann_callback_imcomm;

  /// Weight initialization scheme
  const weight_initialization m_weight_initialization;
  /// Convolutional filter dimensions
  std::vector<int> m_conv_dims;
  /// Size of convolutional filters
  int m_conv_size;
  /// Convolution padding
  std::vector<int> m_conv_pads;
  /// Convolution strides
  std::vector<int> m_conv_strides;

#ifdef __LIB_CUDNN

  /// Input tensor descriptor
  cudnnTensorDescriptor_t m_input_desc;
  /// Output tensor descriptor
  cudnnTensorDescriptor_t m_output_desc;
  /// Bias tensor descriptor
  cudnnTensorDescriptor_t m_bias_desc;
  /// Filter descriptor
  cudnnFilterDescriptor_t m_filter_desc;
  /// Convolution descriptor
  cudnnConvolutionDescriptor_t m_convolution_desc;

  /// Forward pass algorithm
  cudnnConvolutionFwdAlgo_t m_forward_algo;
  /// Backward pass filter algorithm
  /** Compute gradient w.r.t. filter. */
  cudnnConvolutionBwdFilterAlgo_t m_backward_filter_algo;
  /// Backward pass data algorithm
  /** Compute gradient w.r.t. data, which is passed to previous layer. */
  cudnnConvolutionBwdDataAlgo_t m_backward_data_algo;

  /// GPU memory for convolution filters and bias
  std::vector<DataType *> m_weights_d;
  /// GPU memory for convolution filters gradient and bias gradient
  std::vector<DataType *> m_weights_gradient_d;

  /// Filter and bias gradients computed on each GPU
  StarMat m_weights_gradient_per_gpu;

#endif // __LIB_CUDNN

  public:

  convolution_layer(int index,
                    lbann_comm *comm,
                    int mini_batch_size,
                    int num_data_dims,
                    int num_output_channels,
                    const int *conv_dims,
                    const int *conv_pads,
                    const int *conv_strides,
                    weight_initialization init,
                    optimizer *opt,
                    cudnn::cudnn_manager *cudnn = NULL)
    : learning(index, comm, mini_batch_size, opt),
      m_weight_initialization(init) {
    // Setup the data distribution
    initialize_distributed_matrices();

    // Initialize convolution parameters
    m_conv_dims.assign(conv_dims, conv_dims+num_data_dims);
    m_conv_pads.assign(conv_pads, conv_pads+num_data_dims);
    m_conv_strides.assign(conv_strides, conv_strides+num_data_dims);

    // Initialize neuron tensor dimensions
    this->m_num_neuron_dims = num_data_dims + 1;
    this->m_neuron_dims.resize(this->m_num_neuron_dims);
    this->m_neuron_dims[0] = num_output_channels;

  #ifdef __LIB_CUDNN
    m_weights_gradient_per_gpu = StarMat(this->m_comm->get_model_grid());
  #endif // #ifdef __LIB_CUDNN

  #ifdef __LIB_CUDNN

    // Initialize cuDNN objects
    m_input_desc = NULL;
    m_output_desc = NULL;
    m_bias_desc = NULL;
    m_filter_desc = NULL;
    m_convolution_desc = NULL;
    m_forward_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    m_backward_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    m_backward_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

    // Set parameters for GPU implementation
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

  ~convolution_layer() {
  #ifdef __LIB_CUDNN
    if(this->m_using_gpus) {

      // Destroy cuDNN objects
      if(m_input_desc) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_input_desc));
      }
      if(m_output_desc) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_output_desc));
      }
      if(m_bias_desc) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_bias_desc));
      }
      if(m_filter_desc) {
        CHECK_CUDNN(cudnnDestroyFilterDescriptor(m_filter_desc));
      }
      if(m_convolution_desc) {
        CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(m_convolution_desc));
      }

      // Deallocate GPU memory
      this->m_cudnn->deallocate_on_gpus(m_weights_d);
      this->m_cudnn->deallocate_on_gpus(this->m_activations_d);
      this->m_cudnn->deallocate_on_gpus(m_weights_gradient_d);
      this->m_cudnn->deallocate_on_gpus(this->m_error_signal_d);
      if(!this->m_prev_layer->using_gpus()) {
        this->m_cudnn->deallocate_on_gpus(this->m_prev_activations_d);
      }
      if(!this->m_next_layer->using_gpus()) {
        this->m_cudnn->deallocate_on_gpus(this->m_prev_error_signal_d);
      }

      // Unpin host memory
      if(this->m_using_gpus) {
        this->m_cudnn->unpin_matrix(*(this->m_weights));
        this->m_cudnn->unpin_matrix(m_weights_gradient_per_gpu);
      }

    }
  #endif // #ifdef __LIB_CUDNN
  }

  std::string get_name() const { return "convolution"; }

  void initialize_distributed_matrices() {
    learning::initialize_distributed_matrices<T_layout>();
  }
  virtual inline data_layout get_data_layout() { return T_layout; }

  void setup(const Layer *prev_layer, const Layer *next_layer) {
    Layer::setup(prev_layer, next_layer);
    
    // Initialize neuron tensor dimensions
    for(int i=0; i<m_num_neuron_dims-1; ++i) {
      const int effective_dim = (this->m_prev_neuron_dims[i+1]
                                 + 2*m_conv_pads[i] - m_conv_dims[i] + 1);
      this->m_neuron_dims[i+1] = ((effective_dim + m_conv_strides[i] - 1)
                                  / m_conv_strides[i]);
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Get size of convolutional filters
    m_conv_size = std::accumulate(m_conv_dims.begin(),
                                  m_conv_dims.end(),
                                  this->m_prev_neuron_dims[0] * this->m_neuron_dims[0],
                                  std::multiplies<int>());

  #ifdef __LIB_CUDNN
    // Setup cuDNN objects
    if(this->m_using_gpus) {
      setup_gpu();
    }
  #endif // #ifdef __LIB_CUDNN

    // Initialize matrices
    El::Zeros(*this->m_weights,
              m_conv_size + this->m_neuron_dims[0],
              1);
    El::Zeros(*this->m_weights_gradient,
              m_conv_size + this->m_neuron_dims[0],
              1);
  #ifdef __LIB_CUDNN
    if(this->m_using_gpus) {
      El::Zeros(m_weights_gradient_per_gpu,
                m_conv_size + this->m_neuron_dims[0],
                this->m_cudnn->get_num_gpus());
    }
  #endif // #ifdef __LIB_CUDNN
    El::Zeros(*this->m_activations, this->m_num_neurons, this->m_mini_batch_size);

  #ifdef __LIB_CUDNN
    // Pin host memory
    if(this->m_using_gpus) {
      this->m_cudnn->pin_matrix(*this->m_weights);
      this->m_cudnn->pin_matrix(m_weights_gradient_per_gpu);
    }
  #endif // #ifdef __LIB_CUDNN

    // Initialize filters
    StarMat filter;
    View(filter, *this->m_weights, IR(0,m_conv_size), ALL);
    const int fan_in = m_conv_size / this->m_neuron_dims[0];
    const int fan_out = m_conv_size / this->m_prev_neuron_dims[0];
    initialize_matrix(filter, this->m_weight_initialization, fan_in, fan_out);

    // Initialize optimizer
    if(this->m_optimizer != NULL) {
      this->m_optimizer->setup(this->m_weights);
    }

  }

  /// Initialize GPU objects
  void setup_gpu() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("convolution_layer: cuDNN not detected");
  #else

    // Get device properties
    cudaDeviceProp device_props;
    CHECK_CUDA(cudaGetDeviceProperties(&device_props, 0));

    // Maximum workspace size
    const size_t work_space_limit = device_props.totalGlobalMem/2;

    // Initialize descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&m_filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&m_convolution_desc));

    // Set input tensor descriptor
    std::vector<int> input_dims = this->m_prev_neuron_dims;
    input_dims.insert(input_dims.begin(),
                      this->m_mini_batch_size_per_gpu);
    std::vector<int> input_strides(input_dims.size());
    input_strides[input_strides.size()-1]  = 1;
    for(int i=input_strides.size()-2; i>=0; --i) {
      input_strides[i] = input_strides[i+1] * input_dims[i+1];
    }
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_input_desc,
                                           this->m_cudnn->get_cudnn_data_type(),
                                           input_dims.size(),
                                           input_dims.data(),
                                           input_strides.data()));

    // Set filter descriptor
    std::vector<int> conv_dims = m_conv_dims;
    conv_dims.insert(conv_dims.begin(), this->m_prev_neuron_dims[0]);
    conv_dims.insert(conv_dims.begin(), this->m_neuron_dims[0]);
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(m_filter_desc,
                                           this->m_cudnn->get_cudnn_data_type(),
                                           CUDNN_TENSOR_NCHW,
                                           conv_dims.size(),
                                           conv_dims.data()));

    // Set convolution descriptor
    // Note: upscales are not supported as of cuDNN v5.1
    std::vector<int> conv_upscales(this->m_num_neuron_dims-1, 1);
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(m_convolution_desc,
                                                m_conv_pads.size(),
                                                m_conv_pads.data(),
                                                m_conv_strides.data(),
                                                conv_upscales.data(),
                                                CUDNN_CONVOLUTION,
                                                this->m_cudnn->get_cudnn_data_type()));

    // Set output tensor descriptor
    std::vector<int> output_dims;
  #ifdef LBANN_DEBUG
    output_dims.resize(this->m_num_neuron_dims+1);
    CHECK_CUDNN(cudnnGetConvolutionNdForwardOutputDim(m_convolution_desc,
                                                      m_input_desc,
                                                      m_filter_desc,
                                                      output_dims.size(),
                                                      output_dims.data()));
    if(output_dims[0] != this->m_mini_batch_size_per_gpu) {
      throw lbann_exception("convolution_layer: invalid output dimensions");
    }
    for(int i=0; i<m_num_neuron_dims; ++i) {
      if(output_dims[i+1] != m_neuron_dims_dims[i]) {
        throw lbann_exception("convolution_layer: invalid output dimensions");
      }
    }
  #else
    output_dims = this->m_neuron_dims;
    output_dims.insert(output_dims.begin(), this->m_mini_batch_size_per_gpu);
  #endif // #ifdef LBANN_DEBUG
    std::vector<int> output_strides(output_dims.size());
    output_strides[output_strides.size()-1]  = 1;
    for(int i=output_strides.size()-2; i>=0; --i) {
      output_strides[i] = output_strides[i+1] * output_dims[i+1];
    }
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_output_desc,
                                           this->m_cudnn->get_cudnn_data_type(),
                                           output_dims.size(),
                                           output_dims.data(),
                                           output_strides.data()));

    // Set bias tensor descriptor
    std::vector<int> bias_dims(this->m_num_prev_neuron_dims+1, 1);
    bias_dims[1] = this->m_neuron_dims[0];
    std::vector<int> bias_strides(this->m_num_prev_neuron_dims+1, 1);
    bias_strides[0] = bias_dims[1];
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_bias_desc,
                                           this->m_cudnn->get_cudnn_data_type(),
                                           bias_dims.size(),
                                           bias_dims.data(),
                                           bias_strides.data()));

    // Choose algorithms
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(this->m_cudnn->get_handle(),
                                                    m_input_desc,
                                                    m_filter_desc,
                                                    m_convolution_desc,
                                                    m_output_desc,
                                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                    work_space_limit,
                                                    &m_forward_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->m_cudnn->get_handle(),
                                                           m_input_desc,
                                                           m_output_desc,
                                                           m_convolution_desc,
                                                           m_filter_desc,
                                                           CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                           work_space_limit,
                                                           &m_backward_filter_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(this->m_cudnn->get_handle(),
                                                         m_filter_desc,
                                                         m_output_desc,
                                                         m_convolution_desc,
                                                         m_input_desc,
                                                         CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                                         work_space_limit,
                                                         &m_backward_data_algo));

    // Initialize work space
    size_t max_work_space = 0;
    size_t required_work_space;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->m_cudnn->get_handle(),
                                                        m_input_desc,
                                                        m_filter_desc,
                                                        m_convolution_desc,
                                                        m_output_desc,
                                                        m_forward_algo,
                                                        &required_work_space));
    max_work_space = Max(max_work_space, required_work_space);
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->m_cudnn->get_handle(),
                                                               m_input_desc,
                                                               m_output_desc,
                                                               m_convolution_desc,
                                                               m_filter_desc,
                                                               m_backward_filter_algo,
                                                               &required_work_space));
    max_work_space = Max(max_work_space, required_work_space);
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(this->m_cudnn->get_handle(),
                                                             m_filter_desc,
                                                             m_output_desc,
                                                             m_convolution_desc,
                                                             m_input_desc,
                                                             m_backward_data_algo,
                                                             &required_work_space));
    max_work_space = Max(max_work_space, required_work_space);
    for(int i=0; i<this->m_cudnn->get_num_gpus(); ++i) {
      if(max_work_space > this->m_cudnn->get_work_space_size(i)) {
        this->m_cudnn->set_work_space_size(i, max_work_space);
      }
    }

    // Allocate GPU memory
    this->m_cudnn->allocate_on_gpus(m_weights_d,
                                    m_conv_size+this->m_neuron_dims[0],
                                    1);
    this->m_cudnn->allocate_on_gpus(m_weights_gradient_d,
                                    m_conv_size+this->m_neuron_dims[0],
                                    1);
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

  #endif // #ifdef __LIB_CUDNN
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

  /// Convolution forward propagation with cuDNN
  void fp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Transfer filters and bias from CPU to GPUs
    this->m_cudnn->broadcast_to_gpus(m_weights_d,
                                     this->m_weights->LockedMatrix());

    // Perform convolution on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnConvolutionForward(this->m_cudnn->get_handle(i),
                                          &one,
                                          m_input_desc,
                                          this->m_prev_activations_d[i],
                                          m_filter_desc,
                                          m_weights_d[i],
                                          m_convolution_desc,
                                          m_forward_algo,
                                          this->m_cudnn->get_work_space(i),
                                          this->m_cudnn->get_work_space_size(i),
                                          &zero,
                                          m_output_desc,
                                          this->m_activations_d[i]));
      CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                 &one,
                                 m_bias_desc,
                                 m_weights_d[i] + m_conv_size,
                                 &one,
                                 m_output_desc,
                                 this->m_activations_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

  /// Convolution backward propagation with cuDNN
  void bp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Clear unused columns
    this->m_cudnn->clear_unused_columns_on_gpus(this->m_prev_error_signal_d,
                                                this->m_num_neurons,
                                                this->m_prev_error_signal_v->LocalWidth(),
                                                this->m_mini_batch_size_per_gpu);

    // Perform back propagation on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnConvolutionBackwardBias(this->m_cudnn->get_handle(i),
                                               &one,
                                               m_output_desc,
                                               this->m_prev_error_signal_d[i],
                                               &zero,
                                               m_bias_desc,
                                               m_weights_gradient_d[i] + m_conv_size));
      CHECK_CUDNN(cudnnConvolutionBackwardFilter(this->m_cudnn->get_handle(i),
                                                 &one,
                                                 m_input_desc,
                                                 this->m_prev_activations_d[i],
                                                 m_output_desc,
                                                 this->m_prev_error_signal_d[i],
                                                 m_convolution_desc,
                                                 m_backward_filter_algo,
                                                 this->m_cudnn->get_work_space(i),
                                                 this->m_cudnn->get_work_space_size(i),
                                                 &zero,
                                                 m_filter_desc,
                                                 m_weights_gradient_d[i]));
      CHECK_CUDNN(cudnnConvolutionBackwardData(this->m_cudnn->get_handle(i),
                                               &one,
                                               m_filter_desc,
                                               m_weights_d[i],
                                               m_output_desc,
                                               this->m_prev_error_signal_d[i],
                                               m_convolution_desc,
                                               m_backward_data_algo,
                                               this->m_cudnn->get_work_space(i),
                                               this->m_cudnn->get_work_space_size(i),
                                               &zero,
                                               m_input_desc,
                                               this->m_error_signal_d[i]));

    }

    // Transfer outputs from GPUs to CPU and reduce
    this->m_cudnn->gather_from_gpus(m_weights_gradient_per_gpu.Matrix(),
                                    m_weights_gradient_d, 1);
    El::Zero(*this->m_weights_gradient);
    this->m_cudnn->synchronize();
    for(int i=0; i<num_gpus; ++i) {
      *this->m_weights_gradient += m_weights_gradient_per_gpu(ALL, IR(i));
    }
    El::AllReduce(*this->m_weights_gradient,
                  this->m_weights_gradient->RedundantComm());
    *this->m_weights_gradient *= DataType(1) / this->get_effective_minibatch_size();

  #endif // #ifndef __LIB_CUDNN
  }

  /// Convolution forward propagation with im2col GEMM algorithm
  void fp_compute_im2col() {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& weights_local = this->m_weights->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();

    // Get filters and bias
    const Mat filter_local = LockedView(weights_local, IR(0,m_conv_size), ALL);
    const Mat bias_local = LockedView(weights_local, IR(m_conv_size,END), ALL);

    // Input, output, and filter entries are divided amongst channels
    const int num_input_channels = this->m_prev_neuron_dims[0];
    const int num_output_channels = this->m_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_output_channels;
    const int current_filter_size = m_conv_size / num_output_channels;

    // Apply bias
    for(int i=0; i<num_output_channels; ++i) {
      Mat activations_channel
        = View(activations_local,
               IR(i*num_per_output_channel, (i+1)*num_per_output_channel),
               ALL);
      Fill(activations_channel, bias_local.Get(i,0));
    }

    // Reshape filters into matrix
    const Mat filter_mat(current_filter_size, num_output_channels,
                         filter_local.LockedBuffer(), current_filter_size);

    // Initialize im2col matrix
    Mat im2col_mat(current_filter_size, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {

      // Construct im2col matrix from input
      const Mat input_mat = LockedView(prev_activations_local, ALL, IR(sample));
      im2col(input_mat,
             im2col_mat,
             num_input_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_conv_pads.data(),
             m_conv_dims.data(),
             m_conv_strides.data());

      // Apply convolution to current data sample
      Mat output_mat(num_per_output_channel, num_output_channels,
                     activations_local.Buffer(0,sample), num_per_output_channel);
      Gemm(TRANSPOSE, NORMAL,
           DataType(1), im2col_mat, filter_mat,
           DataType(1), output_mat);

    }

  }

  /// Convolution backward propagation with im2col GEMM algorithm
  void bp_compute_im2col() {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& weights_local = this->m_weights->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal_v->LockedMatrix();
    Mat& weights_gradient_local = this->m_weights_gradient->Matrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();

    // Get filters and bias
    const Mat filter_local = LockedView(weights_local, IR(0,m_conv_size), ALL);
    Mat filter_gradient_local = View(weights_gradient_local, IR(0,m_conv_size), ALL);
    Mat bias_gradient_local = View(weights_gradient_local, IR(m_conv_size,END), ALL);

    // Initialize weight gradients to zero
    Zero(weights_gradient_local);

    // Input, output, and filter entries are divided amongst channels
    const int num_input_channels = this->m_prev_neuron_dims[0];
    const int num_output_channels = this->m_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_output_channels;
    const int current_filter_size = m_conv_size / num_output_channels;

    // Compute bias gradient
    #pragma omp parallel for
    for(int output_channel = 0;
        output_channel < num_output_channels;
        ++output_channel) {
      DataType& bias_gradient_entry = bias_gradient_local(output_channel, 0);
      for(int col = 0; col < prev_error_signal_local.Width(); ++col) {
        for(int row = output_channel * num_per_output_channel;
            row < (output_channel+1) * num_per_output_channel;
            ++row) {
          bias_gradient_entry += prev_error_signal_local(row, col);
        }
      }
    }

    // Initialize filter and im2col matrices
    const Mat filter_mat(current_filter_size, num_output_channels,
                         filter_local.LockedBuffer(), current_filter_size);
    Mat filter_gradient_mat(current_filter_size, num_output_channels,
                            filter_gradient_local.Buffer(), current_filter_size);
    Mat im2col_mat(current_filter_size, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {

      // Reshape previous error signal into matrix
      const Mat prev_error_signal_mat(num_per_output_channel,
                                      num_output_channels,
                                      prev_error_signal_local.LockedBuffer(0,sample),
                                      num_per_output_channel);

      // Compute gradient w.r.t. input im2col matrix
      Gemm(NORMAL, TRANSPOSE,
           DataType(1), filter_mat, prev_error_signal_mat,
           DataType(0), im2col_mat);

      // Compute error signal (i.e. gradient w.r.t. input)
      Mat output_mat = View(error_signal_local, ALL, IR(sample));
      col2im(im2col_mat,
             output_mat,
             num_input_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_conv_pads.data(),
             m_conv_dims.data(),
             m_conv_strides.data());

      // Construct im2col matrix from input
      const Mat input_mat = LockedView(prev_activations_local,
                                       ALL, IR(sample));
      im2col(input_mat,
             im2col_mat,
             num_input_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_conv_pads.data(),
             m_conv_dims.data(),
             m_conv_strides.data());

      // Compute gradient w.r.t. filter
      Gemm(NORMAL, NORMAL,
           DataType(1), im2col_mat, prev_error_signal_mat,
           DataType(1), filter_gradient_mat);

    }

    // Scale and accumulate gradients
    *this->m_weights_gradient *= DataType(1) / this->get_effective_minibatch_size();
    AllReduce(*this->m_weights_gradient, this->m_weights_gradient->RedundantComm());

  }

 public:

  /// Update convolution filters and biases
  bool update_compute() {
    if(this->m_execution_mode == execution_mode::training) {
      this->l2_regularize();
      this->m_optimizer->update(this->m_weights_gradient);
    }
    return true;
  }

};
}

#endif // LBANN_LAYER_CONVOLUTION_HPP_INCLUDED
