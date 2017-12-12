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
// base_convolution.hpp - Base class for convolution and deconvolution layers
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED

#include <vector>
#include <omp.h>
#include "lbann/layers/learning/learning.hpp"
#include "lbann/base.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/fan_in_fan_out_initializers.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {

/** Base convolution layer.
 *  Parent class for convolution and deconvolution layers.
 */
class base_convolution_layer : public learning_layer {

 protected:

  /** Convolution kernel dimensions. */
  std::vector<int> m_kernel_dims;
  /** Convolution padding. */
  std::vector<int> m_conv_pads;
  /** Convolution strides. */
  std::vector<int> m_conv_strides;

  /** Size of convolutional kernel. */
  int m_kernel_size;
  
  /** Scaling factor for bias term.
   *  If the scaling factor is zero, bias is not applied.
   */
  DataType m_bias_scaling_factor;

  /** View into convolutional kernel weights. */
  AbsDistMat *m_kernel_weights_v;
  /** View into bias weights. */
  AbsDistMat *m_bias_weights_v;

  /** Convolutional kernel weights gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the convolutional kernel weights.
   */
  AbsDistMat *m_kernel_weights_gradient;
  /** Bias weights gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the bias weights.
   */
  AbsDistMat *m_bias_weights_gradient;

#ifdef __LIB_CUDNN

  /** Bias tensor cuDNN descriptor. */
  cudnnTensorDescriptor_t m_bias_cudnn_desc;
  /** Convolution kernel cuDNN descriptor. */
  cudnnFilterDescriptor_t m_kernel_cudnn_desc;
  /** Convolution cuDNN descriptor. */
  cudnnConvolutionDescriptor_t m_convolution_cudnn_desc;

  /** GPU memory for convolution kernel gradient. */
  std::vector<DataType*> m_kernel_weights_gradient_d;
  /** GPU memory for convolution bias gradient. */
  std::vector<DataType*> m_bias_weights_gradient_d;

#endif // __LIB_CUDNN

  public:

  base_convolution_layer(lbann_comm *comm,
                         int num_data_dims,
                         int num_output_channels,
                         std::vector<int> conv_dims,
                         std::vector<int> conv_pads,
                         std::vector<int> conv_strides,
                         bool has_bias,
                         cudnn::cudnn_manager *cudnn)
    : learning_layer(comm) {

    if (conv_dims.size() == 1) {
      conv_dims.resize(num_data_dims, conv_dims[0]);
    } 
    if (conv_pads.size() == 1) {
      conv_pads.resize(num_data_dims, conv_pads[0]);
    }
    if (conv_strides.size() == 1) {
      conv_strides.resize(num_data_dims, conv_strides[0]);
    }
    if ((int)conv_dims.size() != num_data_dims
        && (int)conv_pads.size() != num_data_dims
        && (int)conv_strides.size() != num_data_dims) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid number of convolution parameters " 
          << "(expected " << num_data_dims << " parameters, "
          << "conv_dims has " << conv_dims.size() << ", "
          << "conv_pads has " << conv_pads.size() << ", "
          << "conv_strides has " << conv_strides.size() << ")";
      throw lbann_exception(err.str());
    }

    // Initialize convolution parameters
    m_kernel_dims = conv_dims;
    m_kernel_dims.insert(m_kernel_dims.begin(), num_output_channels);
    m_conv_pads = conv_pads;
    m_conv_strides = conv_strides;

    // Initialize bias
    m_bias_scaling_factor = has_bias ? DataType(1) : DataType(0);

  #ifdef __LIB_CUDNN

    // Initialize cuDNN objects
    this->m_cudnn = cudnn;
    m_bias_cudnn_desc = nullptr;
    m_kernel_cudnn_desc = nullptr;
    m_convolution_cudnn_desc = nullptr;

  #endif // #ifdef __LIB_CUDNN

  }

  base_convolution_layer(const base_convolution_layer& other) :
    learning_layer(other),
    m_kernel_dims(other.m_kernel_dims),
    m_conv_pads(other.m_conv_pads),
    m_conv_strides(other.m_conv_strides),
    m_kernel_size(other.m_kernel_size),
    m_bias_scaling_factor(other.m_bias_scaling_factor) {

    // Copy matrices
    m_kernel_weights_v = other.m_kernel_weights_v->Copy();
    m_bias_weights_v = other.m_bias_weights_v->Copy();
    m_kernel_weights_gradient = other.m_kernel_weights_gradient->Copy();
    m_bias_weights_gradient = other.m_bias_weights_gradient->Copy();

  #ifdef __LIB_CUDNN

    // Copy cuDNN objects
    m_bias_cudnn_desc = nullptr;
    m_kernel_cudnn_desc = nullptr;
    m_convolution_cudnn_desc = nullptr;
    cudnn::copy_tensor_cudnn_desc(other.m_bias_cudnn_desc,
                                  m_bias_cudnn_desc);
    cudnn::copy_kernel_cudnn_desc(other.m_kernel_cudnn_desc,
                                  m_kernel_cudnn_desc);
    cudnn::copy_convolution_cudnn_desc(other.m_convolution_cudnn_desc,
                                       m_convolution_cudnn_desc);

    // Copy GPU data
    m_kernel_weights_gradient_d = m_cudnn->copy(other.m_kernel_weights_gradient_d,
                                                this->m_kernel_weights_gradient->Height(),
                                                this->m_kernel_weights_gradient->Width());
    m_bias_weights_gradient_d = m_cudnn->copy(other.m_bias_weights_gradient_d,
                                              this->m_bias_weights_gradient->Height(),
                                              this->m_bias_weights_gradient->Width());

  #endif // __LIB_CUDNN

  }

  base_convolution_layer& operator=(const base_convolution_layer& other) {
    learning_layer::operator=(other);
    m_kernel_dims = other.m_kernel_dims;
    m_conv_pads = other.m_conv_pads;
    m_conv_strides = other.m_conv_strides;
    m_kernel_size = other.m_kernel_size;
    m_bias_scaling_factor = other.m_bias_scaling_factor;

    // Copy matrices
  #define COPY_MATRIX(src, dst)                 \
    do {                                        \
      if(src != nullptr && dst != nullptr) {    \
        El::Copy(*src, *dst);                   \
      }                                         \
      if(src != nullptr && dst == nullptr) {    \
        dst = src->Copy();                      \
      }                                         \
      if(src == nullptr && dst != nullptr) {    \
        delete dst;                             \
        dst = nullptr;                          \
      }                                         \
    } while(false)
    COPY_MATRIX(other.m_kernel_weights_v, m_kernel_weights_v);
    COPY_MATRIX(other.m_bias_weights_v, m_bias_weights_v);
    COPY_MATRIX(other.m_kernel_weights_gradient, m_kernel_weights_gradient);
    COPY_MATRIX(other.m_bias_weights_gradient, m_bias_weights_gradient);
  #undef COPY_MATRIX

  #ifdef __LIB_CUDNN

    // Copy cuDNN objects
    cudnn::copy_tensor_cudnn_desc(other.m_bias_cudnn_desc,
                                  m_bias_cudnn_desc);
    cudnn::copy_kernel_cudnn_desc(other.m_kernel_cudnn_desc,
                                  m_kernel_cudnn_desc);
    cudnn::copy_convolution_cudnn_desc(other.m_convolution_cudnn_desc,
                                       m_convolution_cudnn_desc);

    // Copy GPU data
    if (m_cudnn != nullptr) {
      m_cudnn->deallocate_on_gpus(m_kernel_weights_gradient_d);
      m_cudnn->deallocate_on_gpus(m_bias_weights_gradient_d);
      m_kernel_weights_gradient_d = m_cudnn->copy(other.m_kernel_weights_gradient_d,
                                                  this->m_kernel_weights_gradient->Height(),
                                                  this->m_kernel_weights_gradient->Width());
      m_bias_weights_gradient_d = m_cudnn->copy(other.m_bias_weights_gradient_d,
                                                this->m_bias_weights_gradient->Height(),
                                                this->m_bias_weights_gradient->Width());
    }

  #endif // __LIB_CUDNN

    return *this;
  }

  ~base_convolution_layer() override {
  #ifdef __LIB_CUDNN
    // Destroy cuDNN objects
    if(m_bias_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_bias_cudnn_desc));
    }
    if(m_kernel_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyFilterDescriptor(m_kernel_cudnn_desc));
    }
    if(m_convolution_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(m_convolution_cudnn_desc));
    }

    if (this->m_cudnn != nullptr) {
      // Deallocate GPU memory
      this->m_cudnn->deallocate_on_gpus(m_kernel_weights_gradient_d);
      this->m_cudnn->deallocate_on_gpus(m_bias_weights_gradient_d);
    }

  #endif // __LIB_CUDNN

    // Delete matrix views
    delete m_kernel_weights_v;
    delete m_bias_weights_v;
    delete m_kernel_weights_gradient;
    delete m_bias_weights_gradient;

  }


  template<data_layout T_layout> void initialize_distributed_matrices() {
    learning_layer::initialize_distributed_matrices<T_layout>();
    m_kernel_weights_gradient = new StarMat(this->m_comm->get_model_grid());
    m_bias_weights_gradient = new StarMat(this->m_comm->get_model_grid());
    m_kernel_weights_v = new StarMat(this->m_comm->get_model_grid());
    m_bias_weights_v = new StarMat(this->m_comm->get_model_grid());
  }

  /** Setup layer data.
   *  The kernel weights are setup in the convolution and
   *  deconvolution classes. */
  void setup_data() override {
    learning_layer::setup_data();

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 2) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to setup " << m_name << " with an invalid number of weights";
      throw lbann_exception(err.str());
    }
    this->m_weights.resize(2, nullptr);
    if (this->m_weights[0] == nullptr) {
      this->m_weights[0] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[0]->set_name(this->m_name + "_kernel_weights");
      this->m_weights[0]->set_initializer(new he_normal_initializer(this->m_comm));
      this->m_weights[0]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[1]->set_name(this->m_name + "_bias_weights");
      this->m_weights[1]->set_initializer(new constant_initializer(this->m_comm, DataType(0)));
      this->m_weights[1]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[1]);
    }

    // Initialize Glorot or He weight initialization
    auto* cast_initializer
      = dynamic_cast<fan_in_fan_out_initializer*>(&this->m_weights[0]->get_initializer());
    if (cast_initializer != nullptr) {
      cast_initializer->set_fan_in(m_kernel_size / this->m_neuron_dims[0]);
      cast_initializer->set_fan_out(m_kernel_size / this->m_prev_neuron_dims[0]);
    }
    
    // Initialize bias
    this->m_weights[1]->setup(this->m_neuron_dims[0], 1,
                              El::STAR, El::STAR);
    El::Zeros(*this->m_bias_weights_gradient,
              this->m_weights[1]->get_height(),
              this->m_weights[1]->get_width());

  }

  /// Initialize GPU objects
  void setup_gpu() override {
    learning_layer::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Set kernel descriptor
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&m_kernel_cudnn_desc));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(m_kernel_cudnn_desc,
                                           cudnn::get_cudnn_data_type(),
                                           CUDNN_TENSOR_NCHW,
                                           m_kernel_dims.size(),
                                           m_kernel_dims.data()));

    // Set convolution descriptor
    // Note: upscales are not supported as of cuDNN v5.1
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&m_convolution_cudnn_desc));
    std::vector<int> conv_upscales(this->m_num_neuron_dims-1, 1);
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(m_convolution_cudnn_desc,
                                                m_conv_pads.size(),
                                                m_conv_pads.data(),
                                                m_conv_strides.data(),
                                                conv_upscales.data(),
                                                CUDNN_CROSS_CORRELATION,
                                                cudnn::get_cudnn_data_type()));

    // Set bias tensor descriptor
    std::vector<int> bias_dims(this->m_num_neuron_dims, 1);
    bias_dims[0] = this->m_neuron_dims[0];
    cudnn::set_tensor_cudnn_desc(m_bias_cudnn_desc, 1, bias_dims);

    // Allocate GPU memory
    this->m_cudnn->allocate_on_gpus(this->m_kernel_weights_gradient_d,
                                    this->m_kernel_weights_gradient->Height(),
                                    this->m_kernel_weights_gradient->Width());
    if(m_bias_scaling_factor != DataType(0)) {
      this->m_cudnn->allocate_on_gpus(this->m_bias_weights_gradient_d,
                                      this->m_bias_weights_gradient->Height(),
                                      this->m_bias_weights_gradient->Width());
    }

  #endif // #ifdef __LIB_CUDNN
  }

 protected:

  void fp_set_std_matrix_view() override {
    learning_layer::fp_set_std_matrix_view();
    if (m_weights.size() < 1 || this->m_weights[0] == nullptr) {
      std::stringstream err;
      err << __FILE__ << ' ' << __LINE__ << " :: "
          << "uninitialized kernel weights";
      throw lbann_exception(err.str());
    } else {
      this->m_weights[0]->get_values_view(*m_kernel_weights_v);
    }
    if (m_bias_scaling_factor != DataType(0)) {
      if (m_weights.size() < 2 || this->m_weights[1] == nullptr) {
        std::stringstream err;
        err << __FILE__ << ' ' << __LINE__ << " :: "
            << "uninitialized bias weights";
        throw lbann_exception(err.str());
      } else {
        this->m_weights[1]->get_values_view(*m_bias_weights_v);
      }
    }
    
  }

  /** Convolution with cuDNN. */
  void apply_convolution_cudnn(bool during_forward_prop) {
  #ifndef __LIB_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Initialize input and output
    cudnnTensorDescriptor_t input_cudnn_desc, output_cudnn_desc;
    std::vector<DataType*> input_d, output_d;
    if(during_forward_prop) {
      input_cudnn_desc = this->m_prev_neurons_cudnn_desc;
      output_cudnn_desc = this->m_neurons_cudnn_desc;
      input_d = this->m_prev_activations_dv;
      output_d = this->m_activations_d;
    }
    else {
      input_cudnn_desc = this->m_neurons_cudnn_desc;
      output_cudnn_desc = this->m_prev_neurons_cudnn_desc;
      input_d = this->m_prev_error_signal_dv;
      output_d = this->m_error_signal_d;
    }

    // Get kernel weights
    std::vector<DataType*> kernel_weights_d = m_weights[0]->get_values_gpu();

    // Perform convolution on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {

      // Get work space
      size_t work_space_size = this->m_cudnn->get_work_space_size(i);
      if(work_space_size == 0) {
        this->m_cudnn->set_maximum_work_space_size(i);
        work_space_size = this->m_cudnn->get_work_space_size(i);
      }
      void *work_space = this->m_cudnn->get_work_space(i);

      // Determine convolution algorithm
      cudnnConvolutionFwdAlgo_t convolution_cudnn_algorithm
        = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(this->m_cudnn->get_handle(i),
                                                      input_cudnn_desc,
                                                      m_kernel_cudnn_desc,
                                                      m_convolution_cudnn_desc,
                                                      output_cudnn_desc,
                                                      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                      work_space_size,
                                                      &convolution_cudnn_algorithm));

      // Apply convolution
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnConvolutionForward(this->m_cudnn->get_handle(i),
                                          &one,
                                          input_cudnn_desc,
                                          input_d[i],
                                          m_kernel_cudnn_desc,
                                          kernel_weights_d[i],
                                          m_convolution_cudnn_desc,
                                          convolution_cudnn_algorithm,
                                          work_space,
                                          work_space_size,
                                          &zero,
                                          output_cudnn_desc,
                                          output_d[i]));

    }

  #endif // #ifndef __LIB_CUDNN
  }

  /** Transposed convolution with cuDNN. */
  void apply_transposed_convolution_cudnn(bool during_forward_prop) {
  #ifndef __LIB_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Initialize input and output
    cudnnTensorDescriptor_t input_cudnn_desc, output_cudnn_desc;
    std::vector<DataType*> input_d, output_d;
    if(during_forward_prop) {
      input_cudnn_desc = this->m_prev_neurons_cudnn_desc;
      output_cudnn_desc = this->m_neurons_cudnn_desc;
      input_d = this->m_prev_activations_dv;
      output_d = this->m_activations_d;
    }
    else {
      input_cudnn_desc = this->m_neurons_cudnn_desc;
      output_cudnn_desc = this->m_prev_neurons_cudnn_desc;
      input_d = this->m_prev_error_signal_dv;
      output_d = this->m_error_signal_d;
    }

    // Get kernel weights
    std::vector<DataType*> kernel_weights_d = m_weights[0]->get_values_gpu();

    // Perform transposed convolution on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {

      // Get work space
      size_t work_space_size = this->m_cudnn->get_work_space_size(i);
      if(work_space_size == 0) {
        this->m_cudnn->set_maximum_work_space_size(i);
        work_space_size = this->m_cudnn->get_work_space_size(i);
      }
      void *work_space = this->m_cudnn->get_work_space(i);

      // Determine transposed convolution algorithm
      cudnnConvolutionBwdDataAlgo_t transposed_convolution_cudnn_algorithm
        = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(this->m_cudnn->get_handle(i),
                                                           m_kernel_cudnn_desc,
                                                           input_cudnn_desc,
                                                           m_convolution_cudnn_desc,
                                                           output_cudnn_desc,
                                                           CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                                           work_space_size,
                                                           &transposed_convolution_cudnn_algorithm));

      // Perform transposed convolution
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnConvolutionBackwardData(this->m_cudnn->get_handle(i),
                                               &one,
                                               m_kernel_cudnn_desc,
                                               kernel_weights_d[i],
                                               input_cudnn_desc,
                                               input_d[i],
                                               m_convolution_cudnn_desc,
                                               transposed_convolution_cudnn_algorithm,
                                               work_space,
                                               work_space_size,
                                               &zero,
                                               output_cudnn_desc,
                                               output_d[i]));

    }

  #endif // #ifndef __LIB_CUDNN
  }

  void apply_bias_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Useful constant
    const DataType one = 1;

    // Return immediately if there is no bias
    if(m_bias_scaling_factor == DataType(0)) return;

    // Get kernel weights
    std::vector<DataType*> bias_weights_d = m_weights[1]->get_values_gpu();

    // Apply bias on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                 &m_bias_scaling_factor,
                                 m_bias_cudnn_desc,
                                 bias_weights_d[i],
                                 &one,
                                 this->m_neurons_cudnn_desc,
                                 this->m_activations_d[i]));
    }

  #endif // __LIB_CUDNN
  }

  void compute_gradients_cudnn(bool using_transposed_convolution) {
  #ifndef __LIB_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType zero = DataType(0);
    const DataType one = DataType(1);
    const int num_gpus = this->m_cudnn->get_num_gpus();
    const int mini_batch_size = this->m_model->get_current_mini_batch_size();

    // Clear unused columns in previous error signal matrix
    this->m_cudnn->clear_unused_columns_on_gpus(this->m_prev_error_signal_dv,
                                                this->m_num_neurons,
                                                mini_batch_size,
                                                this->m_mini_batch_size_per_gpu);

    // Compute bias gradient
    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if(bias_optimizer != nullptr && m_bias_scaling_factor != DataType(0)) {
      for(int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));
        CHECK_CUDNN(cudnnConvolutionBackwardBias(this->m_cudnn->get_handle(i),
                                                 &one,
                                                 this->m_neurons_cudnn_desc,
                                                 this->m_prev_error_signal_dv[i],
                                                 &zero,
                                                 m_bias_cudnn_desc,
                                                 m_bias_weights_gradient_d[i]));
      }
      bias_optimizer->allreduce_and_add_to_gradient_gpu(m_bias_weights_gradient_d,
                                                        m_bias_scaling_factor / mini_batch_size);
    }

    // Compute kernel gradient
    optimizer* kernel_optimizer = m_weights[0]->get_optimizer();
    if(kernel_optimizer != nullptr) {
      for(int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));

        // Get work space
        size_t work_space_size = this->m_cudnn->get_work_space_size(i);
        if(work_space_size == 0) {
          this->m_cudnn->set_maximum_work_space_size(i);
          work_space_size = this->m_cudnn->get_work_space_size(i);
        }
        void *work_space = this->m_cudnn->get_work_space(i);
      
        // Determine algorithm and compute kernel gradient
        cudnnConvolutionBwdFilterAlgo_t kernel_gradient_cudnn_algorithm
          = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        if(using_transposed_convolution) {
          CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->m_cudnn->get_handle(i),
                                                                 this->m_neurons_cudnn_desc,
                                                                 this->m_prev_neurons_cudnn_desc,
                                                                 m_convolution_cudnn_desc,
                                                                 m_kernel_cudnn_desc,
                                                                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                                 work_space_size,
                                                                 &kernel_gradient_cudnn_algorithm));
          CHECK_CUDNN(cudnnConvolutionBackwardFilter(this->m_cudnn->get_handle(i),
                                                     &one,
                                                     this->m_neurons_cudnn_desc,
                                                     this->m_prev_error_signal_dv[i],
                                                     this->m_prev_neurons_cudnn_desc,
                                                     this->m_prev_activations_dv[i],
                                                     m_convolution_cudnn_desc,
                                                     kernel_gradient_cudnn_algorithm,
                                                     work_space,
                                                     work_space_size,
                                                     &zero,
                                                     m_kernel_cudnn_desc,
                                                     m_kernel_weights_gradient_d[i]));
        }
        else {
          CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->m_cudnn->get_handle(i),
                                                                 this->m_prev_neurons_cudnn_desc,
                                                                 this->m_neurons_cudnn_desc,
                                                                 m_convolution_cudnn_desc,
                                                                 m_kernel_cudnn_desc,
                                                                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                                 work_space_size,
                                                                 &kernel_gradient_cudnn_algorithm));
          CHECK_CUDNN(cudnnConvolutionBackwardFilter(this->m_cudnn->get_handle(i),
                                                     &one,
                                                     this->m_prev_neurons_cudnn_desc,
                                                     this->m_prev_activations_dv[i],
                                                     this->m_neurons_cudnn_desc,
                                                     this->m_prev_error_signal_dv[i],
                                                     m_convolution_cudnn_desc,
                                                     kernel_gradient_cudnn_algorithm,
                                                     work_space,
                                                     work_space_size,
                                                     &zero,
                                                     m_kernel_cudnn_desc,
                                                     m_kernel_weights_gradient_d[i]));
        }

      }

      // Add gradient contribution
      kernel_optimizer->allreduce_and_add_to_gradient_gpu(m_kernel_weights_gradient_d,
                                                          one / mini_batch_size);
      
    }

  #endif // __LIB_CUDNN
  }

  /** Convolution with im2col GEMM algorithm. */
  void apply_convolution_im2col(bool during_forward_prop) {

    // Initialize input and output
    AbsDistMat *input, *output;
    std::vector<int> input_dims, output_dims;
    int output_size;
    if(during_forward_prop) {
      input = this->m_prev_activations_v;
      output = this->m_activations_v;
      input_dims = this->m_prev_neuron_dims;
      output_dims = this->m_neuron_dims;
      output_size = this->m_num_neurons;
    }
    else {
      input = this->m_prev_error_signal_v;
      output = this->m_error_signal_v;      
      input_dims = this->m_neuron_dims;
      output_dims = this->m_prev_neuron_dims;
      output_size = this->m_num_prev_neurons;
    }

    // Setup view
    this->m_weights[0]->get_values_view(*m_kernel_weights_v);

    // Get local matrices
    const Mat& input_local = input->LockedMatrix();
    const Mat& kernel_weights_local = m_kernel_weights_v->LockedMatrix();
    Mat& output_local = output->Matrix();

    // Initialize matrices
    const int m = output_size / output_dims[0];
    const int n = output_dims[0];
    const int k = m_kernel_size / output_dims[0];
    Mat im2col_matrix(k, m);
    Mat input_col, output_col;

    // Iterate through input columns
    const El::Int width_local = input_local.Width();
    for(El::Int col = 0; col < width_local; ++col) {

      // Construct im2col matrix from current input column
      El::LockedView(input_col, input_local, El::ALL, El::IR(col));
      im2col(input_col,
             im2col_matrix,
             input_dims[0],
             input_dims.size() - 1,
             &input_dims[1],
             m_conv_pads.data(),
             &m_kernel_dims[2],
             m_conv_strides.data());

      // Apply convolution to current input column
      output_col.Attach(m, n, output_local.Buffer(0, col), m);
      El::Gemm(El::TRANSPOSE, El::NORMAL,
               DataType(1), im2col_matrix, kernel_weights_local,
               DataType(0), output_col);

    }

  }

  /** Transposed convolution with im2col GEMM algorithm. */
  void apply_transposed_convolution_im2col(bool during_forward_prop) {

    // Initialize input and output
    AbsDistMat *input, *output;
    std::vector<int> input_dims, output_dims;
    int input_size;
    if(during_forward_prop) {
      input = this->m_prev_activations_v;
      output = this->m_activations_v;
      input_dims = this->m_prev_neuron_dims;
      input_size = this->m_num_prev_neurons;
      output_dims = this->m_neuron_dims;
    }
    else {
      input = this->m_prev_error_signal_v;
      output = this->m_error_signal_v;
      input_dims = this->m_neuron_dims;
      input_size = this->m_num_neurons;
      output_dims = this->m_prev_neuron_dims;
    }

    // Setup view
    this->m_weights[0]->get_values_view(*m_kernel_weights_v);

    // Get local matrices
    const Mat& input_local = input->LockedMatrix();
    const Mat& kernel_weights_local = m_kernel_weights_v->LockedMatrix();
    Mat& output_local = output->Matrix();

    // Initialize matrices
    const int m = m_kernel_size / input_dims[0];
    const int n = input_size / input_dims[0];
    const int k = input_dims[0];
    Mat im2col_matrix(m, n);
    Mat input_col, output_col;

    // Iterate through input columns
    const El::Int width_local = input_local.Width();
    for(El::Int col = 0; col < width_local; ++col) {

      // Apply transposed convolution to current input column
      input_col.LockedAttach(n, k, input_local.LockedBuffer(0, col), n);
      El::Gemm(El::NORMAL, El::TRANSPOSE,
               DataType(1), kernel_weights_local, input_col,
               DataType(0), im2col_matrix);

      // Perform col2im to accumulate contributions from each kernel
      // position
      El::View(output_col, output_local, El::ALL, El::IR(col));
      col2im(im2col_matrix,
             output_col,
             output_dims[0],
             output_dims.size() - 1,
             &output_dims[1],
             m_conv_pads.data(),
             &m_kernel_dims[2],
             m_conv_strides.data());

    }

  }

  void apply_bias_cpu() {

    // Return immediately if there is no bias
    if(m_bias_scaling_factor == DataType(0)) return;

    // Setup view
    this->m_weights[1]->get_values_view(*m_bias_weights_v);

    // Get local matrices
    const Mat& bias_weights_local = this->m_bias_weights_v->LockedMatrix();
    Mat& activations_local = m_activations_v->Matrix();

    // Get output parameters
    const El::Int width_local = activations_local.Width();
    const El::Int num_output_channels = this->m_neuron_dims[0];
    const El::Int num_per_output_channel = this->m_num_neurons / num_output_channels;

    // Apply bias to each output channel
    #pragma omp parallel for
    for(El::Int channel = 0; channel < num_output_channels; ++channel) {
      const El::Int row_start = channel * num_per_output_channel;
      const El::Int row_end = (channel+1) * num_per_output_channel;
      const DataType bias_term
        = m_bias_scaling_factor * bias_weights_local(channel, 0);
      for(El::Int col = 0; col < width_local; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          activations_local(row, col) += bias_term;
        }
      }
    }

  }

  void compute_gradients_im2col(bool using_transposed_convolution) {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal_v->LockedMatrix();
    Mat& kernel_weights_gradient_local = m_kernel_weights_gradient->Matrix();
    Mat& bias_weights_gradient_local = m_bias_weights_gradient->Matrix();

    // Get convolution parameters
    const El::Int width_local = prev_activations_local.Width();
    const int num_input_channels = this->m_prev_neuron_dims[0];
    const int num_output_channels = this->m_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_output_channels;
    const int mini_batch_size = this->m_model->get_current_mini_batch_size();

    // Compute bias gradient
    // Note: Sum is computed with Kahan summation
    optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
    if(m_bias_scaling_factor != DataType(0) && bias_optimizer != nullptr) {
      #pragma omp parallel for
      for(int channel = 0; channel < num_output_channels; ++channel) {
        const El::Int row_start = channel * num_per_output_channel;
        const El::Int row_end = (channel+1) * num_per_output_channel;
        DataType sum = 0;
        DataType correction = 0;
        for(El::Int col = 0; col < width_local; ++col) {
          for(El::Int row = row_start; row < row_end; ++row) {
            DataType term = prev_error_signal_local(row, col);
            term += correction;
            const DataType next_sum = sum + term;
            correction = term - (next_sum - sum);
            sum = next_sum;
          }
        }
        bias_weights_gradient_local(channel, 0) = m_bias_scaling_factor * sum;
      }
      bias_optimizer->allreduce_and_add_to_gradient(*m_bias_weights_gradient,
                                                    DataType(1) / mini_batch_size);
    }

    // Stop early if kernel is not being optimized
    optimizer* kernel_optimizer = this->m_weights[0]->get_optimizer();
    if (kernel_optimizer == nullptr) { return; }

    // Initialize im2col matrix
    const int m = (using_transposed_convolution ?
                   m_kernel_size / num_input_channels :
                   m_kernel_size / num_output_channels);
    const int n = (using_transposed_convolution ?
                   num_input_channels :
                   num_output_channels);
    const int k = (using_transposed_convolution ?
                   this->m_num_prev_neurons / num_input_channels :
                   this->m_num_neurons / num_output_channels);
    Mat im2col_matrix(m, k);

    // Compute kernel gradient contributions from each data sample
    El::Zero(kernel_weights_gradient_local);
    for(El::Int col = 0; col < width_local; ++col) {
      if(using_transposed_convolution) {
        const Mat prev_activations_col(k, n, prev_activations_local.LockedBuffer(0,col), k);
        const Mat prev_error_signal_col
          = El::LockedView(prev_error_signal_local, El::ALL, El::IR(col));
        im2col(prev_error_signal_col,
               im2col_matrix,
               num_output_channels,
               this->m_num_neuron_dims - 1,
               &this->m_neuron_dims[1],
               m_conv_pads.data(),
               &m_kernel_dims[2],
               m_conv_strides.data());
        El::Gemm(El::NORMAL, El::NORMAL,
                 DataType(1), im2col_matrix, prev_activations_col,
                 DataType(1), kernel_weights_gradient_local);
      }
      else {
        const Mat prev_activations_col
          = El::LockedView(prev_activations_local, El::ALL, El::IR(col));
        const Mat prev_error_signal_col(k, n, prev_error_signal_local.LockedBuffer(0,col), k);
        im2col(prev_activations_col,
               im2col_matrix,
               num_input_channels,
               this->m_num_prev_neuron_dims - 1,
               &this->m_prev_neuron_dims[1],
               m_conv_pads.data(),
               &m_kernel_dims[2],
               m_conv_strides.data());
        El::Gemm(El::NORMAL, El::NORMAL,
                 DataType(1), im2col_matrix, prev_error_signal_col,
                 DataType(1), kernel_weights_gradient_local);
      }
    }

    // Scale and accumulate gradients
    kernel_optimizer->allreduce_and_add_to_gradient(*m_kernel_weights_gradient,
                                                    DataType(1) / mini_batch_size);

  }

};
}

#endif // LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED
