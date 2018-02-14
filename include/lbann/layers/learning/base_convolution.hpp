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

#ifndef LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED

#include <vector>
#include <omp.h>
#include "lbann/layers/learning/learning.hpp"
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
  /** Size of convolutional kernel. */
  int m_kernel_size;
  /** Convolution padding. */
  std::vector<int> m_pads;
  /** Convolution strides. */
  std::vector<int> m_strides;


  /** Scaling factor for bias term.
   *  If the scaling factor is zero, bias is not applied.
   */
  DataType m_bias_scaling_factor;

  /** Convolutional kernel gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the convolutional kernel weights.
   */
  StarMat m_kernel_gradient;
  /** Bias gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the bias weights.
   */
  StarMat m_bias_gradient;

#ifdef LBANN_HAS_CUDNN

  /** Bias tensor cuDNN descriptor. */
  cudnnTensorDescriptor_t m_bias_cudnn_desc;
  /** Convolution kernel cuDNN descriptor. */
  cudnnFilterDescriptor_t m_kernel_cudnn_desc;
  /** Convolution cuDNN descriptor. */
  cudnnConvolutionDescriptor_t m_convolution_cudnn_desc;

  /** GPU memory for linearity gradient. */
  cudnn::matrix m_kernel_gradient_d;
  /** GPU memory for bias gradient. */
  cudnn::matrix m_bias_gradient_d;

#endif // LBANN_HAS_CUDNN

  public:

  base_convolution_layer(lbann_comm *comm,
                         int num_data_dims,
                         int num_output_channels,
                         const std::vector<int> conv_dims,
                         const std::vector<int> pads,
                         const std::vector<int> strides,
                         bool has_bias,
                         cudnn::cudnn_manager *cudnn)
    : learning_layer(comm),
      m_kernel_dims(conv_dims),
      m_kernel_size(0),
      m_pads(pads),
      m_strides(strides),
      m_bias_scaling_factor(has_bias ? DataType(1) : DataType(0)),
      m_kernel_gradient(this->m_comm->get_model_grid()),
      m_bias_gradient(this->m_comm->get_model_grid()) {

    // Check dimensions of convolution parameters
    if ((int) m_kernel_dims.size() != num_data_dims
        || (int) m_pads.size() != num_data_dims
        || (int) m_strides.size() != num_data_dims) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid number of convolution parameters "
          << "(expected " << num_data_dims << " parameters, "
          << "conv_dims has " << m_kernel_dims.size() << ", "
          << "pads has " << m_pads.size() << ", "
          << "strides has " << m_strides.size() << ")";
      throw lbann_exception(err.str());
    }

    // Record number of output channels
    m_kernel_dims.insert(m_kernel_dims.begin(), num_output_channels);

  #ifdef LBANN_HAS_CUDNN
    // Initialize cuDNN objects
    this->m_cudnn = cudnn;
    m_bias_cudnn_desc = nullptr;
    m_kernel_cudnn_desc = nullptr;
    m_convolution_cudnn_desc = nullptr;
  #endif // LBANN_HAS_CUDNN

  }

  base_convolution_layer(const base_convolution_layer& other)
    : learning_layer(other),
      m_kernel_dims(other.m_kernel_dims),
      m_kernel_size(other.m_kernel_size),
      m_pads(other.m_pads),
      m_strides(other.m_strides),
      m_bias_scaling_factor(other.m_bias_scaling_factor),
      m_kernel_gradient(other.m_kernel_gradient),
      m_bias_gradient(other.m_bias_gradient) {

  #ifdef LBANN_HAS_CUDNN

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
    m_kernel_gradient_d = other.m_kernel_gradient_d;
    m_bias_gradient_d = other.m_bias_gradient_d;

  #endif // LBANN_HAS_CUDNN

  }

  base_convolution_layer& operator=(const base_convolution_layer& other) {
    learning_layer::operator=(other);
    m_kernel_dims = other.m_kernel_dims;
    m_kernel_size = other.m_kernel_size;
    m_pads = other.m_pads;
    m_strides = other.m_strides;
    m_bias_scaling_factor = other.m_bias_scaling_factor;
    m_kernel_gradient = other.m_kernel_gradient;
    m_bias_gradient = other.m_bias_gradient;

  #ifdef LBANN_HAS_CUDNN

    // Copy cuDNN objects
    cudnn::copy_tensor_cudnn_desc(other.m_bias_cudnn_desc,
                                  m_bias_cudnn_desc);
    cudnn::copy_kernel_cudnn_desc(other.m_kernel_cudnn_desc,
                                  m_kernel_cudnn_desc);
    cudnn::copy_convolution_cudnn_desc(other.m_convolution_cudnn_desc,
                                       m_convolution_cudnn_desc);

    // Copy GPU data
    m_kernel_gradient_d = other.m_kernel_gradient_d;
    m_bias_gradient_d = other.m_bias_gradient_d;

  #endif // LBANN_HAS_CUDNN

    return *this;
  }

  ~base_convolution_layer() override {
  #ifdef LBANN_HAS_CUDNN
    // Destroy cuDNN objects
    if (m_bias_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_bias_cudnn_desc));
    }
    if (m_kernel_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyFilterDescriptor(m_kernel_cudnn_desc));
    }
    if (m_convolution_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(m_convolution_cudnn_desc));
    }
  #endif // LBANN_HAS_CUDNN
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
      this->m_weights[0]->set_name(this->m_name + "_kernel");
      this->m_weights[0]->set_initializer(new he_normal_initializer(this->m_comm));
      this->m_weights[0]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[1]->set_name(this->m_name + "_bias");
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
    this->m_weights[1]->setup(this->m_neuron_dims[0]);
    El::Zeros(m_bias_gradient,
              this->m_weights[1]->get_matrix_height(),
              this->m_weights[1]->get_matrix_width());

  }

  /// Initialize GPU objects
  void setup_gpu() override {
    learning_layer::setup_gpu();
  #ifndef LBANN_HAS_CUDNN
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
    std::vector<int> upscales(this->m_num_neuron_dims-1, 1);
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(m_convolution_cudnn_desc,
                                                m_pads.size(),
                                                m_pads.data(),
                                                m_strides.data(),
                                                upscales.data(),
                                                CUDNN_CROSS_CORRELATION,
                                                cudnn::get_cudnn_data_type()));

    // Set bias tensor descriptor
    std::vector<int> bias_dims(this->m_num_neuron_dims, 1);
    bias_dims[0] = this->m_neuron_dims[0];
    cudnn::set_tensor_cudnn_desc(m_bias_cudnn_desc, 1, bias_dims);

    // Allocate GPU memory
    m_kernel_gradient_d = cudnn::matrix(m_cudnn,
                                        m_kernel_gradient.Height(),
                                        m_kernel_gradient.Width());
    if (m_bias_scaling_factor != DataType(0)) {
      m_bias_gradient_d = cudnn::matrix(m_cudnn,
                                        m_bias_gradient.Height(),
                                        m_bias_gradient.Width());
    }

  #endif // LBANN_HAS_CUDNN
  }

  virtual void check_setup() override {
    learning_layer::check_setup();
    std::stringstream err;

    // Check that kernel and bias weights are both initialized
    if (this->m_weights.size() != 2
        || this->m_weights[0] == nullptr
        || this->m_weights[1] == nullptr) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid weights setup in layer " << m_name;
      throw lbann_exception(err.str());
    }

    // Check that kernel data is contiguous
    const auto& kernel = this->m_weights[0]->get_values();
    if (kernel.LocalWidth() > 1
        && kernel.LDim() != kernel.LocalHeight()) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "kernel data in layer " << m_name << " "
          << "is not contiguous";
      throw lbann_exception(err.str());
    }

    // Check that kernel gradient data is contiguous
    if (m_kernel_gradient.LocalWidth() > 1
        && m_kernel_gradient.LDim() != m_kernel_gradient.LocalHeight()) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "kernel gradient data in layer " << m_name << " "
          << "is not contiguous";
      throw lbann_exception(err.str());
    }

  }

 protected:

  /** Convolution with cuDNN. */
  void apply_convolution_cudnn(bool during_forward_prop) {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;

    // GPU data
    const auto& kernel_d = m_weights[0]->get_values_gpu();
    const auto& input_d = (during_forward_prop ?
                           m_prev_activations_d[0] :
                           m_prev_error_signals_d[0]);
    auto& output_d = (during_forward_prop ?
                      m_activations_d[0] :
                      m_error_signals_d[0]);
    auto&& work_spaces_d = m_cudnn->get_work_spaces();

    // Convolution parameters
    DataType mixing_factor;
    cudnnTensorDescriptor_t input_cudnn_desc, output_cudnn_desc;
    if (during_forward_prop) {
      mixing_factor = DataType(0);
      input_cudnn_desc = this->m_prev_activations_cudnn_desc;
      output_cudnn_desc = this->m_activations_cudnn_desc;
    }
    else {
      mixing_factor = DataType(1);
      input_cudnn_desc = this->m_prev_error_signals_cudnn_desc;
      output_cudnn_desc = this->m_error_signals_cudnn_desc;
    }

    // Perform convolution on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for (int i = 0; i < num_gpus; ++i) {

      // Determine convolution algorithm
      const size_t work_space_size = this->m_cudnn->get_work_space_size(i);
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
                                          input_d.get_locked_data(i),
                                          m_kernel_cudnn_desc,
                                          kernel_d[i],
                                          m_convolution_cudnn_desc,
                                          convolution_cudnn_algorithm,
                                          work_spaces_d[i],
                                          work_space_size,
                                          &mixing_factor,
                                          output_cudnn_desc,
                                          output_d.get_data(i)));

    }

  #endif // LBANN_HAS_CUDNN
  }

  /** Transposed convolution with cuDNN. */
  void apply_transposed_convolution_cudnn(bool during_forward_prop) {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;

    // GPU data
    const auto& kernel_d = m_weights[0]->get_values_gpu();
    const auto& input_d = (during_forward_prop ?
                           m_prev_activations_d[0] :
                           m_prev_error_signals_d[0]);
    auto& output_d = (during_forward_prop ?
                      m_activations_d[0] :
                      m_error_signals_d[0]);
    auto&& work_spaces_d = m_cudnn->get_work_spaces();

    // Convolution transpose parameters
    DataType mixing_factor;
    cudnnTensorDescriptor_t input_cudnn_desc, output_cudnn_desc;
    if (during_forward_prop) {
      mixing_factor = DataType(0);
      input_cudnn_desc = this->m_prev_activations_cudnn_desc;
      output_cudnn_desc = this->m_activations_cudnn_desc;
    }
    else {
      mixing_factor = DataType(1);
      input_cudnn_desc = this->m_prev_error_signals_cudnn_desc;
      output_cudnn_desc = this->m_error_signals_cudnn_desc;
    }

    // Perform transposed convolution on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for (int i = 0; i < num_gpus; ++i) {

      // Determine transposed convolution algorithm
      const size_t work_space_size = this->m_cudnn->get_work_space_size(i);
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
                                               kernel_d[i],
                                               input_cudnn_desc,
                                               input_d.get_locked_data(i),
                                               m_convolution_cudnn_desc,
                                               transposed_convolution_cudnn_algorithm,
                                               work_spaces_d[i],
                                               work_space_size,
                                               &mixing_factor,
                                               output_cudnn_desc,
                                               output_d.get_data(i)));

    }

  #endif // LBANN_HAS_CUDNN
  }

  void apply_bias_cudnn() {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else
    if (m_bias_scaling_factor != DataType(0)) {
      const DataType one = 1;
      const auto& bias_weights_d = m_weights[1]->get_values_gpu();
      auto& output_d = this->m_activations_d[0];
      const int num_gpus = this->m_cudnn->get_num_gpus();
      for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));
        CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                   &m_bias_scaling_factor,
                                   m_bias_cudnn_desc,
                                   bias_weights_d[i],
                                   &one,
                                   this->m_activations_cudnn_desc,
                                   output_d.get_data(i)));
      }
    }
  #endif // LBANN_HAS_CUDNN
  }

  void compute_gradients_cudnn(bool using_transposed_convolution) {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType zero = DataType(0);
    const DataType one = DataType(1);
    const int num_gpus = this->m_cudnn->get_num_gpus();
    const int effective_mini_batch_size = this->m_model->get_effective_mini_batch_size();

    const auto& input_d = this->m_prev_activations_d[0];
    const auto& gradient_wrt_output_d = this->m_prev_error_signals_d[0];

    // Compute bias gradient
    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr && m_bias_scaling_factor != DataType(0)) {
      for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));
        CHECK_CUDNN(cudnnConvolutionBackwardBias(this->m_cudnn->get_handle(i),
                                                 &one,
                                                 this->m_prev_error_signals_cudnn_desc,
                                                 gradient_wrt_output_d.get_locked_data(i),
                                                 &zero,
                                                 m_bias_cudnn_desc,
                                                 m_bias_gradient_d.get_data(i)));
      }
      bias_optimizer->stage_gradient_for_accumulation_gpu(
        m_bias_gradient_d.get_locked_data(),
        m_bias_scaling_factor / effective_mini_batch_size);
    }

    // Compute kernel gradient
    optimizer* kernel_optimizer = m_weights[0]->get_optimizer();
    if (kernel_optimizer != nullptr) {
      auto&& work_spaces_d = this->m_cudnn->get_work_spaces();
      for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));

        // Determine algorithm and compute kernel gradient
        const size_t work_space_size = this->m_cudnn->get_work_space_size(i);
        cudnnConvolutionBwdFilterAlgo_t kernel_gradient_cudnn_algorithm
          = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        if (using_transposed_convolution) {
          CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->m_cudnn->get_handle(i),
                                                                 this->m_prev_error_signals_cudnn_desc,
                                                                 this->m_prev_activations_cudnn_desc,
                                                                 m_convolution_cudnn_desc,
                                                                 m_kernel_cudnn_desc,
                                                                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                                 work_space_size,
                                                                 &kernel_gradient_cudnn_algorithm));
          CHECK_CUDNN(cudnnConvolutionBackwardFilter(this->m_cudnn->get_handle(i),
                                                     &one,
                                                     this->m_prev_error_signals_cudnn_desc,
                                                     gradient_wrt_output_d.get_locked_data(i),
                                                     this->m_prev_activations_cudnn_desc,
                                                     input_d.get_locked_data(i),
                                                     m_convolution_cudnn_desc,
                                                     kernel_gradient_cudnn_algorithm,
                                                     work_spaces_d[i],
                                                     work_space_size,
                                                     &zero,
                                                     m_kernel_cudnn_desc,
                                                     m_kernel_gradient_d.get_data(i)));
        }
        else {
          CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->m_cudnn->get_handle(i),
                                                                 this->m_prev_activations_cudnn_desc,
                                                                 this->m_prev_error_signals_cudnn_desc,
                                                                 m_convolution_cudnn_desc,
                                                                 m_kernel_cudnn_desc,
                                                                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                                 work_space_size,
                                                                 &kernel_gradient_cudnn_algorithm));
          CHECK_CUDNN(cudnnConvolutionBackwardFilter(this->m_cudnn->get_handle(i),
                                                     &one,
                                                     this->m_prev_activations_cudnn_desc,
                                                     input_d.get_locked_data(i),
                                                     this->m_prev_error_signals_cudnn_desc,
                                                     gradient_wrt_output_d.get_locked_data(i),
                                                     m_convolution_cudnn_desc,
                                                     kernel_gradient_cudnn_algorithm,
                                                     work_spaces_d[i],
                                                     work_space_size,
                                                     &zero,
                                                     m_kernel_cudnn_desc,
                                                     m_kernel_gradient_d.get_data(i)));
        }

      }

      // Add gradient contribution
      kernel_optimizer->stage_gradient_for_accumulation_gpu(
        m_kernel_gradient_d.get_locked_data(),
        one / effective_mini_batch_size);
    }

  #endif // LBANN_HAS_CUDNN
  }

  /** Convolution with im2col GEMM algorithm. */
  void apply_convolution_im2col(bool during_forward_prop) {

    // Local matrices
    const auto& local_kernel = this->m_weights[0]->get_values().LockedMatrix();
    const auto& local_input = (during_forward_prop ?
                               get_local_prev_activations() :
                               get_local_prev_error_signals());
    auto& local_output = (during_forward_prop ?
                          get_local_activations() :
                          get_local_error_signals());

    // Matrix parameters
    const int output_size = local_output.Height();
    const El::Int local_width = local_input.Width();
    DataType mixing_factor;
    std::vector<int> input_dims, output_dims;
    if (during_forward_prop) {
      mixing_factor = DataType(0);
      input_dims = this->m_prev_neuron_dims;
      output_dims = this->m_neuron_dims;
    }
    else {
      mixing_factor = DataType(1);
      input_dims = this->m_neuron_dims;
      output_dims = this->m_prev_neuron_dims;
    }

    // Initialize matrices
    const int m = output_size / output_dims[0];
    const int n = output_dims[0];
    const int k = m_kernel_size / output_dims[0];
    Mat input_col, output_col;
    Mat im2col_matrix(k, m);
    const Mat kernel_matrix(k, n, local_kernel.LockedBuffer(), k);

    // Iterate through input columns
    for (El::Int col = 0; col < local_width; ++col) {

      // Construct im2col matrix from current input column
      El::LockedView(input_col, local_input, El::ALL, El::IR(col));
      im2col(input_col,
             im2col_matrix,
             input_dims[0],
             input_dims.size() - 1,
             &input_dims[1],
             m_pads.data(),
             &m_kernel_dims[2],
             m_strides.data());

      // Apply convolution to current input column
      output_col.Attach(m, n, local_output.Buffer(0, col), m);
      El::Gemm(El::TRANSPOSE, El::NORMAL,
               DataType(1), im2col_matrix, kernel_matrix,
               mixing_factor, output_col);

    }

  }

  /** Transposed convolution with im2col GEMM algorithm. */
  void apply_transposed_convolution_im2col(bool during_forward_prop) {

    // Local matrices
    const auto& local_kernel = this->m_weights[0]->get_values().LockedMatrix();
    const auto& local_input = (during_forward_prop ?
                               get_local_prev_activations() :
                               get_local_prev_error_signals());
    auto& local_output = (during_forward_prop ?
                          get_local_activations() :
                          get_local_error_signals());

    // Matrix parameters
    const int input_size = local_input.Height();
    const int output_size = local_output.Height();
    const El::Int local_width = local_input.Width();
    std::vector<int> input_dims, output_dims;
    if (during_forward_prop) {
      input_dims = this->m_prev_neuron_dims;
      output_dims = this->m_neuron_dims;
    }
    else {
      input_dims = this->m_neuron_dims;
      output_dims = this->m_prev_neuron_dims;
    }

    // Initialize matrices
    const int m = m_kernel_size / input_dims[0];
    const int n = input_size / input_dims[0];
    const int k = input_dims[0];
    Mat input_col, output_col;
    Mat im2col_matrix(m, n);
    const Mat kernel_matrix(m, k, local_kernel.LockedBuffer(), m);

    // Iterate through input columns
    for (El::Int col = 0; col < local_width; ++col) {

      // Apply transposed convolution to current input column
      input_col.LockedAttach(n, k, local_input.LockedBuffer(0, col), n);
      El::Gemm(El::NORMAL, El::TRANSPOSE,
               DataType(1), kernel_matrix, input_col,
               DataType(0), im2col_matrix);

      // Perform col2im to accumulate contributions from each kernel
      // position
      if (during_forward_prop) {
        El::View(output_col, local_output, El::ALL, El::IR(col));
      } else {
        output_col.Resize(output_size, 1);
      }
      col2im(im2col_matrix,
             output_col,
             output_dims[0],
             output_dims.size() - 1,
             &output_dims[1],
             m_pads.data(),
             &m_kernel_dims[2],
             m_strides.data());
      if (!during_forward_prop) {
        local_output(El::ALL, El::IR(col)) += output_col;
      }

    }

  }

  void apply_bias_cpu() {

    // Return immediately if there is no bias
    if (m_bias_scaling_factor == DataType(0)) return;

    // Local matrices
    const auto& local_bias = m_weights[1]->get_values().LockedMatrix();
    auto& local_output = get_local_activations();

    // Matrix parameters
    const El::Int local_width = local_output.Width();
    const El::Int num_output_channels = this->m_neuron_dims[0];
    const El::Int num_per_output_channel = this->m_num_neurons / num_output_channels;

    // Apply bias to each output channel
    #pragma omp parallel for
    for (El::Int channel = 0; channel < num_output_channels; ++channel) {
      const El::Int row_start = channel * num_per_output_channel;
      const El::Int row_end = (channel+1) * num_per_output_channel;
      const DataType bias_term = m_bias_scaling_factor * local_bias(channel, 0);
      for (El::Int col = 0; col < local_width; ++col) {
        for (El::Int row = row_start; row < row_end; ++row) {
          local_output(row, col) += bias_term;
        }
      }
    }

  }

  void compute_gradients_im2col(bool using_transposed_convolution) {

    // Local matrices
    const auto& local_input = get_local_prev_activations();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_kernel_gradient = m_kernel_gradient.Matrix();
    auto& local_bias_gradient = m_bias_gradient.Matrix();
    
    // Get convolution parameters
    const El::Int local_width = local_input.Width();
    const int num_input_channels = this->m_prev_neuron_dims[0];
    const int num_output_channels = this->m_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_output_channels;
    const int effective_mini_batch_size = this->m_model->get_effective_mini_batch_size();

    // Compute bias gradient
    // Note: Sum is computed with Kahan summation
    optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
    if (m_bias_scaling_factor != DataType(0) && bias_optimizer != nullptr) {
      #pragma omp parallel for
      for (int channel = 0; channel < num_output_channels; ++channel) {
        const El::Int row_start = channel * num_per_output_channel;
        const El::Int row_end = (channel+1) * num_per_output_channel;
        DataType sum = 0;
        DataType correction = 0;
        for (El::Int col = 0; col < local_width; ++col) {
          for (El::Int row = row_start; row < row_end; ++row) {
            DataType term = local_gradient_wrt_output(row, col);
            term += correction;
            const DataType next_sum = sum + term;
            correction = term - (next_sum - sum);
            sum = next_sum;
          }
        }
        local_bias_gradient(channel, 0) = m_bias_scaling_factor * sum;
      }
      bias_optimizer->stage_gradient_for_accumulation(
        m_bias_gradient,
        DataType(1) / effective_mini_batch_size);
    }

    // Stop early if kernel is not being optimized
    optimizer* kernel_optimizer = this->m_weights[0]->get_optimizer();
    if (kernel_optimizer == nullptr) { return; }

    // Initialize matrices
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
    Mat kernel_gradient_matrix(m, n, local_kernel_gradient.Buffer(), m);
    El::Zero(kernel_gradient_matrix);

    // Compute kernel gradient contributions from each data sample
    for (El::Int col = 0; col < local_width; ++col) {
      if (using_transposed_convolution) {
        const Mat input_col(k, n, local_input.LockedBuffer(0,col), k);
        const Mat gradient_wrt_output_col
          = El::LockedView(local_gradient_wrt_output, El::ALL, El::IR(col));
        im2col(gradient_wrt_output_col,
               im2col_matrix,
               num_output_channels,
               this->m_num_neuron_dims - 1,
               &this->m_neuron_dims[1],
               m_pads.data(),
               &m_kernel_dims[2],
               m_strides.data());
        El::Gemm(El::NORMAL, El::NORMAL,
                 DataType(1), im2col_matrix, input_col,
                 DataType(1), kernel_gradient_matrix);
      }
      else {
        const Mat input_col
          = El::LockedView(local_input, El::ALL, El::IR(col));
        const Mat gradient_wrt_output_col(k, n, local_gradient_wrt_output.LockedBuffer(0,col), k);
        im2col(input_col,
               im2col_matrix,
               num_input_channels,
               this->m_num_prev_neuron_dims - 1,
               &this->m_prev_neuron_dims[1],
               m_pads.data(),
               &m_kernel_dims[2],
               m_strides.data());
        El::Gemm(El::NORMAL, El::NORMAL,
                 DataType(1), im2col_matrix, gradient_wrt_output_col,
                 DataType(1), kernel_gradient_matrix);
      }
    }

    // Scale and accumulate gradients
    kernel_optimizer->stage_gradient_for_accumulation(
      m_kernel_gradient,
      DataType(1) / effective_mini_batch_size);

  }

};

} // namespace lbann

#endif // LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED
