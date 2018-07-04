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
#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {

/** Base convolution layer.
 *  Parent class for convolution and deconvolution layers.
 */
template <El::Device Dev>
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
  StarMat<Dev> m_kernel_gradient;
  /** Bias gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the bias weights.
   */
  StarMat<Dev> m_bias_gradient;

#ifdef LBANN_HAS_CUDNN

  /** Convolution kernel cuDNN descriptor. */
  cudnnFilterDescriptor_t m_kernel_cudnn_desc;
  /** Convolution cuDNN descriptor. */
  cudnnConvolutionDescriptor_t m_convolution_cudnn_desc;
  /** Bias tensor cuDNN descriptor. */
  cudnnTensorDescriptor_t m_bias_cudnn_desc;
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager m_tensors_cudnn_desc;

#endif // LBANN_HAS_CUDNN

  public:

  base_convolution_layer(lbann_comm *comm,
                         int num_data_dims,
                         int num_output_channels,
                         const std::vector<int> conv_dims,
                         const std::vector<int> pads,
                         const std::vector<int> strides,
                         bool has_bias)
    : learning_layer(comm),
      m_kernel_dims(conv_dims),
      m_kernel_size(0),
      m_pads(pads),
      m_strides(strides),
      m_bias_scaling_factor(has_bias ? DataType(1) : DataType(0)),
      m_kernel_gradient(this->m_comm->get_model_grid()),
      m_bias_gradient(this->m_comm->get_model_grid())
#ifdef LBANN_HAS_CUDNN
    , m_kernel_cudnn_desc(nullptr),
      m_convolution_cudnn_desc(nullptr),
      m_bias_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {

    // Check dimensions of convolution parameters
    if ((int) m_kernel_dims.size() != num_data_dims
        || (int) m_pads.size() != num_data_dims
        || (int) m_strides.size() != num_data_dims) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" "
          << "has an invalid number of convolution parameters "
          << "(expected " << num_data_dims << " parameters, "
          << "conv_dims has " << m_kernel_dims.size() << ", "
          << "pads has " << m_pads.size() << ", "
          << "strides has " << m_strides.size() << ")";
      LBANN_ERROR(err.str());
    }

    // Record number of output channels
    m_kernel_dims.insert(m_kernel_dims.begin(), num_output_channels);

  }

  base_convolution_layer(const base_convolution_layer& other)
    : learning_layer(other),
      m_kernel_dims(other.m_kernel_dims),
      m_kernel_size(other.m_kernel_size),
      m_pads(other.m_pads),
      m_strides(other.m_strides),
      m_bias_scaling_factor(other.m_bias_scaling_factor),
      m_kernel_gradient(other.m_kernel_gradient),
      m_bias_gradient(other.m_bias_gradient)
#ifdef LBANN_HAS_CUDNN
    , m_kernel_cudnn_desc(nullptr),
      m_convolution_cudnn_desc(nullptr),
      m_bias_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {
#ifdef LBANN_HAS_CUDNN
    copy_kernel_cudnn_desc(other.m_kernel_cudnn_desc,
                           m_kernel_cudnn_desc);
    copy_convolution_cudnn_desc(other.m_convolution_cudnn_desc,
                                m_convolution_cudnn_desc);
    cudnn::copy_tensor_desc(other.m_bias_cudnn_desc,
                            m_bias_cudnn_desc);
    m_tensors_cudnn_desc.set_layer(this);
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
    copy_kernel_cudnn_desc(other.m_kernel_cudnn_desc,
                           m_kernel_cudnn_desc);
    copy_convolution_cudnn_desc(other.m_convolution_cudnn_desc,
                                m_convolution_cudnn_desc);
    cudnn::copy_tensor_desc(other.m_bias_cudnn_desc,
                            m_bias_cudnn_desc);
    m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN

    return *this;
  }

  ~base_convolution_layer() {
#ifdef LBANN_HAS_CUDNN
    if (m_kernel_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyFilterDescriptor(m_kernel_cudnn_desc));
    }
    if (m_convolution_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(m_convolution_cudnn_desc));
    }
    if (m_bias_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_bias_cudnn_desc));
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
      err << "attempted to setup layer \"" << get_name() << "\" "
          << "with an invalid number of weights "
          << "(expected at most 2, "
          << "found " << this->m_weights.size() << ")";
      LBANN_ERROR(err.str());
    }
    this->m_weights.resize(2, nullptr);
    if (this->m_weights[0] == nullptr) {
      this->m_weights[0] = new weights(this->m_comm);
      this->m_weights[0]->set_name(this->m_name + "_kernel");
      this->m_weights[0]->set_initializer(new he_normal_initializer(this->m_comm));
      this->m_weights[0]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(this->m_comm);
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
    this->m_weights[1]->setup(this->m_neuron_dims[0], Dev);
    El::Zeros(m_bias_gradient,
              this->m_weights[1]->get_matrix_height(),
              this->m_weights[1]->get_matrix_width());

    if (m_frozen) {
      this->m_weights[0]->freeze();
      this->m_weights[1]->freeze();
    } else {
      if (this->m_weights[0]->is_frozen()) {
        std::stringstream err;
        err << "unfrozen layer \"" << get_name() << "\" "
            << "has frozen weights \"" << this->m_weights[0]->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
      if (this->m_weights[1]->is_frozen()) {
        std::stringstream err;
        err << "unfrozen layer \"" << get_name() << "\" "
            << "has frozen weights \"" << this->m_weights[1]->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
    }
  }

  /// Initialize GPU objects
  void setup_gpu() override {
    learning_layer::setup_gpu();
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else

    // Set kernel descriptor
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&m_kernel_cudnn_desc));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(m_kernel_cudnn_desc,
                                           cudnn::get_data_type(),
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
                                                cudnn::get_data_type()));

    // Set bias tensor descriptor
    std::vector<int> bias_dims(get_num_neuron_dims() + 1, 1);
    bias_dims[1] = get_neuron_dims()[0];
    cudnn::set_tensor_desc(m_bias_cudnn_desc, bias_dims);

  #endif // LBANN_HAS_CUDNN
  }

  virtual void check_setup() override {
    learning_layer::check_setup();
    std::stringstream err;

    // Check that kernel and bias weights are both initialized
    if (this->m_weights.size() != 2
        || this->m_weights[0] == nullptr
        || this->m_weights[1] == nullptr) {
      err << "failed to setup weights in layer \"" << get_name() << "\"";
      LBANN_ERROR(err.str());
    }

    // Check that kernel data and kernel gradient data are contiguous
    const auto& kernel = this->m_weights[0]->get_values();
    if (kernel.LocalWidth() > 1
        && kernel.LDim() != kernel.LocalHeight()) {
      err << "kernel data in layer \"" << m_name << "\" "
          << "is not contiguous";
      LBANN_ERROR(err.str());
    }
    if (m_kernel_gradient.LocalWidth() > 1
        && m_kernel_gradient.LDim() != m_kernel_gradient.LocalHeight()) {
      err << "kernel gradient data in layer \"" << m_name << "\" "
          << "is not contiguous";
      LBANN_ERROR(err.str());
    }

  }

 protected:

  /** Convolution with cuDNN. */
  void apply_convolution_cudnn(bool during_forward_prop) {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else

    // Useful constants
    const DataType zero = DataType(0);
    const DataType one = DataType(1);

    // Matrices
    const auto& kernel = m_weights[0]->get_values();
    const auto& input = (during_forward_prop ?
                         get_local_prev_activations() :
                         get_local_prev_error_signals());
    auto& output = (during_forward_prop ?
                    get_local_activations() :
                    get_local_error_signals());

    // Do nothing if there is no local data
    if (input.Height() < 1 || input.Width() < 1
        || output.Height() < 1 || output.Width() < 1) {
      return;
    }

    // Initialize GPU workspace
    GPUMat workspace;
#ifdef HYDROGEN_HAVE_CUB
    workspace.SetMemoryMode(1);
#endif // HYDROGEN_HAVE_CUB
    size_t workspace_size = 1 << 30; /// @todo Allocate largest free block
    workspace.Resize(workspace_size / sizeof(DataType), 1);
    workspace_size = workspace.Height() * sizeof(DataType);

    // Convolution parameters
    std::vector<int> input_dims, output_dims;
    cudnnTensorDescriptor_t input_desc, output_desc;
    if (during_forward_prop) {
      input_dims = get_prev_neuron_dims();
      output_dims = get_neuron_dims();
      input_desc = m_tensors_cudnn_desc.get_prev_activations();
      output_desc = m_tensors_cudnn_desc.get_activations();
    }
    else {
      input_dims = get_neuron_dims();
      output_dims = get_prev_neuron_dims();
      input_desc = m_tensors_cudnn_desc.get_prev_error_signals();
      output_desc = m_tensors_cudnn_desc.get_error_signals();
    }

    // Perform convolution on the GPU
    // Determine convolution algorithm
    cudnnConvolutionFwdAlgo_t convolution_cudnn_algorithm
      = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn::get_handle(),
                                                    input_desc,
                                                    m_kernel_cudnn_desc,
                                                    m_convolution_cudnn_desc,
                                                    output_desc,
                                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                    workspace_size,
                                                    &convolution_cudnn_algorithm));

    // Apply convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn::get_handle(),
                                        &one,
                                        input_desc,
                                        input.LockedBuffer(),
                                        m_kernel_cudnn_desc,
                                        kernel.LockedBuffer(),
                                        m_convolution_cudnn_desc,
                                        convolution_cudnn_algorithm,
                                        workspace.Buffer(),
                                        workspace_size,
                                        &zero,
                                        output_desc,
                                        output.Buffer()));

#endif // LBANN_HAS_CUDNN
  }

  /** Transposed convolution with cuDNN. */
  void apply_transposed_convolution_cudnn(bool during_forward_prop) {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else

    // Useful constants
    const DataType zero = DataType(0);
    const DataType one = DataType(1);

    // GPU data
    const auto& kernel = m_weights[0]->get_values();
    const auto& input = (during_forward_prop ?
                         get_local_prev_activations() :
                         get_local_prev_error_signals());
    auto& output = (during_forward_prop ?
                    get_local_activations() :
                    get_local_error_signals());

    // Do nothing if there is no local data
    if (input.Height() < 1 || input.Width() < 1
        || output.Height() < 1 || output.Width() < 1) {
      return;
    }

    // Initialize GPU workspace
    // Note: Use CUB GPU memory pool if possible
    GPUMat workspace;
#ifdef HYDROGEN_HAVE_CUB
    workspace.SetMemoryMode(1);
#endif // HYDROGEN_HAVE_CUB
    size_t workspace_size = 1 << 30; /// @todo Allocate largest free block
    workspace.Resize(workspace_size / sizeof(DataType), 1);
    workspace_size = workspace.Height() * sizeof(DataType);

    // Convolution transpose parameters
    std::vector<int> input_dims, output_dims;
    cudnnTensorDescriptor_t input_desc, output_desc;
    if (during_forward_prop) {
      input_dims = get_prev_neuron_dims();
      output_dims = get_neuron_dims();
      input_desc = m_tensors_cudnn_desc.get_prev_activations();
      output_desc = m_tensors_cudnn_desc.get_activations();
    }
    else {
      input_dims = get_neuron_dims();
      output_dims = get_prev_neuron_dims();
      input_desc = m_tensors_cudnn_desc.get_prev_error_signals();
      output_desc = m_tensors_cudnn_desc.get_error_signals();
    }

    // Perform transposed convolution on the GPU
    // Determine transposed convolution algorithm
    #ifndef LBANN_DETERMINISTIC
    cudnnConvolutionBwdDataAlgo_t transposed_convolution_cudnn_algorithm
      = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn::get_handle(),
                                                         m_kernel_cudnn_desc,
                                                         input_desc,
                                                         m_convolution_cudnn_desc,
                                                         output_desc,
                                                         CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                                         workspace_size,
                                                         &transposed_convolution_cudnn_algorithm));
    #else
    cudnnConvolutionBwdDataAlgo_t transposed_convolution_cudnn_algorithm
      = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    #endif
    // Perform transposed convolution
    CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn::get_handle(),
                                             &one,
                                             m_kernel_cudnn_desc,
                                             kernel.LockedBuffer(),
                                             input_desc,
                                             input.LockedBuffer(),
                                             m_convolution_cudnn_desc,
                                             transposed_convolution_cudnn_algorithm,
                                             workspace.Buffer(),
                                             workspace_size,
                                             &zero,
                                             output_desc,
                                             output.Buffer()));


  #endif // LBANN_HAS_CUDNN
  }

  void apply_bias_cudnn() {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    auto& local_output = get_local_activations();
    if (m_bias_scaling_factor != DataType(0)
        && local_output.Height() > 0
        && local_output.Width() > 0) {
      const DataType one = 1;
      const auto& bias = m_weights[1]->get_values();
      CHECK_CUDNN(cudnnAddTensor(cudnn::get_handle(),
                                 &m_bias_scaling_factor,
                                 m_bias_cudnn_desc,
                                 bias.LockedBuffer(),
                                 &one,
                                 m_tensors_cudnn_desc.get_activations(),
                                 local_output.Buffer()));
    }
  #endif // LBANN_HAS_CUDNN
  }

  void compute_gradients_cudnn(bool using_transposed_convolution) {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else

    // Matrices
    const auto& local_input = get_local_prev_activations();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();

    // Useful constants
    const DataType zero = DataType(0);
    const DataType one = DataType(1);
    const int effective_mini_batch_size = this->m_model->get_effective_mini_batch_size();
    const bool has_local_data = (local_input.Height() > 0
                                 && local_input.Width() > 0
                                 && local_gradient_wrt_output.Height() > 0
                                 && local_gradient_wrt_output.Width() > 0);

    // Compute bias gradient
    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr && m_bias_scaling_factor != DataType(0)) {
      if (!has_local_data) {
        El::Zero(m_bias_gradient);
      } else {
        CHECK_CUDNN(cudnnConvolutionBackwardBias(cudnn::get_handle(),
                                                 &one,
                                                 m_tensors_cudnn_desc.get_prev_error_signals(),
                                                 local_gradient_wrt_output.LockedBuffer(),
                                                 &zero,
                                                 m_bias_cudnn_desc,
                                                 m_bias_gradient.Buffer()));
      }
      bias_optimizer->add_to_gradient_staging(m_bias_gradient,
                                              m_bias_scaling_factor / effective_mini_batch_size);
    }

    // Compute kernel gradient
    optimizer* kernel_optimizer = m_weights[0]->get_optimizer();
    if (kernel_optimizer != nullptr) {
      if (!has_local_data) {
        El::Zero(m_kernel_gradient);
      } else {

        // Initialize GPU workspace
        GPUMat workspace;
#ifdef HYDROGEN_HAVE_CUB
        workspace.SetMemoryMode(1); // CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
        size_t workspace_size = 1 << 30; /// @todo Allocate largest free block
        workspace.Resize(workspace_size / sizeof(DataType), 1);
        workspace_size = workspace.Height() * sizeof(DataType);

        // Initialize cuDNN objects
        auto&& input_desc = m_tensors_cudnn_desc.get_prev_activations();
        auto&& gradient_wrt_output_desc = m_tensors_cudnn_desc.get_prev_error_signals();

        // Determine algorithm and compute kernel gradient
        #ifndef LBANN_DETERMINISTIC
        cudnnConvolutionBwdFilterAlgo_t kernel_gradient_cudnn_algorithm
          = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        #else
        cudnnConvolutionBwdFilterAlgo_t kernel_gradient_cudnn_algorithm
          = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        #endif
        if (using_transposed_convolution) {
          #ifndef LBANN_DETERMINISTIC
          CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn::get_handle(),
                                                                 gradient_wrt_output_desc,
                                                                 input_desc,
                                                                 m_convolution_cudnn_desc,
                                                                 m_kernel_cudnn_desc,
                                                                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                                 workspace_size,
                                                                 &kernel_gradient_cudnn_algorithm));
          #endif
          CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn::get_handle(),
                                                     &one,
                                                     gradient_wrt_output_desc,
                                                     local_gradient_wrt_output.LockedBuffer(),
                                                     input_desc,
                                                     local_input.LockedBuffer(),
                                                     m_convolution_cudnn_desc,
                                                     kernel_gradient_cudnn_algorithm,
                                                     workspace.Buffer(),
                                                     workspace_size,
                                                     &zero,
                                                     m_kernel_cudnn_desc,
                                                     m_kernel_gradient.Buffer()));
        }
        else {
          #ifndef LBANN_DETERMINISTIC
          CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn::get_handle(),
                                                                 input_desc,
                                                                 gradient_wrt_output_desc,
                                                                 m_convolution_cudnn_desc,
                                                                 m_kernel_cudnn_desc,
                                                                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                                 workspace_size,
                                                                 &kernel_gradient_cudnn_algorithm));
          #endif
          CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn::get_handle(),
                                                     &one,
                                                     input_desc,
                                                     local_input.LockedBuffer(),
                                                     gradient_wrt_output_desc,
                                                     local_gradient_wrt_output.LockedBuffer(),
                                                     m_convolution_cudnn_desc,
                                                     kernel_gradient_cudnn_algorithm,
                                                     workspace.Buffer(),
                                                     workspace_size,
                                                     &zero,
                                                     m_kernel_cudnn_desc,
                                                     m_kernel_gradient.Buffer()));

        }

      }

      // Add gradient contribution
      kernel_optimizer->add_to_gradient_staging(m_kernel_gradient,
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
    const int m = output_size / output_dims[0];
    const int n = output_dims[0];
    const int k = m_kernel_size / output_dims[0];
    DMat<Dev> input_col, output_col;
    DMat<Dev> im2col_matrix(k, m);
    const DMat<Dev> kernel_matrix(k, n, local_kernel.LockedBuffer(), k);

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
               DataType(0), output_col);

    }

  }

  /** Transposed convolution with im2col GEMM algorithm. */
  void apply_transposed_convolution_im2col(bool during_forward_prop) {

    // Local matrices
    const auto& local_kernel = this->m_weights[0]->get_values().LockedMatrix();
    const auto& local_input = (during_forward_prop ?
                               get_local_prev_activations() :
                               get_local_prev_error_signals());
    DMat<Dev>& local_output = (during_forward_prop ?
                               get_local_activations() :
                               get_local_error_signals());

    // Matrix parameters
    const int input_size = local_input.Height();
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
    DMat<Dev> input_col, output_col;
    DMat<Dev> im2col_matrix(m, n);
    const DMat<Dev> kernel_matrix(m, k, local_kernel.LockedBuffer(), m);

    // Iterate through input columns
    for (El::Int col = 0; col < local_width; ++col) {

      // Apply transposed convolution to current input column
      input_col.LockedAttach(n, k, local_input.LockedBuffer(0, col), n);
      El::Gemm(El::NORMAL, El::TRANSPOSE,
               DataType(1), kernel_matrix, input_col,
               DataType(0), im2col_matrix);

      // Perform col2im to accumulate contributions from each kernel
      // position
      El::View(output_col, local_output, El::ALL, El::IR(col));
      col2im(im2col_matrix,
             output_col,
             output_dims[0],
             output_dims.size() - 1,
             &output_dims[1],
             m_pads.data(),
             &m_kernel_dims[2],
             m_strides.data());

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
#pragma omp taskloop default(shared)
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
    const DMat<Dev>& local_input = get_local_prev_activations();
    const DMat<Dev>& local_gradient_wrt_output = get_local_prev_error_signals();
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
#pragma omp taskloop default(shared)
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
      const DataType bias_scale = m_bias_scaling_factor / effective_mini_batch_size;
      bias_optimizer->add_to_gradient_staging(m_bias_gradient,
                                              bias_scale);
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
    DMat<Dev> im2col_matrix(m, k);
    DMat<Dev> kernel_gradient_matrix(m, n, local_kernel_gradient.Buffer(), m);
    El::Zero(kernel_gradient_matrix);

    // Compute kernel gradient contributions from each data sample
    for (El::Int col = 0; col < local_width; ++col) {
      if (using_transposed_convolution) {
        const DMat<Dev> input_col(k, n, local_input.LockedBuffer(0,col), k);
        const DMat<Dev> gradient_wrt_output_col =
          El::LockedView(local_gradient_wrt_output, El::ALL, El::IR(col));
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
        const DMat<Dev> input_col
          = El::LockedView(local_input, El::ALL, El::IR(col));
        const DMat<Dev> gradient_wrt_output_col(k, n, local_gradient_wrt_output.LockedBuffer(0,col), k);
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
    const DataType kernel_scale = DataType(1) / effective_mini_batch_size;
    kernel_optimizer->add_to_gradient_staging(m_kernel_gradient,
                                              kernel_scale);

  }

 private:

#ifdef LBANN_HAS_CUDNN

  /** Copy convolution kernel cuDNN descriptor. */
  static void copy_kernel_cudnn_desc(const cudnnFilterDescriptor_t& src,
                                     cudnnFilterDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
      CHECK_CUDNN(cudnnCreateFilterDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
      CHECK_CUDNN(cudnnDestroyFilterDescriptor(dst));
      dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
      cudnnDataType_t data_type;
      cudnnTensorFormat_t format;
      int num_dims;
      CHECK_CUDNN(cudnnGetFilterNdDescriptor(src,
                                             0,
                                             &data_type,
                                             &format,
                                             &num_dims,
                                             nullptr));
      std::vector<int> dims(num_dims);
      CHECK_CUDNN(cudnnGetFilterNdDescriptor(src,
                                             num_dims,
                                             &data_type,
                                             &format,
                                             &num_dims,
                                             dims.data()));
      CHECK_CUDNN(cudnnSetFilterNdDescriptor(dst,
                                             data_type,
                                             format,
                                             num_dims,
                                             dims.data()));
    }

  }

  /** Copy convolution cuDNN descriptor. */
  static void copy_convolution_cudnn_desc(const cudnnConvolutionDescriptor_t& src,
                                          cudnnConvolutionDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
      CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
      CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(dst));
      dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
      cudnnConvolutionMode_t mode;
      cudnnDataType_t data_type;
      int num_dims;
      CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(src,
                                                  0,
                                                  &num_dims,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  &mode,
                                                  &data_type));
      std::vector<int> pads(num_dims), strides(num_dims), upscales(num_dims);
      CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(src,
                                                  num_dims,
                                                  &num_dims,
                                                  pads.data(),
                                                  strides.data(),
                                                  upscales.data(),
                                                  &mode,
                                                  &data_type));
      CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(dst,
                                                  num_dims,
                                                  pads.data(),
                                                  strides.data(),
                                                  upscales.data(),
                                                  mode,
                                                  data_type));
    }

  }

#endif // LBANN_HAS_CUDNN

};

} // namespace lbann

#endif // LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED
