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
// deconvolution.hpp - Deconvolution Layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/base.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

// Forward declaration.
class lbann_callback_imcomm;

/// Convolution layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class deconvolution_layer : public base_convolution_layer {
 private:

  friend class lbann_callback_imcomm;

  public:


  deconvolution_layer(int index,
                      lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      int conv_dim,
                      int conv_pad,
                      int conv_stride,
                      weight_initialization init,
                      optimizer *opt,
                      bool has_bias = true,
                      DataType bias_initial_value = DataType(0),
                      cudnn::cudnn_manager *cudnn = nullptr)
    : deconvolution_layer(index,
                          comm,
                          num_data_dims,
                          num_output_channels,
                          std::vector<int>(num_data_dims, conv_dim).data(),
                          std::vector<int>(num_data_dims, conv_pad).data(),
                          std::vector<int>(num_data_dims, conv_stride).data(),
                          init,
                          opt,
                          has_bias,
                          bias_initial_value,
                          cudnn) {}

  deconvolution_layer(int index,
                      lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      const int *conv_dims,
                      const int *conv_pads,
                      const int *conv_strides,
                      weight_initialization init,
                      optimizer *opt,
                      bool has_bias = true,
                      DataType bias_initial_value = DataType(0),
                      cudnn::cudnn_manager *cudnn = nullptr)
    : base_convolution_layer(index,
                             comm,
                             num_data_dims,
                             num_output_channels,
                             conv_dims,
                             conv_pads,
                             conv_strides,
                             init,
                             opt,
                             has_bias,
                             bias_initial_value,
                             cudnn) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "convolution only supports DATA_PARALLEL");
    
    // Setup the data distribution
    initialize_distributed_matrices();

    // Use GPUs if cuDNN manager is available
    if(this->m_cudnn != nullptr) {
      this->m_using_gpus = true;
    }

  }

  /** Returns description of ctor params */
  std::string get_description() const {
    std::stringstream s;
    s << " deconvolution; conv_dims: ";
    for (size_t h=0; h<this->m_kernel_dims.size(); h++) {
      s << this->m_kernel_dims[h] << " ";
    }
    s << " conv_pads: ";
    for (size_t h=0; h<this->m_conv_pads.size(); h++) {
      s << this->m_conv_pads[h] << " ";
    }
    s << " conv_strides: ";
    for (size_t h=0; h<this->m_conv_strides.size(); h++) {
      s << this->m_conv_strides[h] << " ";
    }
    s << " num_output_channels: " << this->m_neuron_dims[0]
      << " weight_init: " + get_weight_initialization_name(this->m_weight_initialization) 
      << " has_bias: " << this->m_bias_scaling_factor
      << " bias_initial_value: " << this->m_bias_initial_value
      << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  deconvolution_layer(const deconvolution_layer& other) :
    base_convolution_layer(other) {}

  deconvolution_layer& operator=(const deconvolution_layer& other) {
    base_convolution_layer::operator=(other);
    return *this;
  }

  ~deconvolution_layer() {}

  deconvolution_layer* copy() const { return new deconvolution_layer(*this); }

  std::string get_name() const { return "deconvolution"; }

  void initialize_distributed_matrices() {
    base_convolution_layer::initialize_distributed_matrices<T_layout>();
  }

  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_dims() {

    // Initialize previous neuron tensor dimensions
    base_convolution_layer::setup_dims();

    // Initialize convolution kernel dimensions
    this->m_kernel_dims.insert(this->m_kernel_dims.begin(),
                               this->m_prev_neuron_dims[0]);

    // Check if previous neuron tensor dimensions are valid
  #ifdef LBANN_DEBUG
    if(this->m_num_neuron_dims != (int) this->m_kernel_dims.size() - 1) {
      throw lbann_exception("deconvolution_layer: previous neuron tensor dimensions are unexpected");
    }
  #endif

    // Initialize neuron tensor dimensions
    this->m_neuron_dims[0] = this->m_kernel_dims[1];
    for(int i=0; i<this->m_num_neuron_dims-1; ++i) {
      this->m_neuron_dims[i+1]
        = ((this->m_prev_neuron_dims[i+1]-1) * this->m_conv_strides[i]
           + this->m_kernel_dims[i+2] - 2*this->m_conv_pads[i]);
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Get size of convolutional kernel
    this->m_kernel_size = std::accumulate(m_kernel_dims.begin(),
                                          m_kernel_dims.end(),
                                          1,
                                          std::multiplies<int>());

  }

  void setup_data() {
    if(m_bias_scaling_factor == DataType(0)) {
      El::Zeros(*this->m_weights,
                m_kernel_size / this->m_prev_neuron_dims[0],
                this->m_prev_neuron_dims[0]);
    }
    else {
      El::Zeros(*this->m_weights,
                m_kernel_size / this->m_prev_neuron_dims[0],
                this->m_prev_neuron_dims[0] + 1);
    }
    El::Zeros(*this->m_weights_gradient,
              this->m_weights->Height(),
              this->m_weights->Width());
    base_convolution_layer::setup_data();
  }

  void setup_views() {
    base_convolution_layer::setup_views();
    El::View(*m_kernel_weights_v, *this->m_weights,
             El::ALL, El::IR(0,this->m_prev_neuron_dims[0]));
    El::View(*m_kernel_weights_gradient_v, *this->m_weights_gradient,
             El::ALL, El::IR(0,this->m_prev_neuron_dims[0]));
    if(m_bias_scaling_factor != DataType(0)) {
      ElMat *bias_weights_v = dynamic_cast<ElMat*>(m_bias_weights_v);
      ElMat *bias_weights_gradient_v = dynamic_cast<ElMat*>(m_bias_weights_gradient_v);
      if(bias_weights_v == nullptr) {
        throw lbann_exception("deconvolution_layer: weights matrix has invalid data distribution");
      }
      if(bias_weights_gradient_v == nullptr) {
        throw lbann_exception("deconvolution_layer: weights gradient matrix has invalid data distribution");
      }
      bias_weights_v->Attach(1,
                             this->m_neuron_dims[0],
                             m_bias_weights_v->Grid(),
                             m_bias_weights_v->ColAlign(),
                             m_bias_weights_v->RowAlign(),
                             m_weights->Buffer(0,this->m_prev_neuron_dims[0]),
                             1,
                             m_bias_weights_v->Root());
      bias_weights_gradient_v->Attach(1,
                                      this->m_neuron_dims[0],
                                      m_bias_weights_gradient_v->Grid(),
                                      m_bias_weights_gradient_v->ColAlign(),
                                      m_bias_weights_gradient_v->RowAlign(),
                                      m_weights_gradient->Buffer(0,this->m_prev_neuron_dims[0]),
                                      1,
                                      m_bias_weights_gradient_v->Root());
    }
  }

  void setup_gpu() {
    base_convolution_layer::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("deconvolution_layer: cuDNN not detected");
  #else

    // Maximum workspace size
    cudaDeviceProp device_props;
    CHECK_CUDA(cudaGetDeviceProperties(&device_props, 0));
    const size_t work_space_limit = device_props.totalGlobalMem / 2;

    // Choose algorithms
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(this->m_cudnn->get_handle(),
                                                    this->m_neurons_cudnn_desc,
                                                    m_kernel_cudnn_desc,
                                                    m_convolution_cudnn_desc,
                                                    this->m_prev_neurons_cudnn_desc,
                                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                    work_space_limit,
                                                    &this->m_convolution_cudnn_algorithm));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(this->m_cudnn->get_handle(),
                                                         m_kernel_cudnn_desc,
                                                         this->m_prev_neurons_cudnn_desc,
                                                         m_convolution_cudnn_desc,
                                                         this->m_neurons_cudnn_desc,
                                                         CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                                         work_space_limit,
                                                         &this->m_transposed_convolution_cudnn_algorithm));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->m_cudnn->get_handle(),
                                                           this->m_neurons_cudnn_desc,
                                                           this->m_prev_neurons_cudnn_desc,
                                                           m_convolution_cudnn_desc,
                                                           m_kernel_cudnn_desc,
                                                           CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                           work_space_limit,
                                                           &this->m_kernel_gradient_cudnn_algorithm));

    // Initialize work space
    size_t max_work_space = 0;
    size_t required_work_space;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->m_cudnn->get_handle(),
                                                        this->m_neurons_cudnn_desc,
                                                        this->m_kernel_cudnn_desc,
                                                        this->m_convolution_cudnn_desc,
                                                        this->m_prev_neurons_cudnn_desc,
                                                        this->m_convolution_cudnn_algorithm,
                                                        &required_work_space));
    max_work_space = std::max(max_work_space, required_work_space);
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(this->m_cudnn->get_handle(),
                                                             this->m_kernel_cudnn_desc,
                                                             this->m_prev_neurons_cudnn_desc,
                                                             this->m_convolution_cudnn_desc,
                                                             this->m_neurons_cudnn_desc,
                                                             this->m_transposed_convolution_cudnn_algorithm,
                                                             &required_work_space));
    max_work_space = std::max(max_work_space, required_work_space);
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->m_cudnn->get_handle(),
                                                               this->m_neurons_cudnn_desc,
                                                               this->m_prev_neurons_cudnn_desc,
                                                               this->m_convolution_cudnn_desc,
                                                               this->m_kernel_cudnn_desc,
                                                               this->m_kernel_gradient_cudnn_algorithm,
                                                               &required_work_space));
    max_work_space = std::max(max_work_space, required_work_space);
    for(int i=0; i<this->m_cudnn->get_num_gpus(); ++i) {
      if(max_work_space > this->m_cudnn->get_work_space_size(i)) {
        this->m_cudnn->set_work_space_size(i, max_work_space);
      }
    }

  #endif // __LIB_CUDNN
    
  }

 protected:

  void fp_compute() {
    if(this->m_using_gpus) {
      apply_transposed_convolution_cudnn(true);
      apply_bias_cudnn();
    } else {
      apply_transposed_convolution_im2col(true);
      apply_bias_cpu();
    }
    l2_regularize_objective_function();
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      apply_convolution_cudnn(false);
      compute_gradients_cudnn(true);
    } else {
      apply_convolution_im2col(false);
      compute_gradients_im2col(true);
    }
    l2_regularize_gradient();
  }

};
}

#endif // LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED
