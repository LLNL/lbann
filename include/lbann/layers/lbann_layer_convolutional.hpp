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
// lbann_layer_convolutional .hpp .cpp - Convolutional Layer
// 07/06/2016: changing distributed matrices to STAR,VC format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_CONVOLUTIONAL_HPP_INCLUDED
#define LBANN_LAYER_CONVOLUTIONAL_HPP_INCLUDED

#include <vector>
#include "lbann/lbann_base.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

namespace lbann
{

  /// Convolutional layer
  class convolutional_layer : public Layer
  {
  public:

    /// Constructor
    convolutional_layer(uint index,
                        Int num_dims,
                        Int num_input_channels,
                        const Int* input_dims,
                        Int num_output_channels,
                        const Int* filter_dims,
                        const Int* conv_pads,
                        const Int* conv_strides,
                        Int mini_batch_size,
                        activation_type activation,
                        weight_initialization init,
                        lbann_comm* comm,
                        Optimizer* optimizer,
                        cudnn::cudnn_manager* cudnn=NULL);

    /// Destructor
    ~convolutional_layer();

    void setup(int num_prev_neurons);

    bool update();

  protected:

    void fp_linearity();
    void fp_nonlinearity();
    void bp_linearity();
    void bp_nonlinearity();

  private:

    /// Weight initialization scheme
    const weight_initialization m_weight_initialization;
    /// Number of data dimensions
    const Int m_num_dims;
    /// Number of input channels
    const Int m_num_input_channels;
    /// Input dimensions
    /** In HW or DHW format */
    std::vector<Int> m_input_dims;
    /// Number of output channels
    const Int m_num_output_channels;
    /// Output dimensions
    std::vector<Int> m_output_dims;
    /// Filter dimensions
    std::vector<Int> m_filter_dims;
    /// Number of filter weights
    Int m_filter_size;
    /// Convolution padding
    std::vector<Int> m_conv_pads;
    /// Convolution strides
    std::vector<Int> m_conv_strides;

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
    /// Activation descriptor
    cudnnActivationDescriptor_t m_activation_desc;

    /// Forward pass algorithm
    cudnnConvolutionFwdAlgo_t m_forward_algo;
    /// Backward pass filter algorithm
    /** Compute gradient w.r.t. filter. */
    cudnnConvolutionBwdFilterAlgo_t m_backward_filter_algo;
    /// Backward pass data algorithm
    /** Compute gradient w.r.t. data, which is passed to previous layer. */
    cudnnConvolutionBwdDataAlgo_t m_backward_data_algo;

    /// GPU memory for convolution filters and bias
    std::vector<DataType*> m_weights_d;
    /// GPU memory for convolution filters gradient and bias gradient
    std::vector<DataType*> m_weights_gradient_d;
    /// GPU memory for work space
    std::vector<DataType*> m_work_space_d;
    /// Size of work space in bytes
    size_t m_work_space_size;

    /// Filter and bias gradients computed on each GPU
    StarMat m_weights_gradient_per_gpu;

#endif // __LIB_CUDNN

    /// Initialize GPU objects
    void setup_gpu();

    /// CPU implementation of forward propagation linearity
    void fp_linearity_cpu();
    /// GPU implementation of forward propagation linearity
    void fp_linearity_gpu();
    /// GPU implementation of forward propagation nonlinearity
    void fp_nonlinearity_gpu();
    /// CPU implementation of backward propagation linearity
    void bp_linearity_cpu();
    /// GPU implementation of backward propagation linearity
    void bp_linearity_gpu();
    /// GPU implementation of backward propagation nonlinearity
    void bp_nonlinearity_gpu();

    bool is_pinned_fwd;
    bool is_pinned_bwd;
    void pin_memory_blocks_fwd(void);
    void pin_memory_blocks_bwd(void);
  };

}

#endif // LBANN_LAYER_CONVOLUTIONAL_HPP_INCLUDED
