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
    convolutional_layer(uint index, int num_dims,
                        int num_input_channels, const int* input_dims,
                        int num_output_channels, const int* filter_dims,
                        const int* conv_pads, const int* conv_strides,
                        uint mini_batch_size,
                        activation_type activation,
                        lbann_comm* comm, Optimizer* optimizer,
                        std::vector<regularizer*> regs,
                        cudnn::cudnn_manager* cudnn=NULL);

    /// Destructor
    ~convolutional_layer();

    void setup(int num_prev_neurons);

    bool update();

  protected:
    
    void fp_linearity(ElMat& _WB, ElMat& _X, ElMat& _Z, ElMat& _Y);
    void bp_linearity();

  public:
    /// Number of data dimensions
    const int m_num_dims;
    /// Number of input channels
    const int m_num_input_channels;
    /// Input dimensions
    /** In HW or DHW format */
    std::vector<int> m_input_dims;
    /// Number of output channels
    const int m_num_output_channels;
    /// Output dimensions
    std::vector<int> m_output_dims;
    /// Filter dimensions
    std::vector<int> m_filter_dims;
    /// Number of filter weights
    int m_filter_size;
    /// Convolution padding
    std::vector<int> m_conv_pads;
    /// Convolution strides
    std::vector<int> m_conv_strides;

  private:
    /// cuDNN convolutional layer
    cudnn::cudnn_convolutional_layer* m_cudnn_layer;
  
  };

}

#endif // LBANN_LAYER_CONVOLUTIONAL_HPP_INCLUDED
