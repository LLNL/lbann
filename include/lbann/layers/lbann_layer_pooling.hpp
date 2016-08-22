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
#include "lbann/lbann_base.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

namespace lbann
{

  /// Pooling layer
  class pooling_layer : public Layer
  {
  
  public:

    /// Constructor
    pooling_layer(uint index,
                  int num_dims,
                  int num_channels,
                  const int* input_dims,
                  const int* pool_dims,
                  const int* pool_pads,
                  const int* pool_strides,
                  pool_mode _pool_mode,
                  uint mini_batch_size,
                  activation_type activation,
                  lbann_comm* comm,
                  std::vector<regularizer*> regs,
                  cudnn::cudnn_manager* cudnn=NULL);

    /// Destructor
    ~pooling_layer();

    void setup(int num_prev_neurons);

    bool update();

  protected:
    
    void fp_linearity(ElMat& _WB, ElMat& _X, ElMat& _Z, ElMat& _Y);
    void bp_linearity();

  public:

    /// Pooling mode
    const pool_mode m_pool_mode;

    /// Number of data dimensions
    const int m_num_dims;
    /// Number of channels
    const int m_num_channels;
    /// Input dimensions
    /** In HW or DHW format */
    std::vector<int> m_input_dims;
    /// Output dimensions
    std::vector<int> m_output_dims;
    /// Pooling padding
    std::vector<int> m_pool_dims;
    /// Pooling padding
    std::vector<int> m_pool_pads;
    /// Pooling strides
    std::vector<int> m_pool_strides;

  private:

    /// cuDNN pooling layer
    cudnn::cudnn_pooling_layer* m_cudnn_layer;
  
  };

}

#endif // LBANN_LAYER_POOLING_HPP_INCLUDED
