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
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/lbann_exception.hpp"

namespace lbann {

/// Local Response Normalization layer
class local_response_normalization_layer : public Layer {

 public:

  /// Constructor
  local_response_normalization_layer
  (uint index,
   int num_dims,
   int num_channels,
   const int *input_dims,
   Int window_width,
   DataType lrn_alpha,
   DataType lrn_beta,
   DataType lrn_k,
   uint mini_batch_size,
   lbann_comm *comm,
   cudnn::cudnn_manager *cudnn=NULL);

  /// Destructor
  ~local_response_normalization_layer(void);

  void setup(int num_prev_neurons);

  bool update(void);

 protected:

  void fp_linearity(void);
  void bp_linearity(void);

 private:

  /// Number of data dimensions
  const Int m_num_dims;
  /// Number of channels
  const Int m_num_channels;
  /// Data dimensions
  /** In HW or DHW format */
  std::vector<Int> m_dims;
  /// Normalization window width
  Int m_window_width;
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

  /// Initialize GPU objects
  void setup_gpu(void);

  /// CPU implementation of forward propagation linearity
  void fp_linearity_cpu(void);
  /// GPU implementation of forward propagation linearity
  void fp_linearity_gpu(void);
  /// CPU implementation of backward propagation linearity
  void bp_linearity_cpu(void);
  /// GPU implementation of backward propagation linearity
  void bp_linearity_gpu(void);

};

}

#endif // LBANN_LAYER_POOLING_HPP_INCLUDED
