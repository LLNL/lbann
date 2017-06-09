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

namespace lbann {

/// Pooling layer
class pooling_layer : public Layer {

 public:

  /// Constructor
  pooling_layer(uint index,
                int num_dims,
                int num_channels,
                const int *input_dims,
                const int *pool_dims,
                const int *pool_pads,
                const int *pool_strides,
                pool_mode _pool_mode,
                uint mini_batch_size,
                lbann_comm *comm,
                cudnn::cudnn_manager *cudnn=NULL);

  /// Destructor
  ~pooling_layer();

  void setup(int num_prev_neurons);

  void forwardProp();
  void backProp();

  bool update();
  void pin_mem(void);
  void unpin_mem(void);

 protected:

  void fp_linearity();
  void bp_linearity();

 private:

  /// Pooling mode
  const pool_mode m_pool_mode;

  /// Number of data dimensions
  const Int m_num_dims;
  /// Number of channels
  const Int m_num_channels;
  /// Input dimensions
  /** In HW or DHW format */
  std::vector<Int> m_input_dims;
  /// Output dimensions
  std::vector<Int> m_output_dims;
  /// Pooling window dimensions
  std::vector<Int> m_pool_dims;
  /// Pooling padding
  std::vector<Int> m_pool_pads;
  /// Pooling strides
  std::vector<Int> m_pool_strides;
  /// Size of pooling window
  Int m_pool_size;

#ifdef __LIB_CUDNN
  /// Input tensor descriptor
  cudnnTensorDescriptor_t m_input_desc;
  /// Output tensor descriptor
  cudnnTensorDescriptor_t m_output_desc;
  /// Pooling descriptor
  cudnnPoolingDescriptor_t m_pooling_desc;
#endif // __LIB_CUDNN

  /// Initialize GPU objects
  void setup_gpu();

  /// CPU implementation of forward propagation linearity
  void fp_linearity_cpu();
  /// GPU implementation of forward propagation linearity
  void fp_linearity_gpu();
  /// CPU implementation of backward propagation linearity
  void bp_linearity_cpu();
  /// GPU implementation of backward propagation linearity
  void bp_linearity_gpu();

  bool to_pin_fwd; ///< request to pin the memory used by cudnn forward path
  bool to_pin_bwd; ///< request to pin the memory used by cudnn backward path
  bool is_pinned_fwd; ///< indicate if the memory blocks for cudnn forward path are pinned
  bool is_pinned_bwd; ///< indicate if the memory blocks for cudnn backward path are pinned
  void pin_memory_blocks_fwd(void); ///< pin the memory used by cudnn forward path
  void pin_memory_blocks_bwd(void); ///< pin the memory used by cudnn backward path
  void unpin_memory_blocks_fwd(void); ///< unpin the memory used by cudnn forward path
  void unpin_memory_blocks_bwd(void); ///< unpin the memory used by cudnn backward path
  void *get_cudnn_manager(void); ///< returns the pointer to cudnn_manager if available, otherwise NULL
};

}

#endif // LBANN_LAYER_POOLING_HPP_INCLUDED
