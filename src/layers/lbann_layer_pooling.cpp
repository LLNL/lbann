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
// lbann_layer_pooling .hpp .cpp - Pooling layer
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_layer_pooling.hpp"
#include "lbann/utils/lbann_exception.hpp"

using namespace std;
using namespace El;
using namespace lbann;

pooling_layer::pooling_layer(const uint index,
                             const int num_dims,
                             const int num_channels,
                             const int* input_dims,
                             const int* pool_dims,
                             const int* pool_pads,
                             const int* pool_strides,
                             const pool_mode _pool_mode,
                             const uint mini_batch_size,
                             activation_type activation,
                             lbann_comm* comm,
                             std::vector<regularizer*> regs,
                             cudnn::cudnn_manager* cudnn)
  : Layer(index, comm, NULL, mini_batch_size, activation, regs),
    m_pool_mode(_pool_mode),
    m_num_dims(num_dims), m_num_channels(num_channels)
{

  // Initialize input dimensions and pooling parameters
  m_input_dims.resize(num_dims);
  m_pool_dims.resize(num_dims);
  m_pool_pads.resize(num_dims);
  m_pool_strides.resize(num_dims);
  for(int i=0; i<num_dims; ++i) {
    m_input_dims[i] = input_dims[i];
    m_pool_dims[i] = pool_dims[i];
    m_pool_pads[i] = pool_pads[i];
    m_pool_strides[i] = pool_strides[i];
  }

  // Calculate output dimensions
  m_output_dims.resize(num_dims);
  NumNeurons = num_channels;
  for(int i=0; i<num_dims; ++i) {
    m_output_dims[i] = input_dims[i]+2*pool_pads[i]-pool_dims[i]+1;
    m_output_dims[i] = (m_output_dims[i]+pool_strides[i]-1)/pool_strides[i];
    NumNeurons *= m_output_dims[i];
  }
  
  // Matrices should be in Star,VC distributions
  delete m_preactivations;
  delete m_prev_error_signal;
  delete m_error_signal;
  delete m_activations;
  m_preactivations = new StarVCMat(comm->get_model_grid());
  m_prev_error_signal = new StarVCMat(comm->get_model_grid());
  m_error_signal = new StarVCMat(comm->get_model_grid());
  m_activations = new StarVCMat(comm->get_model_grid());

  // Initialize cuDNN pooling layer
  m_cudnn_layer = NULL;
#ifdef __LIB_CUDNN
  if(cudnn)
    m_cudnn_layer = new cudnn::cudnn_pooling_layer(num_dims,
                                                   num_channels,
                                                   input_dims,
                                                   m_pool_mode,
                                                   pool_dims,
                                                   pool_pads,
                                                   pool_strides,
                                                   cudnn);
#endif // __LIB_CUDNN

}

pooling_layer::~pooling_layer()
{
#ifdef __LIB_CUDNN
  delete m_cudnn_layer;
#endif // __LIB_CUDNN
}

void pooling_layer::setup(const int num_prev_neurons)
{
  Layer::setup(num_prev_neurons);

#ifdef __LIB_CUDNN
  if(m_cudnn_layer) {
    // Setup cuDNN pooling layer
    m_cudnn_layer->setup();

    // Get output dimensions
    if(NumNeurons != m_cudnn_layer->m_dst_size)
      throw lbann_exception("lbann_layer_pooling: unexpected number of neurons");
    NumNeurons = m_cudnn_layer->m_dst_size;
    for(int i=0; i<m_num_dims; ++i)
      m_output_dims[i] = m_cudnn_layer->m_dst_dims[i+2];
  }
#endif // __LIB_CUDNN

  // Check if input dimensions are valid
  int num_inputs = m_num_channels;
  for(int i=0; i<m_num_dims; ++i)
    num_inputs *= m_input_dims[i];
  if(num_inputs != num_prev_neurons) {
    throw lbann_exception("lbann_layer_pooling: unexpected number of input neurons");
  }

  // Initialize matrices
  Ones(*m_preactivations, NumNeurons+1, m_mini_batch_size);
  Zeros(*m_prev_error_signal, NumNeurons+1, m_mini_batch_size);
  Zeros(*m_error_signal, num_prev_neurons+1, m_mini_batch_size);
  Ones(*m_activations, NumNeurons+1, m_mini_batch_size);

}

void lbann::pooling_layer::fp_linearity() {
  
  // Convert matrices to desired formats
  DistMatrixReadProxy<DataType,DataType,STAR,VC> XProxy(*fp_input);
  DistMatrixWriteProxy<DataType,DataType,STAR,VC> ZProxy(*m_preactivations);
  DistMatrixWriteProxy<DataType,DataType,STAR,VC> YProxy(*m_activations);
  StarVCMat& X = XProxy.Get();
  StarVCMat& Z = ZProxy.Get();
  StarVCMat& Y = YProxy.Get();

  // Get local matrices
  const Mat& XLocal = X.LockedMatrix();
  Mat& ZLocal = Z.Matrix();
  Mat& YLocal = Y.Matrix();

  // Apply pooling on local data samples
  if(m_cudnn_layer) {
#ifdef __LIB_CUDNN
    m_cudnn_layer->forward(XLocal, ZLocal);
#else
    throw lbann_exception("lbann_layer_pooling: cuDNN not detected");
#endif
  }
  else {

    ////////////////////////////////////////////////////////////
    // CPU implementation of pooling layer forward pass
    /// @todo Write a more efficient implementation
    ////////////////////////////////////////////////////////////

    // Throw exception if pooling mode is not max pooling
    if(m_pool_mode != pool_mode::max) {
      throw lbann_exception("lbann_layer_pooling: CPU pooling layer only implements max pooling");
    }

    // Iterate through data samples in mini-batch
    for(int sample = 0; sample < XLocal.Width(); ++sample) {
      const Mat input_sample = XLocal(IR(0,XLocal.Height()-1),
                                      IR(sample));
      Mat output_sample = ZLocal(IR(0,NumNeurons), IR(sample));

      // Iterate through channels
      for(int channel = 0; channel < m_num_channels; ++channel) {
        const int input_channel_size
          = input_sample.Height() / m_num_channels;
        const int output_channel_size = NumNeurons / m_num_channels;
        const Mat input_channel
          = input_sample(IR(channel*input_channel_size,
                            (channel+1)*input_channel_size),
                         ALL);
        Mat output_channel
          = output_sample(IR(channel*output_channel_size,
                             (channel+1)*output_channel_size),
                          ALL);

        // Iterate through pool offsets
        // Note: each offset corresponds to an output entry
        int output_pos = 0;
        std::vector<int> pool_offset(m_num_dims);
        for(int d = 0; d < m_num_dims; ++d) {
          pool_offset[d] = -m_pool_pads[d];
        }
        while(pool_offset[0] + m_pool_dims[0] <= m_input_dims[0] + m_pool_pads[0]) {

          // Iterate through pool entries and find maximum
          std::vector<int> pool_pos(m_num_dims, 0);
          DataType max_value = -INFINITY;
          while(pool_pos[0] < m_pool_dims[0]) {

            // Get position of pool entry
            int input_pos = 0;
            bool valid_pos = true;
            for(int d = 0; d < m_num_dims; ++d) {
              if(pool_offset[d] + pool_pos[d] < 0
                 || pool_offset[d] + pool_pos[d] >= m_input_dims[d]) {
                valid_pos = false;
                break;
              }
              input_pos *= m_input_dims[d];
              input_pos += pool_offset[d] + pool_pos[d];
            }

            // Check if pool entry is larger than previous
            DataType value = valid_pos ? input_channel.Get(input_pos, 0) : 0.0;
            max_value = value > max_value ? value : max_value;

            // Move to next pool entry
            ++pool_pos[m_num_dims-1];
            for(int d = m_num_dims - 1; d > 0; --d) {
              if(pool_pos[d] >= m_pool_dims[d]) {
                pool_pos[d] = 0;
                ++pool_pos[d-1];
              }
            }

          }

          // Set output entry
          output_channel.Set(output_pos, 0, max_value);

          // Move to next output entry
          ++output_pos;

          // Move to next pool offset
          pool_offset[m_num_dims-1] += m_pool_strides[m_num_dims-1];
          for(int d = m_num_dims - 1; d > 0; --d) {
            if(pool_offset[d] + m_pool_dims[d] > m_input_dims[d] + m_pool_pads[d]) {
              pool_offset[d] = -m_pool_pads[d];
              pool_offset[d-1] += m_pool_strides[d-1];
            }
            
          }

        }

      }

    }

  }

  // Z and Y are identical after fp linearity step
  Copy(ZLocal, YLocal);

}

void lbann::pooling_layer::bp_linearity() {

  // Convert matrices to desired formats
  DistMatrixReadProxy<DataType,DataType,STAR,VC> input_proxy(*fp_input); // TODO: store from fp step
  StarVCMat& input = input_proxy.Get();

  // Get local matrices
  const Mat& input_local = input.LockedMatrix();
  const Mat& output_local = m_activations->LockedMatrix();
  const Mat& prev_error_signal_local = m_prev_error_signal->LockedMatrix();
  Mat& error_signal_local = m_error_signal->Matrix();

  // Compute gradients on local data samples
  if(m_cudnn_layer) {
#ifdef __LIB_CUDNN
    m_cudnn_layer->backward(input_local,
                            output_local,
                            prev_error_signal_local,
                            error_signal_local);
#else
    throw lbann_exception("lbann_layer_pooling: cuDNN not detected");
#endif
  }
  else {

    ////////////////////////////////////////////////////////////
    // CPU implementation of pooling layer backward pass
    /// @todo Write a more efficient implementation
    ////////////////////////////////////////////////////////////

    // Throw exception if pooling mode is not max pooling
    if(m_pool_mode != pool_mode::max) {
      throw lbann_exception("lbann_layer_pooling: CPU pooling layer only implements max pooling");
    }

    // Iterate through data samples in mini-batch
    for(int sample = 0; sample < input_local.Width(); ++sample) {
      const Mat input_sample = input_local(IR(0,input_local.Height()-1),
                                           IR(sample));
      const Mat prev_error_signal_sample
        = prev_error_signal_local(IR(0,NumNeurons), IR(sample));
      Mat error_signal_sample
        = error_signal_local(IR(0,input_local.Height()-1), IR(sample));

      // Iterate through channels
      for(int channel = 0; channel < m_num_channels; ++channel) {
        const int input_channel_size = input_sample.Height() / m_num_channels;
        const int output_channel_size = NumNeurons / m_num_channels;
        const Mat input_channel
          = input_sample(IR(channel*input_channel_size,
                            (channel+1)*input_channel_size),
                         ALL);
        const Mat prev_error_signal_channel
          = prev_error_signal_sample(IR(channel*output_channel_size,
                                        (channel+1)*input_channel_size),
                                     ALL);
        Mat error_signal_channel
          = error_signal_sample(IR(channel*input_channel_size,
                                   (channel+1)*input_channel_size),
                                ALL);

        // Iterate through pool offsets
        // Note: each offset corresponds to an output entry
        int output_pos = 0;
        std::vector<int> pool_offset(m_num_dims);
        for(int d = 0; d < m_num_dims; ++d) {
          pool_offset[d] = -m_pool_pads[d];
        }
        while(pool_offset[0] + m_pool_dims[0] <= m_input_dims[0] + m_pool_pads[0]) {

          // Iterate through pool entries and find maximum
          std::vector<int> pool_pos(m_num_dims, 0);
          int max_input_pos = -1;
          DataType max_value = -INFINITY;
          while(pool_pos[0] < m_pool_dims[0]) {

            // Get position of pool entry
            int input_pos = 0;
            bool valid_pos = true;
            for(int d = 0; d < m_num_dims; ++d) {
              if(pool_offset[d] + pool_pos[d] < 0
                 || pool_offset[d] + pool_pos[d] >= m_input_dims[d]) {
                valid_pos = false;
                break;
              }
              input_pos *= m_input_dims[d];
              input_pos += pool_offset[d] + pool_pos[d];
            }

            // Check if pool entry is larger than previous
            DataType value = valid_pos ? input_channel.Get(input_pos, 0) : 0.0;
            if(value > max_value) {
              if(valid_pos) {
                max_value = value;
                max_input_pos = input_pos;
              }
              else {
                max_input_pos = -1;
              }
            }

            // Move to next pool entry
            ++pool_pos[m_num_dims-1];
            for(int d = m_num_dims - 1; d > 0; --d) {
              if(pool_pos[d] >= m_pool_dims[d]) {
                pool_pos[d] = 0;
                ++pool_pos[d-1];
              }
            }

          }

          // Propagate error signal
          if(max_input_pos >= 0) {
            error_signal_channel.Set(max_input_pos, 0,
                                     prev_error_signal_channel.Get(output_pos,
                                                                   0));
          }

          // Move to next output entry
          ++output_pos;

          // Move to next pool offset
          pool_offset[m_num_dims-1] += m_pool_strides[m_num_dims-1];
          for(int d = m_num_dims - 1; d > 0; --d) {
            if(pool_offset[d] + m_pool_dims[d] > m_input_dims[d] + m_pool_pads[d]) {
              pool_offset[d] = -m_pool_pads[d];
              pool_offset[d-1] += m_pool_strides[d-1];
            }
            
          }

        }

      }

    }

  }
  
}

bool pooling_layer::update()
{
  return true;
}
