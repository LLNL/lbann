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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_layer_convolutional.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/utils/lbann_random.hpp"

using namespace std;
using namespace El;
using namespace lbann;

convolutional_layer::convolutional_layer(const uint index,
                                         const int num_dims,
                                         const int num_input_channels,
                                         const int* input_dims,
                                         const int num_output_channels,
                                         const int* filter_dims,
                                         const int* conv_pads,
                                         const int* conv_strides,
                                         const uint mini_batch_size,
                                         const activation_type activation,
                                         const weight_initialization init,
                                         lbann_comm* comm,
                                         Optimizer* optimizer,
                                         std::vector<regularizer*> regs,
                                         cudnn::cudnn_manager* cudnn)
  : Layer(index, comm, optimizer, mini_batch_size, activation, regs),
    m_weight_initialization(init),
    m_num_dims(num_dims),
    m_num_input_channels(num_input_channels),
    m_num_output_channels(num_output_channels)
{

  m_type = layer_type::convolutional;

  // Initialize input dimensions and convolution parameters
  m_input_dims.resize(num_dims);
  m_filter_dims.resize(num_dims);
  m_filter_size = num_input_channels*num_output_channels;
  m_conv_pads.resize(num_dims);
  m_conv_strides.resize(num_dims);
  for(int i=0; i<num_dims; ++i) {
    m_input_dims[i] = input_dims[i];
    m_filter_dims[i] = filter_dims[i];
    m_filter_size *= filter_dims[i];
    m_conv_pads[i] = conv_pads[i];
    m_conv_strides[i] = conv_strides[i];
  }

  // Calculate output dimensions
  m_output_dims.resize(num_dims);
  NumNeurons = num_output_channels;
  for(int i=0; i<num_dims; ++i) {
    m_output_dims[i] = input_dims[i]+2*conv_pads[i]-filter_dims[i]+1;
    m_output_dims[i] = (m_output_dims[i]+conv_strides[i]-1)/conv_strides[i];
    NumNeurons *= m_output_dims[i];
  }
  
  // Matrices should be in Star,Star and Star,VC distributions
  delete m_weights;
  delete m_weights_gradient;
  delete m_weighted_sum;
  delete m_prev_activations;
  delete m_activations;
  delete m_prev_error_signal;
  delete m_error_signal;
  m_weights             = new StarMat(comm->get_model_grid());
  m_weights_gradient    = new StarMat(comm->get_model_grid());
  m_weighted_sum        = new StarVCMat(comm->get_model_grid());
  m_prev_activations    = new StarVCMat(comm->get_model_grid());
  m_activations         = new StarVCMat(comm->get_model_grid());
  m_prev_error_signal   = new StarVCMat(comm->get_model_grid());
  m_error_signal        = new StarVCMat(comm->get_model_grid());

  // Matrix views should be in Star,Star and Star,VC distributions
  delete m_weighted_sum_v;
  delete m_prev_activations_v;
  delete m_activations_v;
  delete m_prev_error_signal_v;
  delete m_error_signal_v;
  m_weighted_sum_v      = new StarVCMat(comm->get_model_grid());
  m_prev_activations_v  = new StarVCMat(comm->get_model_grid());
  m_activations_v       = new StarVCMat(comm->get_model_grid());
  m_prev_error_signal_v = new StarVCMat(comm->get_model_grid());
  m_error_signal_v      = new StarVCMat(comm->get_model_grid());

  // Initialize cuDNN convolutional layer
  m_cudnn_layer = NULL;
#ifdef __LIB_CUDNN
  if(cudnn)
    m_cudnn_layer
      = new cudnn::cudnn_convolutional_layer(num_dims,
                                             num_input_channels,
                                             num_output_channels,
                                             input_dims,
                                             filter_dims,
                                             conv_pads,
                                             conv_strides,
                                             m_mini_batch_size,
                                             cudnn);
#endif // __LIB_CUDNN

}

convolutional_layer::~convolutional_layer()
{
#ifdef __LIB_CUDNN
  delete m_cudnn_layer;
#endif // __LIB_CUDNN
}

void convolutional_layer::setup(const int num_prev_neurons)
{
  Layer::setup(num_prev_neurons);

#ifdef __LIB_CUDNN
  if(m_cudnn_layer) {
    // Setup cuDNN convolutional layer
    m_cudnn_layer->setup();

    // Check for errors
    if(NumNeurons != m_cudnn_layer->m_dst_size)
      throw lbann_exception("lbann_layer_convolutional: unexpected number of neurons");

    // Get output dimensions
    for(int i=0; i<m_num_dims; ++i)
      m_output_dims[i] = m_cudnn_layer->m_dst_dims[i+2];
  }
#endif // __LIB_CUDNN

  // Check if input dimensions are valid
  int num_inputs = m_num_input_channels;
  for(int i=0; i<m_num_dims; ++i)
    num_inputs *= m_input_dims[i];
  if(num_inputs != num_prev_neurons) {
    throw lbann_exception("lbann_layer_convolutional: unexpected number of input neurons");
  }

  // Initialize optimizer
  if(optimizer)
    optimizer->setup(1, m_filter_size+NumNeurons);

  // Initialize weight-bias matrix
  Zeros(*m_weights, m_filter_size+NumNeurons, 1);

  // Initialize filters
  StarMat filters;
  View(filters, *m_weights, IR(0,m_filter_size), ALL);
  Int fan_in = m_filter_size / m_num_output_channels;
  Int fan_out = m_filter_size / m_num_input_channels;
  switch(m_weight_initialization) {
  case weight_initialization::uniform:
    uniform_fill(filters, filters.Height(), filters.Width(),
                 DataType(0), DataType(1));
    break;
  case weight_initialization::normal:
    gaussian_fill(filters, filters.Height(), filters.Width(),
                  DataType(0), DataType(1));
    break;
  case weight_initialization::glorot_normal: {
    const DataType var = 2.0 / (fan_in + fan_out);
    gaussian_fill(filters, filters.Height(), filters.Width(),
                  DataType(0), sqrt(var));
    break;
  }
  case weight_initialization::glorot_uniform: {
    const DataType var = 2.0 / (fan_in + fan_out);
    uniform_fill(filters, filters.Height(), filters.Width(),
                 DataType(0), sqrt(3*var));
    break;
  }
  case weight_initialization::he_normal: {
    const DataType var = 1.0 / fan_in;
    gaussian_fill(filters, filters.Height(), filters.Width(),
                  DataType(0), sqrt(var));
    break;
  }
  case weight_initialization::he_uniform: {
    const DataType var = 1.0 / fan_in;
    uniform_fill(filters, filters.Height(), filters.Width(),
                 DataType(0), sqrt(3*var));
    break;
  }
  case weight_initialization::zero: // Zero initialization is default
  default:
    Zero(filters);
    break;
  }
  
  // Initialize matrices
  Zeros(*m_weights_gradient, m_filter_size+NumNeurons, 1);
  Ones(*m_weighted_sum, NumNeurons, m_mini_batch_size);
  Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size);
  Zeros(*m_error_signal, num_prev_neurons, m_mini_batch_size);
  Ones(*m_activations, NumNeurons, m_mini_batch_size);

}

void lbann::convolutional_layer::fp_linearity() {
  
  // Get local matrices
  /// @todo Rename variables to something more descriptive
  const Mat& WBLocal = m_weights->LockedMatrix();
  const Mat& XLocal = m_prev_activations_v->LockedMatrix();
  Mat& ZLocal = m_weighted_sum_v->Matrix();
  Mat& YLocal = m_activations_v->Matrix();
  Mat filters = WBLocal(IR(0,m_filter_size),ALL);
  Mat bias = WBLocal(IR(m_filter_size,END),ALL);

  // Apply convolution on local data samples
  if(m_cudnn_layer) {
#ifdef __LIB_CUDNN
    // cuDNN convolutional layer forward pass
    m_cudnn_layer->forward(XLocal, filters, bias, ZLocal);
#else
    throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#endif
  }
  else {

    ////////////////////////////////////////////////////////////
    // CPU implementation of convolutional layer forward pass
    // Note: explicitly constructs a dense convolution matrix
    /// @todo Write a more efficient implementation
    ////////////////////////////////////////////////////////////

    // Apply bias to each sample in mini-batch
    for(int sample = 0; sample < XLocal.Width(); ++sample) {
      Mat output_sample = ZLocal(IR(0,NumNeurons), IR(sample));
      Copy(bias, output_sample);
    }

    // Initialize convolution matrix
    Mat convolution_matrix;
    Zeros(convolution_matrix, NumNeurons, XLocal.Height());

    // Iterate through filters
    int row = 0;
    for(int output_channel = 0;
        output_channel < m_num_output_channels;
        ++output_channel) {
      const int current_filter_size = m_filter_size / m_num_output_channels;
      const Mat filter = filters(IR(output_channel*current_filter_size,
                                    (output_channel+1)*current_filter_size),
                                 ALL);

      // Iterate through filter offsets
      // Note: each offset corresponds to a row of the convolution matrix
      std::vector<int> filter_offset(m_num_dims);
      for(int d = 0; d < m_num_dims; ++d) {
        filter_offset[d] = -m_conv_pads[d];
      }
      while(filter_offset[0] + m_filter_dims[0] <= m_input_dims[0] + m_conv_pads[0]) {

        // Iterate through filter entries
        // Note: each filter entry corresponds to entry of convolution matrix
        std::vector<int> filter_pos(m_num_dims, 0);
        while(filter_pos[0] < m_filter_dims[0]) {

          // Get convolution matrix entry corresponding to filter entry
          int col = 0;
          int filter_flat_pos = 0;
          bool valid_pos = true;
          for(int d = 0; d < m_num_dims; ++d) {
            if(filter_offset[d] + filter_pos[d] < 0
               || filter_offset[d] + filter_pos[d] >= m_input_dims[d]) {
              valid_pos = false;
              break;
            }
            col *= m_input_dims[d];
            col += filter_offset[d] + filter_pos[d];
            filter_flat_pos *= m_filter_dims[d];
            filter_flat_pos += filter_pos[d];
          }

          if(valid_pos) {

            // Iterate through input channels
            for(int input_channel = 0;
                input_channel < m_num_input_channels;
                ++input_channel) {

              // Set convolution matrix entry
              const DataType w = filter.Get(filter_flat_pos, 0);
              convolution_matrix.Set(row, col, w);

              // Move to next convolution matrix entry
              col += XLocal.Height() / m_num_input_channels;
              filter_flat_pos += current_filter_size / m_num_input_channels;

            }

          }
          
          // Move to next position in filter
          ++filter_pos[m_num_dims-1];
          for(int d = m_num_dims - 1; d > 0; --d) {
            if(filter_pos[d] >= m_filter_dims[d]) {
              filter_pos[d] = 0;
              ++filter_pos[d-1];
            }
          }
          
        }

        // Move to next filter offset
        filter_offset[m_num_dims-1] += m_conv_strides[m_num_dims-1];
        for(int d = m_num_dims - 1; d > 0; --d) {
          if(filter_offset[d] + m_filter_dims[d] > m_input_dims[d] + m_conv_pads[d]) {
            filter_offset[d] = -m_conv_pads[d];
            filter_offset[d-1] += m_conv_strides[d-1];
          }
        }

        // Move to next row in convolution matrix
        ++row;

      }
      
    }

    // Apply convolution matrix
    Gemm(NORMAL, NORMAL,
         DataType(1), convolution_matrix, XLocal,
         DataType(1), ZLocal);

  }

  // Z and Y are identical after fp linearity step
  Copy(ZLocal, YLocal);

}

void lbann::convolutional_layer::bp_linearity() {

  // Get local matrices
  const Mat& input_local = m_prev_activations_v->LockedMatrix();
  const Mat& filters_local = m_weights->LockedMatrix()(IR(0,m_filter_size),ALL);
  const Mat& prev_error_signal_local = m_prev_error_signal_v->LockedMatrix();
  Mat filters_gradient_local = m_weights_gradient->Matrix()(IR(0,m_filter_size),ALL);
  Mat bias_gradient_local = m_weights_gradient->Matrix()(IR(m_filter_size,END),ALL);
  Mat& error_signal_local = m_error_signal_v->Matrix();

  // Compute gradients on local data samples
  if(m_cudnn_layer) {
#ifdef __LIB_CUDNN
    m_cudnn_layer->backward(input_local,
                            filters_local,
                            prev_error_signal_local,
                            filters_gradient_local,
                            bias_gradient_local,
                            error_signal_local);
#else
    throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#endif
  }
  else {

    ////////////////////////////////////////////////////////////
    // CPU implementation of convolutional layer backward pass
    // Note: explicitly constructs a dense convolution matrix
    /// @todo Write a more efficient implementation
    ////////////////////////////////////////////////////////////

    //////////////////////////////////////////////
    // Construct convolution matrix
    //////////////////////////////////////////////

    // Initialize convolution matrix
    Mat convolution_matrix;
    Zeros(convolution_matrix, NumNeurons, input_local.Height());

    // Iterate through filters
    int row = 0;
    for(int output_channel = 0;
        output_channel < m_num_output_channels;
        ++output_channel) {
      const int current_filter_size = m_filter_size / m_num_output_channels;
      const Mat filter = filters_local(IR(output_channel*current_filter_size,
                                          (output_channel+1)*current_filter_size),
                                       ALL);

      // Iterate through filter offsets
      // Note: each offset corresponds to a row of the convolution matrix
      std::vector<int> filter_offset(m_num_dims);
      for(int d = 0; d < m_num_dims; ++d) {
        filter_offset[d] = -m_conv_pads[d];
      }
      while(filter_offset[0] + m_filter_dims[0] <= m_input_dims[0] + m_conv_pads[0]) {

        // Iterate through filter entries
        // Note: each filter entry corresponds to entry of convolution matrix
        std::vector<int> filter_pos(m_num_dims, 0);
        while(filter_pos[0] < m_filter_dims[0]) {

          // Get convolution matrix entry corresponding to filter entry
          int col = 0;
          int filter_flat_pos = 0;
          bool valid_pos = true;
          for(int d = 0; d < m_num_dims; ++d) {
            if(filter_offset[d] + filter_pos[d] < 0
               || filter_offset[d] + filter_pos[d] >= m_input_dims[d]) {
              valid_pos = false;
              break;
            }
            col *= m_input_dims[d];
            col += filter_offset[d] + filter_pos[d];
            filter_flat_pos *= m_filter_dims[d];
            filter_flat_pos += filter_pos[d];
          }

          if(valid_pos) {

            // Iterate through input channels
            for(int input_channel = 0;
                input_channel < m_num_input_channels;
                ++input_channel) {

              // Set convolution matrix entry
              const DataType w = filter.Get(filter_flat_pos, 0);
              convolution_matrix.Set(row, col, w);

              // Move to next convolution matrix entry
              col += input_local.Height()  / m_num_input_channels;
              filter_flat_pos += current_filter_size / m_num_input_channels;

            }

          }
          
          // Move to next position in filter
          ++filter_pos[m_num_dims-1];
          for(int d = m_num_dims - 1; d > 0; --d) {
            if(filter_pos[d] >= m_filter_dims[d]) {
              filter_pos[d] = 0;
              ++filter_pos[d-1];
            }
          }
          
        }

        // Move filter to next position
        filter_offset[m_num_dims-1] += m_conv_strides[m_num_dims-1];
        for(int d = m_num_dims - 1; d > 0; --d) {
          if(filter_offset[d] + m_filter_dims[d] > m_input_dims[d] + m_conv_pads[d]) {
            filter_offset[d] = -m_conv_pads[d];
            filter_offset[d-1] += m_conv_strides[d-1];
          }
        }

        // Move to next row in convolution matrix
        ++row;

      }
      
    }

    //////////////////////////////////////////////
    // Compute error signal
    //////////////////////////////////////////////

    // Compute error signal
    Gemm(TRANSPOSE, NORMAL,
         DataType(1), convolution_matrix, prev_error_signal_local,
         DataType(0), error_signal_local);

    // Compute bias gradient
    Mat ones;
    Ones(ones, input_local.Width(), Int(1));
    Gemv(NORMAL, DataType(1.0), prev_error_signal_local, ones,
         DataType(0.0), bias_gradient_local);

    // Compute error signal w.r.t. convolution matrix
    Mat conv_error_signal(convolution_matrix.Height(),
                          convolution_matrix.Width());
    Gemm(NORMAL, TRANSPOSE,
         DataType(1), prev_error_signal_local, input_local,
         DataType(0), conv_error_signal);

    // Initialize filter gradient
    Zero(filters_gradient_local);

    // Iterate through filters
    row = 0;
    for(int output_channel = 0;
        output_channel < m_num_output_channels;
        ++output_channel) {
      const int current_filter_size = m_filter_size / m_num_output_channels;
      Mat filter_gradient
        = filters_gradient_local(IR(output_channel*current_filter_size,
                                    (output_channel+1)*current_filter_size),
                                 ALL);

      // Iterate through filter offsets
      // Note: each offset corresponds to a row of the convolution matrix
      std::vector<int> filter_offset(m_num_dims);
      for(int d = 0; d < m_num_dims; ++d) {
        filter_offset[d] = -m_conv_pads[d];
      }
      while(filter_offset[0] + m_filter_dims[0] <= m_input_dims[0] + m_conv_pads[0]) {

        // Iterate through filter entries
        // Note: each filter entry corresponds to entry of convolution matrix
        std::vector<int> filter_pos(m_num_dims, 0);
        while(filter_pos[0] < m_filter_dims[0]) {

          // Get convolution matrix entry corresponding to filter entry
          int col = 0;
          int filter_flat_pos = 0;
          bool valid_pos = true;
          for(int d = 0; d < m_num_dims; ++d) {
            if(filter_offset[d] + filter_pos[d] < 0
               || filter_offset[d] + filter_pos[d] >= m_input_dims[d]) {
              valid_pos = false;
              break;
            }
            col *= m_input_dims[d];
            col += filter_offset[d] + filter_pos[d];
            filter_flat_pos *= m_filter_dims[d];
            filter_flat_pos += filter_pos[d];
          }

          if(valid_pos) {

            // Iterate through input channels
            for(int input_channel = 0;
                input_channel < m_num_input_channels;
                ++input_channel) {

              // Get error signal for convolution matrix entry
              filter_gradient.Update(filter_flat_pos, 0,
                                     conv_error_signal.Get(row, col));

              // Move to next convolution matrix entry
              col += input_local.Height() / m_num_input_channels;
              filter_flat_pos += current_filter_size / m_num_input_channels;

            }

          }
          
          // Move to next position in filter
          ++filter_pos[m_num_dims-1];
          for(int d = m_num_dims - 1; d > 0; --d) {
            if(filter_pos[d] >= m_filter_dims[d]) {
              filter_pos[d] = 0;
              ++filter_pos[d-1];
            }
          }
          
        }

        // Move filter to next position
        filter_offset[m_num_dims-1] += m_conv_strides[m_num_dims-1];
        for(int d = m_num_dims - 1; d > 0; --d) {
          if(filter_offset[d] + m_filter_dims[d] > m_input_dims[d] + m_conv_pads[d]) {
            filter_offset[d] = -m_conv_pads[d];
            filter_offset[d-1] += m_conv_strides[d-1];
          }
        }

        // Move to next row in convolution matrix
        ++row;

      }
      
    }

  }

  // Obtain filter gradient with reduction and scaling
  AllReduce(*m_weights_gradient, m_weights_gradient->DistComm());
  *m_weights_gradient *= 1.0/get_effective_minibatch_size();

}

bool convolutional_layer::update()
{
  if(m_execution_mode == execution_mode::training) {
    optimizer->update_weight_bias_matrix(*m_weights_gradient, *m_weights);
  }
  return true;
}

