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
#include "lbann/utils/lbann_timer.hpp"
#include "lbann/utils/lbann_im2col.hpp"

using namespace std;
using namespace El;
using namespace lbann;

#ifdef __LIB_CUDNN
/// Get cuDNN activation mode
cudnnActivationMode_t get_cudnn_activation_mode(activation_type type) {
  switch(type) {
  case activation_type::SIGMOID: return CUDNN_ACTIVATION_SIGMOID;
  case activation_type::RELU:    return CUDNN_ACTIVATION_RELU;
  case activation_type::TANH:    return CUDNN_ACTIVATION_TANH;
  default:
    throw lbann_exception("convolutional_layer: invalid activation type for cuDNN");
  }
}
#endif // #ifdef __LIB_CUDNN

convolutional_layer::convolutional_layer(const uint index,
                                         const Int num_dims,
                                         const Int num_input_channels,
                                         const Int* input_dims,
                                         const Int num_output_channels,
                                         const Int* filter_dims,
                                         const Int* conv_pads,
                                         const Int* conv_strides,
                                         const Int mini_batch_size,
                                         const activation_type activation,
                                         const weight_initialization init,
                                         lbann_comm* comm,
                                         optimizer* opt,
                                         std::vector<regularizer*> regs,
                                         cudnn::cudnn_manager* cudnn)
  : Layer(data_layout::DATA_PARALLEL, index, comm, opt, mini_batch_size, activation, regs),
    m_weight_initialization(init),
    m_num_dims(num_dims),
    m_num_input_channels(num_input_channels),
    m_num_output_channels(num_output_channels)
{
  m_type = layer_type::convolution;

  // Initialize input dimensions and convolution parameters
  m_input_dims.resize(m_num_dims);
  m_filter_dims.resize(m_num_dims);
  m_filter_size = m_num_input_channels*m_num_output_channels;
  m_conv_pads.resize(num_dims);
  m_conv_strides.resize(num_dims);
  for(Int i=0; i<num_dims; ++i) {
    m_input_dims[i] = input_dims[i];
    m_filter_dims[i] = filter_dims[i];
    m_filter_size *= filter_dims[i];
    m_conv_pads[i] = conv_pads[i];
    m_conv_strides[i] = conv_strides[i];
  }

  // Calculate output dimensions
  m_output_dims.resize(num_dims);
  NumNeurons = num_output_channels;
  for(Int i=0; i<num_dims; ++i) {
    m_output_dims[i] = input_dims[i]+2*conv_pads[i]-filter_dims[i]+1;
    m_output_dims[i] = (m_output_dims[i]+conv_strides[i]-1)/conv_strides[i];
    NumNeurons *= m_output_dims[i];
  }

#ifdef __LIB_CUDNN
  m_weights_gradient_per_gpu = StarMat(comm->get_model_grid());
#endif // #ifdef __LIB_CUDNN

#ifdef __LIB_CUDNN

  to_pin_fwd = false;
  to_pin_bwd = false;
  is_pinned_fwd = false;
  is_pinned_bwd = false;

  // Initialize cuDNN objects
  m_input_desc = NULL;
  m_output_desc = NULL;
  m_bias_desc = NULL;
  m_filter_desc = NULL;
  m_convolution_desc = NULL;
  m_activation_desc = NULL;
  m_forward_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  m_backward_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  m_backward_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

  // Set parameters for GPU implementation
  if(cudnn) {
    m_using_gpus = true;
    m_cudnn = cudnn;
    const Int num_gpus = m_cudnn->get_num_gpus();
    const Int num_processes = comm->get_procs_per_model();
    const Int local_mini_batch_size = (mini_batch_size + num_processes - 1) / num_processes;
    m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;
    
    // Default behavior set to pin memory blocks used by cuDNN
    pin_mem();
  }

#endif // #ifdef __LIB_CUDNN

}

convolutional_layer::~convolutional_layer()
{
#ifdef __LIB_CUDNN
  if(m_using_gpus) {

    // Destroy cuDNN objects
    if(m_input_desc)
      checkCUDNN(cudnnDestroyTensorDescriptor(m_input_desc));
    if(m_output_desc)
      checkCUDNN(cudnnDestroyTensorDescriptor(m_output_desc));
    if(m_bias_desc)
      checkCUDNN(cudnnDestroyTensorDescriptor(m_bias_desc));
    if(m_filter_desc)
      checkCUDNN(cudnnDestroyFilterDescriptor(m_filter_desc));
    if(m_convolution_desc)
      checkCUDNN(cudnnDestroyConvolutionDescriptor(m_convolution_desc));
    if(m_activation_desc)
      checkCUDNN(cudnnDestroyActivationDescriptor(m_activation_desc));

    // Unpin pinned memory blocks
    unpin_mem();

    // Deallocate GPU memory
    m_cudnn->deallocate_on_gpus(m_weights_d);
    m_cudnn->deallocate_on_gpus(m_weighted_sum_d);
    m_cudnn->deallocate_on_gpus(m_activations_d);
    m_cudnn->deallocate_on_gpus(m_weights_gradient_d);
    m_cudnn->deallocate_on_gpus(m_error_signal_d);
    if(!m_prev_layer_using_gpus)
      m_cudnn->deallocate_on_gpus(m_prev_activations_d);
    if(!m_next_layer_using_gpus)
      m_cudnn->deallocate_on_gpus(m_prev_error_signal_d);

  }
#endif // #ifdef __LIB_CUDNN
}

void convolutional_layer::setup(const int num_prev_neurons)
{
  Layer::setup(num_prev_neurons);

#ifdef __LIB_CUDNN
  // Setup cuDNN objects
  if(m_using_gpus) {
    setup_gpu();
  }
#endif // #ifdef __LIB_CUDNN

#ifdef LBANN_DEBUG
  // Check if input dimensions are valid
  Int num_inputs = m_num_input_channels;
  for(Int i=0; i<m_num_dims; ++i)
    num_inputs *= m_input_dims[i];
  if(num_inputs != m_num_prev_neurons) {
    throw lbann_exception("lbann_layer_convolutional: unexpected number of input neurons");
  }
#endif // #ifdef LBANN_DEBUG

  // Initialize matrices
  Zeros(*m_weights, m_filter_size+m_num_output_channels, 1);
  Zeros(*m_weights_gradient, m_filter_size+m_num_output_channels, 1);
#ifdef __LIB_CUDNN
  if(m_using_gpus) {
    Zeros(m_weights_gradient_per_gpu,
          m_filter_size+m_num_output_channels,
          m_cudnn->get_num_gpus());
  }
#endif // #ifdef __LIB_CUDNN
  Zeros(*m_prev_activations, m_num_prev_neurons, m_mini_batch_size);
  Zeros(*m_error_signal, m_num_prev_neurons, m_mini_batch_size);
  Zeros(*m_activations, NumNeurons, m_mini_batch_size);
  Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size);
  Zeros(*m_weighted_sum, NumNeurons, m_mini_batch_size);

  // Initialize filters
  StarMat filter;
  View(filter, *m_weights, IR(0,m_filter_size), ALL);
  const Int fan_in = m_filter_size / m_num_output_channels;
  const Int fan_out = m_filter_size / m_num_input_channels;
  initialize_matrix(filter, m_weight_initialization, fan_in, fan_out);

  // Initialize optimizer
  if(m_optimizer != NULL)
    m_optimizer->setup(m_weights);
  
}

void lbann::convolutional_layer::setup_gpu()
{
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#else

  // Get device properties
  cudaDeviceProp device_props;
  checkCUDA(cudaGetDeviceProperties(&device_props, 0));

  // Initialize descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&m_input_desc));
  checkCUDNN(cudnnCreateTensorDescriptor(&m_output_desc));
  checkCUDNN(cudnnCreateTensorDescriptor(&m_bias_desc));
  checkCUDNN(cudnnCreateFilterDescriptor(&m_filter_desc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convolution_desc));
  checkCUDNN(cudnnCreateActivationDescriptor(&m_activation_desc));

  // Set input tensor descriptor
  std::vector<int> input_dims(m_num_dims+2);
  input_dims[0] = m_mini_batch_size_per_gpu;
  input_dims[1] = m_num_input_channels;
  for(Int i=0; i<m_num_dims; ++i)
    input_dims[i+2] = m_input_dims[i];
  std::vector<int> input_strides(m_num_dims+2);
  input_strides[m_num_dims + 1]  = 1;
  for(Int i=m_num_dims; i>=0; --i)
    input_strides[i] = input_strides[i+1] * input_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_input_desc,
                                        m_cudnn->get_cudnn_data_type(),
                                        m_num_dims+2,
                                        input_dims.data(),
                                        input_strides.data()));

  // Set filter descriptor
  std::vector<int> filter_dims(m_num_dims+2);
  filter_dims[0] = m_num_output_channels;
  filter_dims[1] = m_num_input_channels;
  for(Int i=0; i<m_num_dims; ++i)
    filter_dims[i+2] = m_filter_dims[i];
  checkCUDNN(cudnnSetFilterNdDescriptor(m_filter_desc,
                                        m_cudnn->get_cudnn_data_type(),
                                        CUDNN_TENSOR_NCHW,
                                        m_num_dims+2,
                                        filter_dims.data()));

  // Set convolution descriptor
  // Note: upscales are not supported as of cuDNN v5.1
  std::vector<int> conv_upscales(m_num_dims, 1);
  std::vector<int> conv_pads(m_num_dims);
  for(Int i=0; i<m_num_dims; ++i)
    conv_pads[i] = m_conv_pads[i];
  std::vector<int> conv_strides(m_num_dims);
  for(Int i=0; i<m_num_dims; ++i)
    conv_strides[i] = m_conv_strides[i];
  checkCUDNN(cudnnSetConvolutionNdDescriptor(m_convolution_desc,
                                             m_num_dims,
                                             conv_pads.data(),
                                             conv_strides.data(),
                                             conv_upscales.data(),
                                             CUDNN_CONVOLUTION,
                                             m_cudnn->get_cudnn_data_type()));

  // Set output tensor descriptor
  std::vector<int> output_dims(m_num_dims+2);
#ifdef LBANN_DEBUG
  checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(m_convolution_desc,
                                                   m_input_desc,
                                                   m_filter_desc,
                                                   m_num_dims+2,
                                                   output_dims.data()));
  if(output_dims[0] != m_mini_batch_size_per_gpu)
    throw lbann_exception("lbann_layer_convolutional: invalid output dimensions");
  if(output_dims[1] != m_num_output_channels)
    throw lbann_exception("lbann_layer_convolutional: invalid output dimensions");
  for(Int i=0; i<m_num_dims; ++i) {
    if(output_dims[i+2] != m_output_dims[i]) {
      throw lbann_exception("lbann_layer_convolutional: invalid output dimensions");
    }
  }
#else
  output_dims[0] = m_mini_batch_size_per_gpu;
  output_dims[1] = m_num_output_channels;
  for(Int i=0; i<m_num_dims; ++i)
    output_dims[i+2] = m_output_dims[i];
#endif // #ifdef LBANN_DEBUG
  std::vector<int> output_strides(m_num_dims+2);
  output_strides[m_num_dims + 1]  = 1;
  for(Int i=m_num_dims; i>=0; --i)
    output_strides[i] = output_strides[i+1] * output_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_output_desc,
                                        m_cudnn->get_cudnn_data_type(),
                                        m_num_dims+2,
                                        output_dims.data(),
                                        output_strides.data()));

  // Set output tensor descriptor
  std::vector<int> bias_dims(m_num_dims+2, 1);
  bias_dims[1] = m_num_output_channels;
  std::vector<int> bias_strides(m_num_dims+2, 1);
  bias_strides[0] = bias_dims[1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_bias_desc,
                                        m_cudnn->get_cudnn_data_type(),
                                        m_num_dims+2,
                                        bias_dims.data(),
                                        bias_strides.data()));
  
  // Choose algorithms
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(m_cudnn->get_handle(),
                                                 m_input_desc,
                                                 m_filter_desc,
                                                 m_convolution_desc,
                                                 m_output_desc,
                                                 CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                 device_props.totalGlobalMem/2,
                                                 &m_forward_algo));
  checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(m_cudnn->get_handle(),
                                                        m_input_desc,
                                                        m_output_desc,
                                                        m_convolution_desc,
                                                        m_filter_desc,
                                                        CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                        device_props.totalGlobalMem/2,
                                                        &m_backward_filter_algo));
  checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(m_cudnn->get_handle(),
                                                      m_filter_desc,
                                                      m_output_desc,
                                                      m_convolution_desc,
                                                      m_input_desc,
                                                      CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                                      device_props.totalGlobalMem/2,
                                                      &m_backward_data_algo));

  // Initialize work space
  size_t max_work_space = 0;
  size_t required_work_space;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_cudnn->get_handle(),
                                                     m_input_desc,
                                                     m_filter_desc,
                                                     m_convolution_desc,
                                                     m_output_desc,
                                                     m_forward_algo,
                                                     &required_work_space));
  max_work_space = Max(max_work_space, required_work_space);
  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(m_cudnn->get_handle(),
                                                            m_input_desc,
                                                            m_output_desc,
                                                            m_convolution_desc,
                                                            m_filter_desc,
                                                            m_backward_filter_algo,
                                                            &required_work_space));
  max_work_space = Max(max_work_space, required_work_space);
  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(m_cudnn->get_handle(),
                                                          m_filter_desc,
                                                          m_output_desc,
                                                          m_convolution_desc,
                                                          m_input_desc,
                                                          m_backward_data_algo,
                                                          &required_work_space));
  max_work_space = Max(max_work_space, required_work_space);
  for(Int i=0; i<m_cudnn->get_num_gpus(); ++i) {
    if(max_work_space > m_cudnn->get_work_space_size(i))
      m_cudnn->set_work_space_size(i, max_work_space);
  }

  // Set activation descriptor
  if(m_activation_type != activation_type::ID) {
    checkCUDNN(cudnnSetActivationDescriptor(m_activation_desc,
                                            get_cudnn_activation_mode(m_activation_type),
                                            CUDNN_PROPAGATE_NAN,
                                            0.0));
  }

  // Allocate GPU memory
  m_cudnn->allocate_on_gpus(m_weights_d,
                            m_filter_size+m_num_output_channels,
                            1);
  m_cudnn->allocate_on_gpus(m_weights_gradient_d,
                            m_filter_size+m_num_output_channels,
                            1);
  m_cudnn->allocate_on_gpus(m_weighted_sum_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_activations_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_error_signal_d,
                            m_num_prev_neurons,
                            m_mini_batch_size_per_gpu);
  if(!m_prev_layer_using_gpus) {
    m_cudnn->allocate_on_gpus(m_prev_activations_d,
                              m_num_prev_neurons,
                              m_mini_batch_size_per_gpu);
  }
  if(!m_next_layer_using_gpus) {
    m_cudnn->allocate_on_gpus(m_prev_error_signal_d,
                              NumNeurons,
                              m_mini_batch_size_per_gpu);
  }

#endif // #ifdef __LIB_CUDNN
}

/**
 * \brief Set to pin the memory blocks used by cudnn.
 * \details The actual pinning occurs at the beginning of next fp_linearity() call.
 *          No effect when cudnn is not employed.
 */
void lbann::convolutional_layer::pin_mem(void) {
#ifdef __LIB_CUDNN
  to_pin_fwd = true;
  to_pin_bwd = true;
#endif
}

/**
 * \brief unpin the memory blocks pinned for cudnn
 * \details The effect is immediate.
 */
void lbann::convolutional_layer::unpin_mem(void) {
#ifdef __LIB_CUDNN
  to_pin_fwd = false;
  to_pin_bwd = false;
  unpin_memory_blocks_fwd();
  unpin_memory_blocks_bwd();
#endif
}

void lbann::convolutional_layer::pin_memory_blocks_fwd(void)
{
#ifdef __LIB_CUDNN
  size_t total_size = 0u;
  total_size += m_cudnn->pin_memory_block(m_weights);
  if(!m_prev_layer_using_gpus) {
    total_size += m_cudnn->pin_memory_block(m_prev_activations);
  }
  if(!m_next_layer_using_gpus)
    total_size += m_cudnn->pin_memory_block(m_activations);
  //std::cout << total_size << " bytes pinned by convolutional layer " 
  //          << get_index() << " forward " << std::endl;

  is_pinned_fwd = true;
#endif // #ifdef __LIB_CUDNN
}

void lbann::convolutional_layer::pin_memory_blocks_bwd(void)
{
#ifdef __LIB_CUDNN
  size_t total_size = 0u;
  total_size += m_cudnn->pin_memory_block(&m_weights_gradient_per_gpu);
  if(!m_next_layer_using_gpus) {
    total_size += m_cudnn->pin_memory_block(m_prev_error_signal);
  }
  if(!m_prev_layer_using_gpus)
    total_size += m_cudnn->pin_memory_block(m_error_signal);
  //std::cout << total_size << " bytes pinned by convolutional layer " 
  //          << get_index() << " backward " << std::endl;

  is_pinned_bwd = true;
#endif // #ifdef __LIB_CUDNN
}

void lbann::convolutional_layer::unpin_memory_blocks_fwd(void)
{
#ifdef __LIB_CUDNN
  m_cudnn->unpin_memory_block(m_weights);
  if(!m_prev_layer_using_gpus)
    m_cudnn->unpin_memory_block(m_prev_activations);
  if(!m_next_layer_using_gpus)
    m_cudnn->unpin_memory_block(m_activations);

  is_pinned_fwd = false;
#endif
}

void lbann::convolutional_layer::unpin_memory_blocks_bwd(void)
{
#ifdef __LIB_CUDNN
  m_cudnn->unpin_memory_block(&m_weights_gradient_per_gpu);
  if(!m_next_layer_using_gpus)
    m_cudnn->unpin_memory_block(m_prev_error_signal);
  if(!m_prev_layer_using_gpus)
    m_cudnn->unpin_memory_block(m_error_signal);

  is_pinned_bwd = false;
#endif
}

void lbann::convolutional_layer::forwardProp() {

  // Perform forward propagation
  Layer::forwardProp();

#ifdef __LIB_CUDNN
  // Pin memory blocks at the first step
  if(to_pin_fwd && !is_pinned_fwd)
    pin_memory_blocks_fwd();
#endif // #ifdef __LIB_CUDNN

}

void lbann::convolutional_layer::backProp() {

  // Perform backward propagation
  Layer::backProp();

#ifdef __LIB_CUDNN
  // Pin memory blocks at the first step
  if(to_pin_bwd && !is_pinned_bwd)
    pin_memory_blocks_bwd();
#endif // #ifdef __LIB_CUDNN

}

/// @todo CPU implementations for 1d and 3d
void lbann::convolutional_layer::fp_linearity() {
  if(m_using_gpus) {
    fp_linearity_gpu();
  }
  else {
    fp_linearity_cpu_gemm();
  }
}

/// @todo CPU implementations for 1d and 3d
void lbann::convolutional_layer::bp_linearity() {

  if(m_using_gpus) {
    bp_linearity_gpu();
  }
  else {
    bp_linearity_cpu_gemm();
  }

}

void lbann::convolutional_layer::fp_nonlinearity() {
  if(m_using_gpus) {
    fp_nonlinearity_gpu();
  }
  else {
    Layer::fp_nonlinearity();
  }
}

void lbann::convolutional_layer::bp_nonlinearity() {
  if(m_using_gpus) {
    bp_nonlinearity_gpu();
  }
  else {
    Layer::bp_nonlinearity();
  }
}

void lbann::convolutional_layer::fp_linearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Transfer filters and bias from CPU to GPUs
  m_cudnn->broadcast_to_gpus(m_weights_d, m_weights->LockedMatrix());

  // Perform convolution on each GPU
  const Int num_gpus = m_cudnn->get_num_gpus();
  for(Int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    checkCUDNN(cudnnSetStream(m_cudnn->get_handle(i),
                              m_cudnn->get_stream(i)));
    checkCUDNN(cudnnConvolutionForward(m_cudnn->get_handle(i),
                                       &one,
                                       m_input_desc,
                                       m_prev_activations_d[i],
                                       m_filter_desc,
                                       m_weights_d[i],
                                       m_convolution_desc,
                                       m_forward_algo,
                                       m_cudnn->get_work_space(i),
                                       m_cudnn->get_work_space_size(i),
                                       &zero,
                                       m_output_desc,
                                       m_weighted_sum_d[i]));
    checkCUDNN(cudnnAddTensor(m_cudnn->get_handle(i),
                              &one,
                              m_bias_desc,
                              m_weights_d[i] + m_filter_size,
                              &one,
                              m_output_desc,
                              m_weighted_sum_d[i]));
  }

  // Copy result to output matrix
  m_cudnn->copy_on_gpus(m_activations_d,
                        m_weighted_sum_d,
                        NumNeurons,
                        m_mini_batch_size_per_gpu);

#endif // #ifndef __LIB_CUDNN
}

void lbann::convolutional_layer::fp_nonlinearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Perform activation with each GPU
  if(m_activation_type != activation_type::ID) {
    const Int num_gpus = m_cudnn->get_num_gpus();
    for(Int i=0; i<num_gpus; ++i) {
      checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
      checkCUDNN(cudnnSetStream(m_cudnn->get_handle(i),
                                m_cudnn->get_stream(i)));
      checkCUDNN(cudnnActivationForward(m_cudnn->get_handle(i),
                                        m_activation_desc,
                                        &one,
                                        m_output_desc,
                                        m_activations_d[i],
                                        &zero,
                                        m_output_desc,
                                        m_activations_d[i]));
    }
  }

#endif // #ifndef __LIB_CUDNN
}  

void lbann::convolutional_layer::fp_linearity_cpu_direct() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& weights_local = m_weights->LockedMatrix();
  Mat& weighted_sum_local = m_weighted_sum_v->Matrix();
  Mat& activations_local = m_activations_v->Matrix();

  // Get filter and bias
  const Mat filter_local = LockedView(weights_local, IR(0,m_filter_size), ALL);
  const Mat bias_local = LockedView(weights_local, IR(m_filter_size,END), ALL);

  // Input, output, and filter entries are divided amongst channels
  const Int num_per_output_channel = NumNeurons / m_num_output_channels;
  const Int num_per_input_channel = m_num_prev_neurons / m_num_input_channels;
  const Int current_filter_size = m_filter_size / m_num_output_channels;
  const Int current_filter_size_per_input_channel = current_filter_size / m_num_input_channels;

  // Apply bias
  for(Int i=0; i<m_num_output_channels; ++i) {
    Mat weighted_sum_channel 
      = View(weighted_sum_local,
             IR(i*num_per_output_channel, (i+1)*num_per_output_channel),
             ALL);
    Fill(weighted_sum_channel, bias_local.Get(i,0));
  }

  ////////////////////////////////////////////////////////////////
  // Apply convolution
  // Note: We are implicitly applying a convolution matrix where
  //   filter_entry is the (output_index, input_index) entry.
  ////////////////////////////////////////////////////////////////

  // Iterate through samples, output channels, and input channels
#pragma omp parallel for
  for(Int output_channel = 0;
      output_channel < m_num_output_channels;
      ++output_channel) {
    std::vector<Int> filter_offsets(m_num_dims);
    std::vector<Int> filter_pos(m_num_dims);
    for(Int input_channel = 0;
        input_channel < m_num_input_channels;
        ++input_channel) {
      for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {

        // Iterate through output entries in current output channel
        // Note: each output entry corresponds to a different offset
        //   of the convolutional kernel
        for(Int d = 0; d < m_num_dims; ++d) {
          filter_offsets[d] = -m_conv_pads[d];
        }
        const Int start_output_index = output_channel*num_per_output_channel;
        const Int end_output_index = (output_channel+1)*num_per_output_channel;
        for(Int output_index = start_output_index;
            output_index < end_output_index;
            ++output_index) {
          DataType& output_entry = weighted_sum_local(output_index, sample);

          // Iterate through filter entries for current input and output channel
          for(Int d = 0; d < m_num_dims; ++d) {
            filter_pos[d] = 0;
          }
          const Int start_filter_index
            = output_channel*current_filter_size + input_channel*current_filter_size_per_input_channel;
          const Int end_filter_index
            = output_channel*current_filter_size + (input_channel+1)*current_filter_size_per_input_channel;
          for(Int filter_index = start_filter_index;
              filter_index < end_filter_index;
              ++filter_index) {
            const DataType filter_entry = filter_local(filter_index,0);

            // Get input entry corresponding to filter entry
            Int input_index = 0;
            bool valid_input_entry = true;
            for(Int d = 0; d < m_num_dims; ++d) {
              if(filter_offsets[d] + filter_pos[d] < 0
                 || filter_offsets[d] + filter_pos[d] >= m_input_dims[d]) {
                valid_input_entry = false;
                break;
              }
              input_index *= m_input_dims[d];
              input_index += filter_offsets[d] + filter_pos[d];
            }
            input_index += input_channel*num_per_input_channel;

            // Update output entry
            if(valid_input_entry) {
              const DataType input_entry = prev_activations_local(input_index, sample);
              output_entry += filter_entry*input_entry;
            }
          
            // Move to next filter entry
            ++filter_pos[m_num_dims-1];
            for(Int d = m_num_dims - 1; d > 0; --d) {
              if(filter_pos[d] >= m_filter_dims[d]) {
                filter_pos[d] = 0;
                ++filter_pos[d-1];
              }
            }
          
          }

          // Move to next filter offset and output entry
          filter_offsets[m_num_dims-1] += m_conv_strides[m_num_dims-1];
          for(Int d = m_num_dims - 1; d > 0; --d) {
            if(filter_offsets[d] + m_filter_dims[d] > m_input_dims[d] + m_conv_pads[d]) {
              filter_offsets[d] = -m_conv_pads[d];
              filter_offsets[d-1] += m_conv_strides[d-1];
            }
          }

        }

      }
    }
  }

  // weighted_sum and output are identical after fp linearity step
  Copy(weighted_sum_local, activations_local);
  
}

void lbann::convolutional_layer::fp_linearity_cpu_direct_2d() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& weights_local = m_weights->LockedMatrix();
  Mat& weighted_sum_local = m_weighted_sum_v->Matrix();
  Mat& activations_local = m_activations_v->Matrix();

  // Get filter and bias
  const Mat filter_local = LockedView(weights_local, IR(0,m_filter_size), ALL);
  const Mat bias_local = LockedView(weights_local, IR(m_filter_size,END), ALL);

  // Input, output, and filter entries are divided amongst channels
  const Int num_per_output_channel = NumNeurons / m_num_output_channels;
  const Int num_per_input_channel = m_num_prev_neurons / m_num_input_channels;
  const Int current_filter_size = m_filter_size / m_num_output_channels;
  const Int current_filter_size_per_input_channel = current_filter_size / m_num_input_channels;

  // Apply bias
  for(Int i=0; i<m_num_output_channels; ++i) {
    Mat weighted_sum_channel 
      = View(weighted_sum_local,
             IR(i*num_per_output_channel, (i+1)*num_per_output_channel),
             ALL);
    Fill(weighted_sum_channel, bias_local.Get(i,0));
  }

  ////////////////////////////////////////////////////////////////
  // Apply convolution
  // Note: We are implicitly applying a convolution matrix where
  //   filter_entry is the (output_index, input_index) entry.
  ////////////////////////////////////////////////////////////////

  // Avoid slow memory accesses by creating local variables
  const Int dim_y = m_input_dims[0];
  const Int dim_x = m_input_dims[1];
  const Int filter_dim_y = m_filter_dims[0];
  const Int filter_dim_x = m_filter_dims[1];
  const Int filter_offset_y_start = -m_conv_pads[0];
  const Int filter_offset_y_end = m_input_dims[0] + m_conv_pads[0] - m_filter_dims[0];
  const Int filter_offset_y_stride = m_conv_strides[0];
  const Int filter_offset_x_start = -m_conv_pads[1];
  const Int filter_offset_x_end = m_input_dims[1] + m_conv_pads[1] - m_filter_dims[1];
  const Int filter_offset_x_stride = m_conv_strides[1];

  // Iterate through samples, output channels, and input channels
#pragma omp parallel for
  for(Int output_channel = 0;
      output_channel < m_num_output_channels;
      ++output_channel) {
    for(Int input_channel = 0;
        input_channel < m_num_input_channels;
        ++input_channel) {
      for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {

        // Iterate through output entries in current output channel
        // Note: each output entry corresponds to a different offset
        //   of the convolutional kernel
        Int output_index = output_channel*num_per_output_channel;
        for(Int filter_offset_y = filter_offset_y_start;
            filter_offset_y <= filter_offset_y_end;
            filter_offset_y += filter_offset_y_stride) {
          for(Int filter_offset_x = filter_offset_x_start;
              filter_offset_x <= filter_offset_x_end;
              filter_offset_x += filter_offset_x_stride) {
            DataType& output_entry = weighted_sum_local(output_index, sample);

            // Iterate through filter entries for current input and output channel
            const Int input_index_start = input_channel*num_per_input_channel;
            const Int filter_index_start = output_channel*current_filter_size + input_channel*current_filter_size_per_input_channel;
            for(Int filter_pos_y = 0;
                filter_pos_y < filter_dim_y;
                ++filter_pos_y) {
              const Int pos_y = filter_offset_y + filter_pos_y;
              if(pos_y < Int(0) || pos_y >= dim_y) continue;
              for(Int filter_pos_x = 0;
                  filter_pos_x < filter_dim_x;
                  ++filter_pos_x) {
                const Int pos_x = filter_offset_x + filter_pos_x;
                if(pos_x < Int(0) || pos_x >= dim_x) continue;

                // Get indices
                const Int filter_index = filter_index_start + filter_pos_y*filter_dim_x + filter_pos_x;
                const Int input_index = input_index_start + pos_y*dim_x + pos_x;
                
                // Update output entry
                const DataType filter_entry = filter_local(filter_index,0);
                const DataType input_entry = prev_activations_local(input_index, sample);
                output_entry += filter_entry*input_entry;

              }              
            }

            // Move to next output entry
            ++output_index;

          }
        }

      }
    }
  }

  // weighted_sum and output are identical after fp linearity step
  Copy(weighted_sum_local, activations_local);

}

void lbann::convolutional_layer::fp_linearity_cpu_gemm() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& weights_local = m_weights->LockedMatrix();
  Mat& weighted_sum_local = m_weighted_sum_v->Matrix();
  Mat& activations_local = m_activations_v->Matrix();

  // Get filters and bias
  const Mat filter_local = LockedView(weights_local, IR(0,m_filter_size), ALL);
  const Mat bias_local = LockedView(weights_local, IR(m_filter_size,END), ALL);

  // Input, output, and filter entries are divided amongst channels
  const Int num_per_output_channel = NumNeurons / m_num_output_channels;
  const Int num_per_input_channel = m_num_prev_neurons / m_num_input_channels;
  const Int current_filter_size = m_filter_size / m_num_output_channels;
  const Int current_filter_size_per_input_channel = current_filter_size / m_num_input_channels;

  // Apply bias
  for(Int i=0; i<m_num_output_channels; ++i) {
    Mat weighted_sum_channel 
      = View(weighted_sum_local,
             IR(i*num_per_output_channel, (i+1)*num_per_output_channel),
             ALL);
    Fill(weighted_sum_channel, bias_local.Get(i,0));
  }

  // Reshape filters into matrix
  const Mat filter_mat(current_filter_size, m_num_output_channels,
                       filter_local.LockedBuffer(), current_filter_size);

  // Initialize im2col matrix
  Mat im2col_mat(current_filter_size, num_per_output_channel);

  // Iterate through data samples
  for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {

    // Construct im2col matrix from input
    const Mat input_mat = LockedView(prev_activations_local, ALL, IR(sample));
    im2col(input_mat, im2col_mat,
           m_input_dims, m_conv_pads, m_num_input_channels,
           m_filter_dims, m_conv_strides);

    // Apply convolution to current data sample
    Mat output_mat(num_per_output_channel, m_num_output_channels,
                   weighted_sum_local.Buffer(0,sample), num_per_output_channel);
    Gemm(TRANSPOSE, NORMAL,
         DataType(1), im2col_mat, filter_mat,
         DataType(1), output_mat);

  }

  // weighted_sum and output are identical after fp linearity step
  Copy(weighted_sum_local, activations_local);

}

void lbann::convolutional_layer::bp_linearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Clear unused columns
  m_cudnn->clear_unused_columns_on_gpus(m_prev_error_signal_d,
                                        NumNeurons,
                                        m_prev_error_signal_v->LocalWidth(),
                                        m_mini_batch_size_per_gpu);

  // Perform back propagation on each GPU
  const Int num_gpus = m_cudnn->get_num_gpus();
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    checkCUDNN(cudnnSetStream(m_cudnn->get_handle(i),
                              m_cudnn->get_stream(i)));
    checkCUDNN(cudnnConvolutionBackwardBias(m_cudnn->get_handle(i),
                                            &one,
                                            m_output_desc,
                                            m_prev_error_signal_d[i],
                                            &zero,
                                            m_bias_desc,
                                            m_weights_gradient_d[i] + m_filter_size));
    checkCUDNN(cudnnConvolutionBackwardFilter(m_cudnn->get_handle(i),
                                              &one,
                                              m_input_desc,
                                              m_prev_activations_d[i],
                                              m_output_desc,
                                              m_prev_error_signal_d[i],
                                              m_convolution_desc,
                                              m_backward_filter_algo,
                                              m_cudnn->get_work_space(i),
                                              m_cudnn->get_work_space_size(i),
                                              &zero,
                                              m_filter_desc,
                                              m_weights_gradient_d[i]));
    checkCUDNN(cudnnConvolutionBackwardData(m_cudnn->get_handle(i),
                                            &one,
                                            m_filter_desc,
                                            m_weights_d[i],
                                            m_output_desc,
                                            m_prev_error_signal_d[i],
                                            m_convolution_desc,
                                            m_backward_data_algo,
                                            m_cudnn->get_work_space(i),
                                            m_cudnn->get_work_space_size(i),
                                            &zero,
                                            m_input_desc,
                                            m_error_signal_d[i]));

  }

  // Transfer outputs from GPUs to CPU
  m_cudnn->reduce_from_gpus(m_weights_gradient->Matrix(),
                            m_weights_gradient_d);
  *m_weights_gradient *= DataType(1) / get_effective_minibatch_size();
  AllReduce(*m_weights_gradient, m_weights_gradient->RedundantComm());

#endif // #ifndef __LIB_CUDNN
}

void lbann::convolutional_layer::bp_nonlinearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Perform back propagation on each GPU
  if(m_activation_type != activation_type::ID) {
    const Int num_gpus = m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
      checkCUDNN(cudnnSetStream(m_cudnn->get_handle(i),
                                m_cudnn->get_stream(i)));
      checkCUDNN(cudnnActivationBackward(m_cudnn->get_handle(i),
                                         m_activation_desc,
                                         &one,
                                         m_output_desc,
                                         m_weighted_sum_d[i],
                                         m_output_desc,
                                         m_prev_error_signal_d[i],
                                         m_output_desc,
                                         m_activations_d[i],
                                         &zero,
                                         m_output_desc,
                                         m_prev_error_signal_d[i]));

    }
  }

#endif // #ifndef __LIB_CUDNN
}

void lbann::convolutional_layer::bp_linearity_cpu_direct() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& weights_local = m_weights->LockedMatrix();
  const Mat& weighted_sum_local = m_weighted_sum_v->LockedMatrix();
  const Mat& prev_error_signal_local = m_prev_error_signal_v->LockedMatrix();
  Mat& weights_gradient_local = m_weights_gradient->Matrix();
  Mat& error_signal_local = m_error_signal_v->Matrix();

  // Get filters and bias  
  const Mat filter_local = LockedView(weights_local, IR(0,m_filter_size), ALL);
  Mat filter_gradient_local = View(weights_gradient_local, IR(0,m_filter_size), ALL);
  Mat bias_gradient_local = View(weights_gradient_local, IR(m_filter_size,END), ALL);

  // Initialize error signal and weight gradients to zero
  Zero(weights_gradient_local);
  Zero(error_signal_local);

  // Input, output, and filter entries are divided amongst channels
  const Int num_per_output_channel = NumNeurons / m_num_output_channels;
  const Int num_per_input_channel = m_num_prev_neurons / m_num_input_channels;
  const Int current_filter_size = m_filter_size / m_num_output_channels;
  const Int current_filter_size_per_input_channel = current_filter_size / m_num_input_channels;

  // Compute bias gradient
  Mat ones;
  Ones(ones, num_per_output_channel, prev_error_signal_local.Width());
  for(Int i=0; i<m_num_output_channels; ++i) {
    const Mat prev_error_signal_channel
      = LockedView(prev_error_signal_local,
                   IR(i*num_per_output_channel,
                      (i+1)*num_per_output_channel),
                   ALL);
    bias_gradient_local.Set(i, Int(0), Dot(prev_error_signal_channel, ones));
  }

  ////////////////////////////////////////////////////////////////
  // Iterate through entries of convolution matrix
  // Note: The (output_index, input_index) entry of the convolution
  //   matrix is filter_entry (notation is from fp_linearity_cpu). The
  //   convolution matrix entries are used to update the error signal
  //   and the filter gradient.
  ////////////////////////////////////////////////////////////////

  // Iterate through samples, output channels, and input channels
#pragma omp parallel for
  for(Int input_channel = 0;
      input_channel < m_num_input_channels;
      ++input_channel) {
    std::vector<Int> filter_offsets(m_num_dims);
    std::vector<Int> filter_pos(m_num_dims);
    for(Int output_channel = 0;
        output_channel < m_num_output_channels;
        ++output_channel) {
      for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {

        // Iterate through output entries in current output channel
        // Note: each output entry corresponds to a different offset
        //   of the convolutional kernel
        for(Int d = 0; d < m_num_dims; ++d) {
          filter_offsets[d] = -m_conv_pads[d];
        }
        const Int start_output_index = output_channel*num_per_output_channel;
        const Int end_output_index = (output_channel+1)*num_per_output_channel;
        for(Int output_index = start_output_index;
            output_index < end_output_index;
            ++output_index) {

          // Iterate through filter entries for current input and output channel
          for(Int d = 0; d < m_num_dims; ++d) {
            filter_pos[d] = 0;
          }
          const Int start_filter_index
            = output_channel*current_filter_size + input_channel*current_filter_size_per_input_channel;
          const Int end_filter_index
            = output_channel*current_filter_size + (input_channel+1)*current_filter_size_per_input_channel;
          for(Int filter_index = start_filter_index;
              filter_index < end_filter_index;
              ++filter_index) {
            const DataType filter_entry = filter_local.Get(filter_index,0);

            // Get input entry corresponding to filter entry
            Int input_index = 0;
            bool valid_input_entry = true;
            for(Int d = 0; d < m_num_dims; ++d) {
              if(filter_offsets[d] + filter_pos[d] < 0
                 || filter_offsets[d] + filter_pos[d] >= m_input_dims[d]) {
                valid_input_entry = false;
                break;
              }
              input_index *= m_input_dims[d];
              input_index += filter_offsets[d] + filter_pos[d];
            }
            input_index += input_channel*num_per_input_channel;

            // Update output entry
            if(valid_input_entry) {

              // Update error signal
              // Note: error_signal = conv_matrix^T * prev_error_signal
              const DataType prev_error_signal_entry = prev_error_signal_local(output_index, sample);
              DataType& error_signal_entry = error_signal_local(input_index, sample);
              error_signal_entry += filter_entry * prev_error_signal_entry;

              // Update filter gradient
              // Note: conv_matrix_gradient = prev_error_signal * prev_activations^T
              const DataType prev_activations_entry = prev_activations_local(input_index, sample);
              DataType& filter_gradient_entry = filter_gradient_local(filter_index, Int(0));
              filter_gradient_entry += prev_error_signal_entry * prev_activations_entry;

            }
          
            // Move to next filter entry
            ++filter_pos[m_num_dims-1];
            for(Int d = m_num_dims - 1; d > 0; --d) {
              if(filter_pos[d] >= m_filter_dims[d]) {
                filter_pos[d] = 0;
                ++filter_pos[d-1];
              }
            }

          }

          // Move to next filter offset and output entry
          filter_offsets[m_num_dims-1] += m_conv_strides[m_num_dims-1];
          for(Int d = m_num_dims - 1; d > 0; --d) {
            if(filter_offsets[d] + m_filter_dims[d] > m_input_dims[d] + m_conv_pads[d]) {
              filter_offsets[d] = -m_conv_pads[d];
              filter_offsets[d-1] += m_conv_strides[d-1];
            }
          }

        }

      }
    }
  }

  // Scale and accumulate gradients
  *m_weights_gradient *= DataType(1) / get_effective_minibatch_size();
  AllReduce(*m_weights_gradient, m_weights_gradient->RedundantComm());

}

void lbann::convolutional_layer::bp_linearity_cpu_direct_2d() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& weights_local = m_weights->LockedMatrix();
  const Mat& weighted_sum_local = m_weighted_sum_v->LockedMatrix();
  const Mat& prev_error_signal_local = m_prev_error_signal_v->LockedMatrix();
  Mat& weights_gradient_local = m_weights_gradient->Matrix();
  Mat& error_signal_local = m_error_signal_v->Matrix();

  // Get filters and bias  
  const Mat filter_local = LockedView(weights_local, IR(0,m_filter_size), ALL);
  Mat filter_gradient_local = View(weights_gradient_local, IR(0,m_filter_size), ALL);
  Mat bias_gradient_local = View(weights_gradient_local, IR(m_filter_size,END), ALL);

  // Initialize error signal and weight gradients to zero
  Zero(weights_gradient_local);
  Zero(error_signal_local);

  // Input, output, and filter entries are divided amongst channels
  const Int num_per_output_channel = NumNeurons / m_num_output_channels;
  const Int num_per_input_channel = m_num_prev_neurons / m_num_input_channels;
  const Int current_filter_size = m_filter_size / m_num_output_channels;
  const Int current_filter_size_per_input_channel = current_filter_size / m_num_input_channels;

  // Compute bias gradient
  Mat ones;
  Ones(ones, num_per_output_channel, prev_error_signal_local.Width());
  for(Int i=0; i<m_num_output_channels; ++i) {
    const Mat prev_error_signal_channel
      = LockedView(prev_error_signal_local,
                   IR(i*num_per_output_channel,
                      (i+1)*num_per_output_channel),
                   ALL);
    bias_gradient_local.Set(i, Int(0), Dot(prev_error_signal_channel, ones));
  }

  ////////////////////////////////////////////////////////////////
  // Iterate through entries of convolution matrix
  // Note: The (output_index, input_index) entry of the convolution
  //   matrix is filter_entry (notation is from fp_linearity_cpu). The
  //   convolution matrix entries are used to update the error signal
  //   and the filter gradient.
  ////////////////////////////////////////////////////////////////

  // Avoid slow memory accesses by creating local variables
  const Int dim_y = m_input_dims[0];
  const Int dim_x = m_input_dims[1];
  const Int filter_dim_y = m_filter_dims[0];
  const Int filter_dim_x = m_filter_dims[1];
  const Int filter_offset_y_start = -m_conv_pads[0];
  const Int filter_offset_y_end = m_input_dims[0] + m_conv_pads[0] - m_filter_dims[0];
  const Int filter_offset_y_stride = m_conv_strides[0];
  const Int filter_offset_x_start = -m_conv_pads[1];
  const Int filter_offset_x_end = m_input_dims[1] + m_conv_pads[1] - m_filter_dims[1];
  const Int filter_offset_x_stride = m_conv_strides[1];

  // Iterate through samples, output channels, and input channels
#pragma omp parallel for
  for(Int input_channel = 0;
      input_channel < m_num_input_channels;
      ++input_channel) {
    for(Int output_channel = 0;
        output_channel < m_num_output_channels;
        ++output_channel) {
      for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {

        // Iterate through output entries in current output channel
        // Note: each output entry corresponds to a different offset
        //   of the convolutional kernel
        Int output_index = output_channel*num_per_output_channel;
        for(Int filter_offset_y = filter_offset_y_start;
            filter_offset_y <= filter_offset_y_end;
            filter_offset_y += filter_offset_y_stride) {
          for(Int filter_offset_x = filter_offset_x_start;
              filter_offset_x <= filter_offset_x_end;
              filter_offset_x += filter_offset_x_stride) {

            // Iterate through filter entries for current input and output channel
            const Int input_index_start = input_channel*num_per_input_channel;
            const Int filter_index_start = output_channel*current_filter_size + input_channel*current_filter_size_per_input_channel;
            for(Int filter_pos_y = 0;
                filter_pos_y < filter_dim_y;
                ++filter_pos_y) {
              const Int pos_y = filter_offset_y + filter_pos_y;
              if(pos_y < Int(0) || pos_y >= dim_y) continue;
              for(Int filter_pos_x = 0;
                  filter_pos_x < filter_dim_x;
                  ++filter_pos_x) {
                const Int pos_x = filter_offset_x + filter_pos_x;
                if(pos_x < Int(0) || pos_x >= dim_x) continue;

                // Get indices
                const Int filter_index = filter_index_start + filter_pos_y*filter_dim_x + filter_pos_x;
                const Int input_index = input_index_start + pos_y*dim_x + pos_x;

                // Update error signal
                // Note: error_signal = conv_matrix^T * prev_error_signal
                const DataType& filter_entry = filter_local(filter_index, Int(0));
                const DataType prev_error_signal_entry = prev_error_signal_local(output_index, sample);
                DataType& error_signal_entry = error_signal_local(input_index, sample);
                error_signal_entry += filter_entry * prev_error_signal_entry;

                // Update filter gradient
                // Note: conv_matrix_gradient = prev_error_signal * prev_activations^T
                const DataType prev_activations_entry = prev_activations_local(input_index, sample);
                DataType& filter_gradient_entry = filter_gradient_local(filter_index, Int(0));
                filter_gradient_entry += prev_error_signal_entry * prev_activations_entry;

              }
            }

            // Move to next output entry
            ++output_index;

          }
        }

      }
    }
  }

  // Scale and accumulate gradients
  *m_weights_gradient *= DataType(1) / get_effective_minibatch_size();
  AllReduce(*m_weights_gradient, m_weights_gradient->RedundantComm());

}

void lbann::convolutional_layer::bp_linearity_cpu_gemm() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& weights_local = m_weights->LockedMatrix();
  const Mat& weighted_sum_local = m_weighted_sum_v->LockedMatrix();
  const Mat& prev_error_signal_local = m_prev_error_signal_v->LockedMatrix();
  Mat& weights_gradient_local = m_weights_gradient->Matrix();
  Mat& error_signal_local = m_error_signal_v->Matrix();

  // Get filters and bias  
  const Mat filter_local = LockedView(weights_local, IR(0,m_filter_size), ALL);
  Mat filter_gradient_local = View(weights_gradient_local, IR(0,m_filter_size), ALL);
  Mat bias_gradient_local = View(weights_gradient_local, IR(m_filter_size,END), ALL);

  // Initialize weight gradients to zero
  Zero(weights_gradient_local);

  // Input, output, and filter entries are divided amongst channels
  const Int num_per_output_channel = NumNeurons / m_num_output_channels;
  const Int num_per_input_channel = m_num_prev_neurons / m_num_input_channels;
  const Int current_filter_size = m_filter_size / m_num_output_channels;
  const Int current_filter_size_per_input_channel = current_filter_size / m_num_input_channels;

  // Compute bias gradient
#pragma omp parallel for
  for(Int output_channel = 0;
      output_channel < m_num_output_channels;
      ++output_channel) {
    DataType& bias_gradient_entry = bias_gradient_local(output_channel, 0);
    for(Int col = 0; col < prev_error_signal_local.Width(); ++col) {
      for(Int row = output_channel * num_per_output_channel;
          row < (output_channel+1) * num_per_output_channel;
          ++row) {
        bias_gradient_entry += prev_error_signal_local(row, col);
      }
    }
  }

  // Initialize filter and im2col matrices
  const Mat filter_mat(current_filter_size, m_num_output_channels,
                       filter_local.LockedBuffer(), current_filter_size);
  Mat filter_gradient_mat(current_filter_size, m_num_output_channels,
                          filter_gradient_local.Buffer(), current_filter_size);
  Mat im2col_mat(current_filter_size, num_per_output_channel);

  // Iterate through data samples
  for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {

    // Reshape previous error signal into matrix
    const Mat prev_error_signal_mat(num_per_output_channel,
                                    m_num_output_channels,
                                    prev_error_signal_local.LockedBuffer(0,sample),
                                    num_per_output_channel);

    // Compute gradient w.r.t. input im2col matrix
    Gemm(NORMAL, TRANSPOSE,
         DataType(1), filter_mat, prev_error_signal_mat,
         DataType(0), im2col_mat);

    // Compute error signal (i.e. gradient w.r.t. input)
    Mat output_mat = View(error_signal_local, ALL, IR(sample));
    col2im(im2col_mat, output_mat,
           m_input_dims, m_conv_pads, m_num_input_channels,
           m_filter_dims, m_conv_strides);

    // Construct im2col matrix from input
    const Mat input_mat = LockedView(prev_activations_local,
                                     ALL, IR(sample));
    im2col(input_mat, im2col_mat,
           m_input_dims, m_conv_pads, m_num_input_channels,
           m_filter_dims, m_conv_strides);

    // Compute gradient w.r.t. filter
    Gemm(NORMAL, NORMAL,
         DataType(1), im2col_mat, prev_error_signal_mat,
         DataType(1), filter_gradient_mat);

  }

  // Scale and accumulate gradients
  *m_weights_gradient *= DataType(1) / get_effective_minibatch_size();
  AllReduce(*m_weights_gradient, m_weights_gradient->RedundantComm());

}

bool convolutional_layer::update()
{
  double start = get_time();

  // Regularize gradients and update regularizers
  Layer::update();

  if(m_execution_mode == execution_mode::training) {
    // Apply optimizer
    m_optimizer->update(m_weights_gradient);
  }

  update_time += get_time() - start;
  return true;
}

