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
    throw lbann_exception("convolutional_layer: invalid activation type");
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
                                         Optimizer* optimizer,
                                         cudnn::cudnn_manager* cudnn)
  : Layer(index, comm, optimizer, mini_batch_size, activation, {}),
    m_weight_initialization(init),
    m_num_dims(num_dims),
    m_num_input_channels(num_input_channels),
    m_num_output_channels(num_output_channels),
    m_cudnn(cudnn)
{
  m_type = layer_type::convolution;

  // Initialize input dimensions and convolution parameters
  m_input_dims.resize(m_num_dims);
  m_filter_dims.resize(m_num_dims);
  m_filter_size = m_num_input_channels*m_num_output_channels;
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

#ifdef __LIB_CUDNN

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
  m_work_space_size = 0;

  // Initialize GPU memory if using GPU
  if(cudnn) {
    m_using_gpu = true;

    // Get number of GPUs
    const int num_gpus = m_cudnn->get_num_gpus();

    // Get number of columns per GPU
    const int num_processes = comm->get_procs_per_model();
    const int local_mini_batch_size = (mini_batch_size + num_processes - 1) / num_processes;
    m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

  }
  is_pinned_fwd = false;
  is_pinned_bwd = false;
#endif // #ifdef __LIB_CUDNN

}

convolutional_layer::~convolutional_layer()
{
#ifdef __LIB_CUDNN
  if(m_using_gpu) {
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
  }
#endif // #ifdef __LIB_CUDNN
}

void convolutional_layer::setup(const int num_prev_neurons)
{
  Layer::setup(num_prev_neurons);

#ifdef __LIB_CUDNN
  // Setup cuDNN objects
  if(m_using_gpu) {
    setup_gpu();
  }
#endif // #ifdef __LIB_CUDNN

#ifdef LBANN_DEBUG
  // Check if input dimensions are valid
  int num_inputs = m_num_input_channels;
  for(int i=0; i<m_num_dims; ++i)
    num_inputs *= m_input_dims[i];
  if(num_inputs != m_num_prev_neurons) {
    throw lbann_exception("lbann_layer_convolutional: unexpected number of input neurons");
  }
#endif // #ifdef LBANN_DEBUG

  // Initialize matrices
  Zeros(*m_prev_activations, m_num_prev_neurons, m_mini_batch_size);
  Zeros(*m_weighted_sum, NumNeurons, m_mini_batch_size);
  Zeros(*m_weights, m_filter_size+m_num_output_channels, 1);
  Zeros(*m_activations, NumNeurons, m_mini_batch_size);
  Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size);
  Zeros(*m_weights_gradient, m_filter_size+m_num_output_channels, 1);
  Zeros(*m_error_signal, m_num_prev_neurons, m_mini_batch_size);

  // Initialize filters
  StarMat filter;
  View(filter, *m_weights, IR(0,m_filter_size), ALL);
  Int fan_in = m_filter_size / m_num_output_channels;
  Int fan_out = m_filter_size / m_num_input_channels;
  switch(m_weight_initialization) {
  case weight_initialization::uniform:
    uniform_fill(filter, filter.Height(), filter.Width(),
                 DataType(0), DataType(1));
    break;
  case weight_initialization::normal:
    gaussian_fill(filter, filter.Height(), filter.Width(),
                  DataType(0), DataType(1));
    break;
  case weight_initialization::glorot_normal: {
    const DataType var = 2.0 / (fan_in + fan_out);
    gaussian_fill(filter, filter.Height(), filter.Width(),
                  DataType(0), sqrt(var));
    break;
  }
  case weight_initialization::glorot_uniform: {
    const DataType var = 2.0 / (fan_in + fan_out);
    uniform_fill(filter, filter.Height(), filter.Width(),
                 DataType(0), sqrt(3*var));
    break;
  }
  case weight_initialization::he_normal: {
    const DataType var = 1.0 / fan_in;
    gaussian_fill(filter, filter.Height(), filter.Width(),
                  DataType(0), sqrt(var));
    break;
  }
  case weight_initialization::he_uniform: {
    const DataType var = 1.0 / fan_in;
    uniform_fill(filter, filter.Height(), filter.Width(),
                 DataType(0), sqrt(3*var));
    break;
  }
  case weight_initialization::zero: // Zero initialization is default
  default:
    Zero(filter);
    break;
  }
  
  // Initialize optimizer
  optimizer->setup(1, m_filter_size+m_num_output_channels);
  
}

void lbann::convolutional_layer::setup_gpu()
{
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#else

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
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                 0,
                                                 &m_forward_algo));
  checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(m_cudnn->get_handle(),
                                                        m_input_desc,
                                                        m_output_desc,
                                                        m_convolution_desc,
                                                        m_filter_desc,
                                                        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                        0,
                                                        &m_backward_filter_algo));
  checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(m_cudnn->get_handle(),
                                                      m_filter_desc,
                                                      m_output_desc,
                                                      m_convolution_desc,
                                                      m_input_desc,
                                                      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                      0,
                                                      &m_backward_data_algo));

  // Choose workspace size
  m_work_space_size = 0;
  size_t required_work_space;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_cudnn->get_handle(),
                                                     m_input_desc,
                                                     m_filter_desc,
                                                     m_convolution_desc,
                                                     m_output_desc,
                                                     m_forward_algo,
                                                     &required_work_space));
  m_work_space_size = Max(m_work_space_size, required_work_space);
  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(m_cudnn->get_handle(),
                                                            m_input_desc,
                                                            m_output_desc,
                                                            m_convolution_desc,
                                                            m_filter_desc,
                                                            m_backward_filter_algo,
                                                            &required_work_space));
  m_work_space_size = Max(m_work_space_size, required_work_space);
  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(m_cudnn->get_handle(),
                                                          m_filter_desc,
                                                          m_output_desc,
                                                          m_convolution_desc,
                                                          m_input_desc,
                                                          m_backward_data_algo,
                                                          &required_work_space));
  m_work_space_size = Max(m_work_space_size, required_work_space);

  // Set activation descriptor
  checkCUDNN(cudnnSetActivationDescriptor(m_activation_desc,
                                          get_cudnn_activation_mode(m_activation_type),
                                          CUDNN_PROPAGATE_NAN,
                                          0.0));

#endif // #ifdef __LIB_CUDNN
}

void lbann::convolutional_layer::pin_memory_blocks_fwd(void)
{
  if (!m_using_gpu) {
    std::cout << "no offloading with convolutional_layer " << get_index() << std::endl;
    return;
  }

#ifdef __LIB_CUDNN
  m_cudnn->pin_memory_block(m_weights);
  m_cudnn->pin_memory_block(m_weighted_sum);
  m_cudnn->pin_memory_block(m_activations);
  m_cudnn->pin_memory_block(m_prev_activations);

  is_pinned_fwd = true;
#endif // #ifdef __LIB_CUDNN
}

void lbann::convolutional_layer::pin_memory_blocks_bwd(void)
{
  if (!m_using_gpu) {
    std::cout << "no offloading with convolutional_layer " << get_index() << std::endl;
    return;
  }

#ifdef __LIB_CUDNN
  m_cudnn->pin_memory_block(m_error_signal);
  m_cudnn->pin_memory_block(m_prev_error_signal);
  m_cudnn->pin_memory_block(m_weights_gradient);

  is_pinned_bwd = true;
#endif // #ifdef __LIB_CUDNN
}

void lbann::convolutional_layer::fp_linearity() {
  if(m_using_gpu) {
    fp_linearity_gpu();
  }
  else {
    fp_linearity_cpu();
  }
}

void lbann::convolutional_layer::bp_linearity() {
  if(m_using_gpu) {
    bp_linearity_gpu();
  }
  else {
    bp_linearity_cpu();
  }
}

void lbann::convolutional_layer::fp_nonlinearity() {
  if(m_using_gpu) {
    fp_nonlinearity_gpu();
  }
  else {
    Layer::fp_nonlinearity();
  }
}

void lbann::convolutional_layer::bp_nonlinearity() {
  if(m_using_gpu) {
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

  // Get local matrices
  const Mat& weights_local = m_weights->LockedMatrix();
  const Mat filter_local = weights_local(IR(0,m_filter_size), ALL);
  const Mat bias_local = weights_local(IR(m_filter_size,END), ALL);

  // Allocate GPU memory
  m_cudnn->allocate_on_gpus(m_filter_d, m_filter_size, 1);
  m_cudnn->allocate_on_gpus(m_bias_d, m_num_output_channels, 1);
  m_cudnn->allocate_on_gpus(m_prev_activations_d,
                            m_num_prev_neurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_weighted_sum_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_activations_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_work_space_d,
                            (m_work_space_size+sizeof(DataType)-1)/sizeof(DataType),
                            1);

  // Transfer data from CPU to GPUs
  m_cudnn->broadcast_to_gpus(m_filter_d, filter_local);
  m_cudnn->broadcast_to_gpus(m_bias_d, bias_local);
  m_cudnn->copy_to_gpus(m_prev_activations_d,
                        m_prev_activations_v->LockedMatrix(),
                        m_mini_batch_size_per_gpu);

  // Perform convolution on each GPU
  const Int num_gpus = m_cudnn->get_num_gpus();
#pragma omp parallel for
  for(Int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    checkCUDNN(cudnnConvolutionForward(m_cudnn->get_handle(i),
                                       &one,
                                       m_input_desc,
                                       m_prev_activations_d[i],
                                       m_filter_desc,
                                       m_filter_d[i],
                                       m_convolution_desc,
                                       m_forward_algo,
                                       m_work_space_d[i],
                                       m_work_space_size,
                                       &zero,
                                       m_output_desc,
                                       m_weighted_sum_d[i]));
    checkCUDNN(cudnnAddTensor(m_cudnn->get_handle(i),
                              &one,
                              m_bias_desc,
                              m_bias_d[i],
                              &one,
                              m_output_desc,
                              m_weighted_sum_d[i]));
  }

  // Copy result to output matrix
  m_cudnn->copy_on_gpus(m_activations_d,
                        m_weighted_sum_d,
                        NumNeurons,
                        m_mini_batch_size_per_gpu);

  // Transfer data from GPUs to CPU
  m_cudnn->copy_from_gpus(m_weighted_sum_v->Matrix(),
                          m_weighted_sum_d,
                          m_mini_batch_size_per_gpu);
  m_cudnn->copy_from_gpus(m_activations_v->Matrix(),
                          m_activations_d,
                          m_mini_batch_size_per_gpu);
  m_cudnn->synchronize();

  // Deallocate GPU memory
  m_cudnn->deallocate_on_gpus(m_filter_d);
  m_cudnn->deallocate_on_gpus(m_bias_d);
  m_cudnn->deallocate_on_gpus(m_prev_activations_d);
  m_cudnn->deallocate_on_gpus(m_weighted_sum_d);
  m_cudnn->deallocate_on_gpus(m_activations_d);
  m_cudnn->deallocate_on_gpus(m_work_space_d);

#endif // #ifndef __LIB_CUDNN
}

void lbann::convolutional_layer::fp_nonlinearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Allocate GPU memory
  m_cudnn->allocate_on_gpus(m_activations_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);

  // Transfer inputs from CPU to GPUs
  m_cudnn->copy_to_gpus(m_activations_d,
                        m_activations_v->LockedMatrix(),
                        m_mini_batch_size_per_gpu);

  // Perform activation with each GPU
  const Int num_gpus = m_cudnn->get_num_gpus();
#pragma omp parallel for
  for(Int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    checkCUDNN(cudnnActivationForward(m_cudnn->get_handle(i),
                                      m_activation_desc,
                                      &one,
                                      m_output_desc,
                                      m_activations_d[i],
                                      &zero,
                                      m_output_desc,
                                      m_activations_d[i]));

  }

  // Transfer outputs from GPUs to CPU
  m_cudnn->copy_from_gpus(m_activations_v->Matrix(),
                          m_activations_d,
                          m_mini_batch_size_per_gpu);
  m_cudnn->synchronize();

  // Deallocate GPU memory
  m_cudnn->deallocate_on_gpus(m_activations_d);

#endif // #ifndef __LIB_CUDNN
}  

/// @todo Write a more efficient implementation
void lbann::convolutional_layer::fp_linearity_cpu() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& weights_local = m_weights->LockedMatrix();
  Mat& weighted_sum_local = m_weighted_sum_v->Matrix();
  Mat& activations_local = m_activations_v->Matrix();

  // Get filter and bias
  const Mat filter_local = weights_local(IR(0,m_filter_size), ALL);
  const Mat bias_local = weights_local(IR(m_filter_size,END), ALL);

  // Apply bias to each sample in mini-batch
  IndexDependentFill(weighted_sum_local, (std::function<DataType(El::Int,El::Int)>)
                     ([this, &bias_local](El::Int r, El::Int c)->DataType {
                       const Int num_per_channel = NumNeurons / m_num_output_channels;
                       const Int channel = r / num_per_channel;
                       return bias_local.Get(channel,0); 
                     }));

  // Initialize convolution matrix
  Mat convolution_matrix;
  Zeros(convolution_matrix, NumNeurons, m_num_prev_neurons);

  // Iterate through filters
  int row = 0;
  for(int output_channel = 0;
      output_channel < m_num_output_channels;
      ++output_channel) {
    const int current_filter_size = m_filter_size / m_num_output_channels;
    const Mat filter = filter_local(IR(output_channel*current_filter_size,
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
            col += prev_activations_local.Height() / m_num_input_channels;
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
       DataType(1), convolution_matrix, prev_activations_local,
       DataType(1), weighted_sum_local);

  // Z and Y are identical after fp linearity step
  Copy(weighted_sum_local, activations_local);

}

void lbann::convolutional_layer::bp_linearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Get local matrices
  const Mat& weights_local = m_weights->LockedMatrix();
  const Mat filter_local = weights_local(IR(0,m_filter_size), ALL);
  Mat& weights_gradient_local = m_weights_gradient->Matrix();
  Mat filter_gradient_local = weights_gradient_local(IR(0,m_filter_size), ALL);
  Mat bias_gradient_local = weights_gradient_local(IR(m_filter_size,END), ALL);
  
  // Get number of samples per GPU
  const DataType mini_batch_size_per_gpu_float = m_mini_batch_size_per_gpu;

  // Allocate GPU memory
  m_cudnn->allocate_on_gpus(m_filter_d, m_filter_size, 1);
  m_cudnn->allocate_on_gpus(m_filter_gradient_d, m_filter_size, 1);
  m_cudnn->allocate_on_gpus(m_bias_gradient_d, m_num_output_channels, 1);
  m_cudnn->allocate_on_gpus(m_prev_activations_d,
                            m_num_prev_neurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_weighted_sum_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_prev_error_signal_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_error_signal_d,
                            m_num_prev_neurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_work_space_d,
                            (m_work_space_size+sizeof(DataType)-1)/sizeof(DataType),
                            1);

  // Transfer data from CPU to GPUs
  m_cudnn->broadcast_to_gpus(m_filter_d, filter_local);
  m_cudnn->copy_to_gpus(m_prev_activations_d,
                        m_prev_activations_v->LockedMatrix(),
                        m_mini_batch_size_per_gpu);
  m_cudnn->copy_to_gpus(m_weighted_sum_d,
                        m_weighted_sum_v->LockedMatrix(),
                        m_mini_batch_size_per_gpu);
  m_cudnn->copy_to_gpus(m_prev_error_signal_d,
                        m_prev_error_signal_v->LockedMatrix(),
                        m_mini_batch_size_per_gpu);

  // Perform back propagation on each GPU
  const Int num_gpus = m_cudnn->get_num_gpus();
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    checkCUDNN(cudnnConvolutionBackwardBias(m_cudnn->get_handle(i),
                                            &mini_batch_size_per_gpu_float,
                                            m_output_desc,
                                            m_prev_error_signal_d[i],
                                            &zero,
                                            m_bias_desc,
                                            m_bias_gradient_d[i]));
    checkCUDNN(cudnnConvolutionBackwardFilter(m_cudnn->get_handle(i),
                                              &mini_batch_size_per_gpu_float,
                                              m_input_desc,
                                              m_prev_activations_d[i],
                                              m_output_desc,
                                              m_prev_error_signal_d[i],
                                              m_convolution_desc,
                                              m_backward_filter_algo,
                                              m_work_space_d[i],
                                              m_work_space_size,
                                              &zero,
                                              m_filter_desc,
                                              m_filter_gradient_d[i]));
    checkCUDNN(cudnnConvolutionBackwardData(m_cudnn->get_handle(i),
                                            &one,
                                            m_filter_desc,
                                            m_filter_d[i],
                                            m_output_desc,
                                            m_prev_error_signal_d[i],
                                            m_convolution_desc,
                                            m_backward_data_algo,
                                            m_work_space_d[i],
                                            m_work_space_size,
                                            &zero,
                                            m_input_desc,
                                            m_error_signal_d[i]));

  }

  // Transfer outputs from GPUs to CPU
  m_cudnn->copy_from_gpus(m_error_signal_v->Matrix(),
                          m_error_signal_d,
                          m_mini_batch_size_per_gpu);
  m_cudnn->reduce_from_gpus(filter_gradient_local,
                            m_filter_gradient_d);
  m_cudnn->reduce_from_gpus(bias_gradient_local,
                            m_bias_gradient_d);

  // Deallocate GPU memory
  m_cudnn->deallocate_on_gpus(m_filter_d);
  m_cudnn->deallocate_on_gpus(m_filter_gradient_d);
  m_cudnn->deallocate_on_gpus(m_bias_gradient_d);
  m_cudnn->deallocate_on_gpus(m_prev_activations_d);
  m_cudnn->deallocate_on_gpus(m_weighted_sum_d);
  m_cudnn->deallocate_on_gpus(m_prev_error_signal_d);
  m_cudnn->deallocate_on_gpus(m_error_signal_d);
  m_cudnn->deallocate_on_gpus(m_work_space_d);

  // Obtain filter and bias gradients with reduction and scaling
  AllReduce(*m_weights_gradient, m_weights_gradient->DistComm());
  *m_weights_gradient *= DataType(1) / get_effective_minibatch_size();

#endif // #ifndef __LIB_CUDNN
}

void lbann::convolutional_layer::bp_nonlinearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_convolutional: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Allocate GPU memory
  m_cudnn->allocate_on_gpus(m_weighted_sum_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_activations_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_prev_error_signal_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);

  // Transfer data from CPU to GPUs
  m_cudnn->copy_to_gpus(m_weighted_sum_d,
                        m_weighted_sum_v->LockedMatrix(),
                        m_mini_batch_size_per_gpu);
  m_cudnn->copy_to_gpus(m_activations_d,
                        m_activations_v->LockedMatrix(),
                        m_mini_batch_size_per_gpu);
  m_cudnn->copy_to_gpus(m_prev_error_signal_d,
                        m_prev_error_signal_v->LockedMatrix(),
                        m_mini_batch_size_per_gpu);  

  // Perform back propagation on each GPU
  const Int num_gpus = m_cudnn->get_num_gpus();
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
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

  // Transfer data from GPUs to CPU
  m_cudnn->copy_from_gpus(m_prev_error_signal_v->Matrix(),
                          m_prev_error_signal_d,
                          m_mini_batch_size_per_gpu);  
  m_cudnn->synchronize();

  // Deallocate GPU memory
  m_cudnn->deallocate_on_gpus(m_weighted_sum_d);
  m_cudnn->deallocate_on_gpus(m_activations_d);
  m_cudnn->deallocate_on_gpus(m_prev_error_signal_d);

#endif // #ifndef __LIB_CUDNN
}

/// @todo Write a more efficient implementation
void lbann::convolutional_layer::bp_linearity_cpu() {

  // Non-linearity forward pass
  m_activation_fn->backwardProp(*m_weighted_sum_v);
  if (m_activation_type != activation_type::ID) {
    Hadamard(*m_prev_error_signal_v, *m_weighted_sum_v, *m_prev_error_signal_v);
  }
  
  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& weights_local = m_weights->LockedMatrix();
  const Mat& weighted_sum_local = m_weighted_sum_v->LockedMatrix();
  const Mat& prev_error_signal_local = m_prev_error_signal_v->LockedMatrix();
  Mat& weights_gradient_local = m_weights_gradient->Matrix();
  Mat& error_signal_local = m_error_signal_v->Matrix();

  // Get filters and bias
  const Mat filter_local = weights_local(IR(0,m_filter_size), ALL);
  Mat filter_gradient_local = weights_gradient_local(IR(0,m_filter_size), ALL);
  Mat bias_gradient_local = weights_gradient_local(IR(m_filter_size,END), ALL);

  //////////////////////////////////////////////
  // Construct convolution matrix
  //////////////////////////////////////////////

  // Initialize convolution matrix
  Mat convolution_matrix;
  Zeros(convolution_matrix, NumNeurons, prev_activations_local.Height());

  // Iterate through filters
  int row = 0;
  for(int output_channel = 0;
      output_channel < m_num_output_channels;
      ++output_channel) {
    const int current_filter_size = m_filter_size / m_num_output_channels;
    const Mat filter = filter_local(IR(output_channel*current_filter_size,
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
            col += prev_activations_local.Height()  / m_num_input_channels;
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
  Int num_per_channel = NumNeurons / m_num_output_channels;
  Mat ones;
  Ones(ones, num_per_channel, prev_error_signal_local.Width());
  for(Int i=0; i<m_num_output_channels; ++i) {
    const Mat& current_channel = prev_error_signal_local(IR(i*num_per_channel,
                                                            (i+1)*num_per_channel),
                                                         ALL);
    bias_gradient_local.Set(i, Int(0), Dot(current_channel,ones));
  }

  // Compute error signal w.r.t. convolution matrix
  Mat conv_error_signal(convolution_matrix.Height(),
                        convolution_matrix.Width());
  Gemm(NORMAL, TRANSPOSE,
       DataType(1), prev_error_signal_local, prev_activations_local,
       DataType(0), conv_error_signal);

  // Initialize filter gradient
  Zero(filter_gradient_local);

  // Iterate through filters
  row = 0;
  for(int output_channel = 0;
      output_channel < m_num_output_channels;
      ++output_channel) {
    const int current_filter_size = m_filter_size / m_num_output_channels;
    Mat filter_gradient
      = filter_gradient_local(IR(output_channel*current_filter_size,
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
            col += prev_activations_local.Height() / m_num_input_channels;
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

  // Obtain filter gradient with reduction and scaling
  AllReduce(*m_weights_gradient, m_weights_gradient->DistComm());
  *m_weights_gradient *= DataType(1) / get_effective_minibatch_size();

}

bool convolutional_layer::update()
{
  if(m_execution_mode == execution_mode::training) {
    optimizer->update_weight_bias_matrix(*m_weights_gradient, *m_weights);
  }
  return true;
}

