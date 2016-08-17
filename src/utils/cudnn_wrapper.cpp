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
// cudnn_wrapper .hpp .cpp - cuDNN support - wrapper classes, utility functions
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/lbann_exception.hpp"

#include <iostream>

#include "El.hpp"

#ifdef __LIB_CUDNN

using namespace cudnn;

// Error utility macros
#define checkCUDA(status) {                                             \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  /* TODO: remove */ \
      cudaDeviceReset();                                                \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }
#define checkCUDNN(status) {                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
      std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n"; /* TODO: remove */ \
      cudaDeviceReset();                                                \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

/// Get cuDNN data type associated with C++ data type
/** Half-, single-, and double-precision floating point data types */
template <typename T>
cudnnDataType_t get_cudnn_data_type()
{
  switch(sizeof(T)) {
  case 2: return CUDNN_DATA_HALF;
  case 4: return CUDNN_DATA_FLOAT;
  case 8: return CUDNN_DATA_DOUBLE;
  default: throw lbann::lbann_exception("cudnn_wrapper: invalid data type for cuDNN");
  }
}

/// Get cuDNN pooling mode
/** 0 = max, 1 = average (include padding), 2 = average (exclude padding)*/
cudnnPoolingMode_t get_cudnn_pool_mode(const int pool_mode)
{
  switch(pool_mode) {
  case 0: return CUDNN_POOLING_MAX;
  case 1: return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  case 2: return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  default: throw lbann::lbann_exception("cudnn_wrapper: invalid pooling mode");
  }
}

cudnn_manager::cudnn_manager(lbann::lbann_comm* _comm, int max_num_gpus)
  : comm(_comm)
{

  // Determine number of available GPUs
  checkCUDA(cudaGetDeviceCount(&m_num_total_gpus));
  if(max_num_gpus >= 0 && max_num_gpus < m_num_total_gpus) {
    m_num_total_gpus = max_num_gpus;
  }
  if(m_num_total_gpus < 1) {
    throw lbann::lbann_exception("cudnn_wrapper: no GPUs allocated or found for cuDNN");
  }

  // Determine number of MPI ranks on current compute node
  const int rank_in_node = comm->get_rank_in_node();
  const int procs_per_node = comm->get_procs_per_node();
  
  // Case where compute node has more GPUs than MPI ranks
  // TODO: smarter way to allocate GPUs to MPI ranks
  if(m_num_total_gpus >= procs_per_node) {
    int gpu = rank_in_node;
    while(gpu < m_num_total_gpus) {
      checkCUDA(cudaSetDevice(gpu));
      m_gpus.push_back(gpu);
      m_handles.push_back(NULL);
      checkCUDNN(cudnnCreate(&m_handles.back()));
      gpu += procs_per_node;
    }
  }

  // Case where compute node has fewers GPUs than MPI ranks
  // TODO: smarter way to allocate GPUs to MPI ranks
  else {
    const int gpu = rank_in_node % m_num_total_gpus;
    checkCUDA(cudaSetDevice(gpu));
    m_gpus.push_back(gpu);
    m_handles.push_back(NULL);
    checkCUDNN(cudnnCreate(&m_handles.back()));
  }

  // Get number of GPUs for current MPI rank
  m_num_gpus = m_gpus.size();

}

cudnn_manager::~cudnn_manager()
{
  // Destroy cuDNN handles
  for(int i=0; i<m_gpus.size(); ++i) {
    const int gpu = m_gpus[i];
    cudnnHandle_t handle = m_handles[i];
    if(handle) {
      checkCUDA(cudaSetDevice(gpu));
      checkCUDNN(cudnnDestroy(handle));
    }
  }
}

void cudnn_manager::print_version() const {
  std::cout << "cudnnGetVersion() :" << (int)cudnnGetVersion() << " , "
            << "CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION
            << std::endl;
}

cudnn_convolutional_layer
::cudnn_convolutional_layer(const int num_dims,
                            const int src_channels,
                            const int dst_channels,
                            const int* src_dims,
                            const int* filter_dims,
                            const int* conv_pads,
                            const int* conv_strides,
                            const int mini_batch_size,
                            cudnn_manager* cudnn)
  : m_num_dims(num_dims), m_cudnn(cudnn),
    m_cudnn_data_type(get_cudnn_data_type<DataType>()),
    m_src_desc(NULL), m_dst_desc(NULL),
    m_filter_desc(NULL), m_conv_desc(NULL)
{

  // Get number of GPUs
  int num_gpus = m_cudnn->m_num_gpus;

  // Get input tensor dimensions
  m_src_dims.resize(m_num_dims+2);
  m_src_dims[0] = (mini_batch_size + num_gpus - 1) / num_gpus;
  m_src_dims[1] = src_channels;
  for(int i=0; i<m_num_dims; ++i)
    m_src_dims[i+2] = src_dims[i];
  m_src_size = 1;
  for(int i=1; i<m_src_dims.size(); ++i)
    m_src_size *= m_src_dims[i];

  // Get filter tensor dimensions
  m_filter_dims.resize(m_num_dims+2);
  m_filter_dims[0] = dst_channels;
  m_filter_dims[1] = src_channels;
  for(int i=0; i<m_num_dims; ++i)
    m_filter_dims[i+2] = filter_dims[i];
  m_filter_size = 1;
  for(int i=0; i<m_filter_dims.size(); ++i)
    m_filter_size *= m_filter_dims[i];

  // Get convolution padding and strides
  m_conv_pads.resize(m_num_dims);
  m_conv_strides.resize(m_num_dims);
  for(int i=0; i<m_num_dims; ++i) {
    m_conv_pads[i] = conv_pads[i];
    m_conv_strides[i] = conv_strides[i];
  }

  // Set default algorithms
  m_forward_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  m_forward_work_space_size = 0;
  m_backward_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  m_backward_filter_work_space_size = 0;
  m_backward_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  m_backward_data_work_space_size = 0;

}

cudnn_convolutional_layer::~cudnn_convolutional_layer()
{
  if(m_src_desc)    checkCUDNN(cudnnDestroyTensorDescriptor(m_src_desc));
  if(m_dst_desc)    checkCUDNN(cudnnDestroyTensorDescriptor(m_dst_desc));
  if(m_filter_desc) checkCUDNN(cudnnDestroyFilterDescriptor(m_filter_desc));
  if(m_conv_desc)   checkCUDNN(cudnnDestroyConvolutionDescriptor(m_conv_desc));
}

void cudnn_convolutional_layer::setup()
{

  // Initialize descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&m_src_desc));
  checkCUDNN(cudnnCreateTensorDescriptor(&m_dst_desc));
  checkCUDNN(cudnnCreateFilterDescriptor(&m_filter_desc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&m_conv_desc));

  // Set input tensor descriptor
  m_src_strides = std::vector<int>(m_num_dims+2);
  m_src_strides[m_num_dims + 1]  = 1;
  for(int i=m_num_dims; i>=0; --i)
    m_src_strides[i] = m_src_strides[i+1] * m_src_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_src_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_src_dims.data(),
                                        m_src_strides.data()));

  // Set filter descriptor
  checkCUDNN(cudnnSetFilterNdDescriptor(m_filter_desc,
                                        m_cudnn_data_type,
                                        CUDNN_TENSOR_NCHW,
                                        m_num_dims+2,
                                        m_filter_dims.data()));

  // Set convolution descriptor
  // Note: upscales are not supported as of cuDNN v5.1
  std::vector<int> conv_upscales(m_num_dims, 1);
  checkCUDNN(cudnnSetConvolutionNdDescriptor(m_conv_desc,
                                             m_num_dims,
                                             m_conv_pads.data(),
                                             m_conv_strides.data(),
                                             conv_upscales.data(),
                                             CUDNN_CONVOLUTION,
                                             m_cudnn_data_type));

  // Get output tensor dimensions
  m_dst_dims.resize(m_num_dims+2);
  checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(m_conv_desc,
                                                   m_src_desc,
                                                   m_filter_desc,
                                                   m_num_dims+2,
                                                   m_dst_dims.data()));
  m_dst_size = 1;
  for(int i=1; i<m_dst_dims.size(); ++i)
    m_dst_size *= m_dst_dims[i];
                                 
  // Set output tensor descriptor
  m_dst_strides = std::vector<int>(m_num_dims+2);
  m_dst_strides[m_num_dims + 1]  = 1;
  for(int i=m_num_dims; i>=0; --i)
    m_dst_strides[i] = m_dst_strides[i+1] * m_dst_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_dst_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_dst_dims.data(),
                                        m_dst_strides.data()));

  // Choose forward pass algorithm
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(m_cudnn->m_handles[0],
                                                 m_src_desc,
                                                 m_filter_desc,
                                                 m_conv_desc,
                                                 m_dst_desc,
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                 0,
                                                 &m_forward_algo));
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_cudnn->m_handles[0],
                                                     m_src_desc,
                                                     m_filter_desc,
                                                     m_conv_desc,
                                                     m_dst_desc,
                                                     m_forward_algo,
                                                     &m_forward_work_space_size));

  // Choose filter backward pass algorithm
  checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(m_cudnn->m_handles[0],
                                                        m_src_desc,
                                                        m_dst_desc,
                                                        m_conv_desc,
                                                        m_filter_desc,
                                                        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                        0,
                                                        &m_backward_filter_algo));
  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(m_cudnn->m_handles[0],
                                                            m_src_desc,
                                                            m_dst_desc,
                                                            m_conv_desc,
                                                            m_filter_desc,
                                                            m_backward_filter_algo,
                                                            &m_backward_filter_work_space_size));

  // Choose data backward pass algorithm
  checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(m_cudnn->m_handles[0],
                                                      m_filter_desc,
                                                      m_dst_desc,
                                                      m_conv_desc,
                                                      m_src_desc,
                                                      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                      0,
                                                      &m_backward_data_algo));
  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(m_cudnn->m_handles[0],
                                                          m_filter_desc,
                                                          m_dst_desc,
                                                          m_conv_desc,
                                                          m_src_desc,
                                                          m_backward_data_algo,
                                                          &m_backward_data_work_space_size));
                                                 
}

void cudnn_convolutional_layer::forward(const Mat& src,
                                        const Mat& filter,
                                        const Mat& bias,
                                        Mat& dst)
{

  // Useful constants
  const DataType one = 1.0;
  const DataType zero = 0.0;

  // Adjust input and output tensor dimensions to match input
  const int num_gpus = m_cudnn->m_num_gpus;
  const int mini_batch_size = src.Width();
  const int samples_per_gpu = (mini_batch_size + num_gpus - 1) / num_gpus;
  m_src_dims[0] = samples_per_gpu;
  m_dst_dims[0] = samples_per_gpu;
  checkCUDNN(cudnnSetTensorNdDescriptor(m_src_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_src_dims.data(),
                                        m_src_strides.data()));
  checkCUDNN(cudnnSetTensorNdDescriptor(m_dst_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_dst_dims.data(),
                                        m_dst_strides.data()));

  // GPU memory pointers
  std::vector<DataType*> d_src(num_gpus, NULL);
  std::vector<DataType*> d_filter(num_gpus, NULL);
  std::vector<DataType*> d_bias(num_gpus, NULL);
  std::vector<DataType*> d_dst(num_gpus, NULL);
  std::vector<DataType*> d_work_space(num_gpus, NULL);
  
  // Iterate through GPUs
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));

    // Data samples assigned to GPU
    const int first_pos = i * samples_per_gpu;
    const int last_pos = Min((i+1) * samples_per_gpu, mini_batch_size);
    
    // Allocate memory on GPU
    checkCUDA(cudaMalloc(&d_src[i],
                         m_src_size*samples_per_gpu*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_filter[i],
                         m_filter_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_bias[i],
                         m_dst_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_dst[i],
                         m_dst_size*samples_per_gpu*sizeof(DataType)));
    if(m_forward_work_space_size > 0) {
      checkCUDA(cudaMalloc(&d_work_space[i],
                           m_forward_work_space_size));
    }

    // Transfer filters to GPU
    checkCUDA(cudaMemcpyAsync(d_filter[i],
                              filter.LockedBuffer(),
                              m_filter_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));

    // Transfer inputs to GPU
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(d_src[i] + (pos-first_pos)*m_src_size,
                                src.LockedBuffer(0,pos),
                                m_src_size*sizeof(DataType),
                                cudaMemcpyHostToDevice));
    }

    // Transfer biases to GPU
    checkCUDA(cudaMemcpyAsync(d_bias[i],
                              bias.LockedBuffer(),
                              m_dst_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(d_dst[i] + (pos-first_pos)*m_dst_size,
                                d_bias[i],
                                m_dst_size*sizeof(DataType),
                                cudaMemcpyDeviceToDevice));
    }

    // Perform convolution
    checkCUDNN(cudnnConvolutionForward(m_cudnn->m_handles[i],
                                       &one,
                                       m_src_desc,
                                       d_src[i],
                                       m_filter_desc,
                                       d_filter[i],
                                       m_conv_desc,
                                       m_forward_algo,
                                       d_work_space[i],
                                       m_forward_work_space_size,
                                       &one,
                                       m_dst_desc,
                                       d_dst[i]));
    
    // Transfer outputs from GPU
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(dst.Buffer(0,pos),
                                d_dst[i] + (pos-first_pos)*m_dst_size,
                                m_dst_size*sizeof(DataType),
                                cudaMemcpyDeviceToHost));
    }

  }

  // Free memory on GPU
  // Note: cudaFree is synchronous
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    checkCUDA(cudaFree(d_src[i]));
    checkCUDA(cudaFree(d_filter[i]));
    checkCUDA(cudaFree(d_bias[i]));
    checkCUDA(cudaFree(d_dst[i]));
    checkCUDA(cudaFree(d_work_space[i]));
  }

}

void cudnn_convolutional_layer::backward(const Mat& src,
                                         const Mat& filter,
                                         const Mat& grad_dst,
                                         Mat& grad_filter,
                                         Mat& grad_bias,
                                         Mat& grad_src)
{

  // Useful constants
  const DataType one = 1.0;
  const DataType zero = 0.0;

  // Adjust input and output tensor dimensions to match input
  const int num_gpus = m_cudnn->m_num_gpus;
  const int mini_batch_size = src.Width();
  const int samples_per_gpu = (mini_batch_size + num_gpus - 1) / num_gpus;
  m_src_dims[0] = samples_per_gpu;
  m_dst_dims[0] = samples_per_gpu;
  checkCUDNN(cudnnSetTensorNdDescriptor(m_src_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_src_dims.data(),
                                        m_src_strides.data()));
  checkCUDNN(cudnnSetTensorNdDescriptor(m_dst_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_dst_dims.data(),
                                        m_dst_strides.data()));

  // Compute bias gradient
  Mat ones;
  El::Ones(ones, grad_dst.Width(), El::Int(1));
  El::Gemv(El::NORMAL, DataType(1.0), grad_dst, ones,
           DataType(0.0), grad_bias);

  // GPU memory pointers
  std::vector<DataType*> d_src(num_gpus, NULL);
  std::vector<DataType*> d_filter(num_gpus, NULL);
  std::vector<DataType*> d_grad_dst(num_gpus, NULL);
  std::vector<DataType*> d_grad_filter(num_gpus, NULL);
  std::vector<DataType*> d_grad_src(num_gpus, NULL);
  std::vector<DataType*> d_filter_work_space(num_gpus, NULL);
  std::vector<DataType*> d_data_work_space(num_gpus, NULL);

  // Iterate through GPUs
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));

    // Data samples assigned to GPU
    const int first_pos = i * samples_per_gpu;
    const int last_pos = Min((i+1) * samples_per_gpu, mini_batch_size);

    // Allocate memory on GPU
    checkCUDA(cudaMalloc(&d_src[i],
                         m_src_size*samples_per_gpu*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_filter[i],
                         m_filter_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_dst[i],
                         m_dst_size*samples_per_gpu*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_filter[i],
                         m_filter_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_src[i],
                         m_src_size*samples_per_gpu*sizeof(DataType)));
    if(m_backward_filter_work_space_size > 0) {
      checkCUDA(cudaMalloc(&d_filter_work_space[i],
                           m_backward_filter_work_space_size));
    }
    if(m_backward_data_work_space_size > 0) {
      checkCUDA(cudaMalloc(&d_data_work_space[i],
                           m_backward_data_work_space_size));
    }

    // Transfer filters to GPU
    checkCUDA(cudaMemcpyAsync(d_filter[i],
                              filter.LockedBuffer(),
                              m_filter_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));

    // Transfer inputs and error signal to GPU
    checkCUDA(cudaMemsetAsync(d_grad_dst[i],
                              0,
                              m_dst_size*samples_per_gpu*sizeof(DataType)));
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(d_src[i] + (pos-first_pos)*m_src_size,
                                src.LockedBuffer(0,pos),
                                m_src_size*sizeof(DataType),
                                cudaMemcpyHostToDevice));
      checkCUDA(cudaMemcpyAsync(d_grad_dst[i] + (pos-first_pos)*m_dst_size,
                                grad_dst.LockedBuffer(0,pos),
                                m_dst_size*sizeof(DataType),
                                cudaMemcpyHostToDevice));
    }

    // Compute filter gradient
    checkCUDNN(cudnnConvolutionBackwardFilter(m_cudnn->m_handles[i],
                                              &one,
                                              m_src_desc,
                                              d_src[i],
                                              m_dst_desc,
                                              d_grad_dst[i],

                                              m_conv_desc,
                                              m_backward_filter_algo,
                                              d_filter_work_space[i],
                                              m_backward_filter_work_space_size,
                                              &zero,
                                              m_filter_desc,
                                              d_grad_filter[i]));

    // Compute error signal to "next" layer
    checkCUDNN(cudnnConvolutionBackwardData(m_cudnn->m_handles[i],
                                            &one,
                                            m_filter_desc,
                                            d_filter[i],
                                            m_dst_desc,
                                            d_grad_dst[i],
                                            m_conv_desc,
                                            m_backward_data_algo,
                                            d_data_work_space[i],
                                            m_backward_data_work_space_size,
                                            &zero,
                                            m_src_desc,
                                            d_grad_src[i]));
    
    // Transfer data from GPU
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(grad_src.Buffer(0,pos),
                                d_grad_src[i] + (pos-first_pos)*m_src_size,
                                m_src_size*sizeof(DataType),
                                cudaMemcpyDeviceToHost));
    }

  }
  
  // Transfer and accumulate filter gradients from GPUs
  Mat temp(grad_filter.Height(), grad_filter.Width());
  El::Zero(grad_filter);
  El::Zero(temp);
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    checkCUDA(cudaMemcpy(temp.Buffer(),
                         d_grad_filter[i],
                         m_filter_size*sizeof(DataType),
                         cudaMemcpyDeviceToHost));
    grad_filter += temp;
  }

  // Free memory on GPU
  // Note: cudaFree is synchronous
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    checkCUDA(cudaFree(d_src[i]));
    checkCUDA(cudaFree(d_filter[i]));
    checkCUDA(cudaFree(d_grad_dst[i]));
    checkCUDA(cudaFree(d_grad_filter[i]));
    checkCUDA(cudaFree(d_grad_src[i]));
    checkCUDA(cudaFree(d_filter_work_space[i]));
    checkCUDA(cudaFree(d_data_work_space[i]));
  }

}

cudnn_pooling_layer::cudnn_pooling_layer(const int  num_dims,
                                         const int  channels,
                                         const int* src_dims,
                                         const int  pool_mode,
                                         const int* pool_dims,
                                         const int* pool_pads,
                                         const int* pool_strides,
                                         cudnn_manager* cudnn)
  : m_num_dims(num_dims), m_cudnn(cudnn),
    m_cudnn_data_type(get_cudnn_data_type<DataType>()),
    m_pool_mode(get_cudnn_pool_mode(pool_mode)),
    m_src_desc(NULL), m_dst_desc(NULL), m_pool_desc(NULL)
{
  
  // Get input tensor dimensions
  m_src_dims.resize(m_num_dims+2);
  m_src_dims[0] = 1;
  m_src_dims[1] = channels;
  for(int i=0; i<m_num_dims; ++i)
    m_src_dims[i+2] = src_dims[i];
  m_src_size = 1;
  for(int i=0; i<m_src_dims.size(); ++i)
    m_src_size *= m_src_dims[i];

  // Get pooling parameters
  m_pool_dims.resize(m_num_dims);
  m_pool_pads.resize(m_num_dims);
  m_pool_strides.resize(m_num_dims);
  for(int i=0; i<m_num_dims; ++i) {
    m_pool_dims[i] = pool_dims[i];
    m_pool_pads[i] = pool_pads[i];
    m_pool_strides[i] = pool_strides[i];
  }

}

cudnn_pooling_layer::~cudnn_pooling_layer()
{
  if(m_src_desc)  checkCUDNN(cudnnDestroyTensorDescriptor(m_src_desc));
  if(m_dst_desc)  checkCUDNN(cudnnDestroyTensorDescriptor(m_dst_desc));
  if(m_pool_desc) checkCUDNN(cudnnDestroyPoolingDescriptor(m_pool_desc));
}

void cudnn_pooling_layer::setup()
{

  // Initialize descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&m_src_desc));
  checkCUDNN(cudnnCreateTensorDescriptor(&m_dst_desc));
  checkCUDNN(cudnnCreatePoolingDescriptor(&m_pool_desc));

  // Set input tensor descriptor
  m_src_strides = std::vector<int>(m_num_dims+2);
  m_src_strides[m_num_dims + 1]  = 1;
  for(int i=m_num_dims; i>=0; --i)
    m_src_strides[i] = m_src_strides[i+1] * m_src_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_src_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_src_dims.data(),
                                        m_src_strides.data()));

  // Set pooling descriptor
  checkCUDNN(cudnnSetPoolingNdDescriptor(m_pool_desc,
                                         m_pool_mode,
                                         CUDNN_PROPAGATE_NAN,
                                         m_num_dims,
                                         m_pool_dims.data(),
                                         m_pool_pads.data(),
                                         m_pool_strides.data()));

  // Get output tensor dimensions
  m_dst_dims.resize(m_num_dims+2);
  checkCUDNN(cudnnGetPoolingNdForwardOutputDim(m_pool_desc,
                                               m_src_desc,
                                               m_num_dims+2,
                                               m_dst_dims.data()));
  m_dst_size = 1;
  for(int i=0; i<m_dst_dims.size(); ++i)
    m_dst_size *= m_dst_dims[i];

  // Set output tensor descriptor
  m_dst_strides= std::vector<int>(m_num_dims+2);
  m_dst_strides[m_num_dims + 1]  = 1;
  for(int i=m_num_dims; i>=0; --i)
    m_dst_strides[i] = m_dst_strides[i+1] * m_dst_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_dst_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_dst_dims.data(),
                                        m_dst_strides.data()));

}

void cudnn_pooling_layer::forward(const Mat& src, Mat& dst)
{

  // Useful constants
  const DataType one = 1.0;
  const DataType zero = 0.0;

  // Adjust input and output tensor dimensions to match input
  const int num_gpus = m_cudnn->m_num_gpus;
  const int mini_batch_size = src.Width();
  const int samples_per_gpu = (mini_batch_size + num_gpus - 1) / num_gpus;
  m_src_dims[0] = samples_per_gpu;
  m_dst_dims[0] = samples_per_gpu;
  checkCUDNN(cudnnSetTensorNdDescriptor(m_src_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_src_dims.data(),
                                        m_src_strides.data()));
  checkCUDNN(cudnnSetTensorNdDescriptor(m_dst_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_dst_dims.data(),
                                        m_dst_strides.data()));

  // GPU memory pointers
  std::vector<DataType*> d_src(num_gpus, NULL);
  std::vector<DataType*> d_dst(num_gpus, NULL);
  
  // Iterate through GPUs
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
 
    // Data samples assigned to GPU
    const int first_pos = i * samples_per_gpu;
    const int last_pos = Min((i+1) * samples_per_gpu, mini_batch_size);

    // Allocate memory on GPU
    checkCUDA(cudaMalloc(&d_src[i],
                         m_src_size*samples_per_gpu*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_dst[i],
                         m_dst_size*samples_per_gpu*sizeof(DataType)));

    // Transfer inputs to GPU
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(d_src[i] + (pos-first_pos)*m_src_size,
                                src.LockedBuffer(0,pos),
                                m_src_size*sizeof(DataType),
                                cudaMemcpyHostToDevice));
    }

    // Perform pooling
    checkCUDNN(cudnnPoolingForward(m_cudnn->m_handles[i],
                                   m_pool_desc,
                                   &one,
                                   m_src_desc,
                                   d_src[i],
                                   &zero,
                                   m_dst_desc,
                                   d_dst[i]));

    // Transfer outputs from GPU
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(dst.Buffer(0,pos),
                                d_dst[i] + (pos-first_pos)*m_dst_size,
                                m_dst_size*sizeof(DataType),
                                cudaMemcpyDeviceToHost));
    }

  }

  // Free memory on GPU
  // Note: cudaFree is synchronous
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    checkCUDA(cudaFree(d_src[i]));
    checkCUDA(cudaFree(d_dst[i]));
  }

}

void cudnn_pooling_layer::backward(const Mat& src,
                                   const Mat& dst,
                                   const Mat& grad_dst,
                                   Mat& grad_src)
{

  // Useful constants
  const DataType one = 1.0;
  const DataType zero = 0.0;

  // Adjust input and output tensor dimensions to match input
  const int num_gpus = m_cudnn->m_num_gpus;
  const int mini_batch_size = src.Width();
  const int samples_per_gpu = (mini_batch_size + num_gpus - 1) / num_gpus;
  m_src_dims[0] = samples_per_gpu;
  m_dst_dims[0] = samples_per_gpu;
  checkCUDNN(cudnnSetTensorNdDescriptor(m_src_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_src_dims.data(),
                                        m_src_strides.data()));
  checkCUDNN(cudnnSetTensorNdDescriptor(m_dst_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_dst_dims.data(),
                                        m_dst_strides.data()));

  // GPU memory pointers
  std::vector<DataType*> d_src(num_gpus, NULL);
  std::vector<DataType*> d_dst(num_gpus, NULL);
  std::vector<DataType*> d_grad_dst(num_gpus, NULL);
  std::vector<DataType*> d_grad_src(num_gpus, NULL);

  // Iterate through GPUs
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));

    // Data samples assigned to GPU
    const int first_pos = i * samples_per_gpu;
    const int last_pos = Min((i+1) * samples_per_gpu, mini_batch_size);

    // Allocate memory on GPU
    checkCUDA(cudaMalloc(&d_src[i],
                         m_src_size*samples_per_gpu*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_dst[i],
                         m_dst_size*samples_per_gpu*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_dst[i],
                         m_dst_size*samples_per_gpu*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_src[i],
                         m_src_size*samples_per_gpu*sizeof(DataType)));

    // Transfer data to GPU
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(d_src[i] + (pos-first_pos)*m_src_size,
                                src.LockedBuffer(0,pos),
                                m_src_size*sizeof(DataType),
                                cudaMemcpyHostToDevice));
      checkCUDA(cudaMemcpyAsync(d_dst[i] + (pos-first_pos)*m_dst_size,
                                dst.LockedBuffer(0,pos),
                                m_dst_size*sizeof(DataType),
                                cudaMemcpyHostToDevice));
      checkCUDA(cudaMemcpyAsync(d_grad_dst[i] + (pos-first_pos)*m_dst_size,
                                grad_dst.LockedBuffer(0,pos),
                                m_dst_size*sizeof(DataType),
                                cudaMemcpyHostToDevice));
    }

    // Compute error signal
    checkCUDNN(cudnnPoolingBackward(m_cudnn->m_handles[i],
                                    m_pool_desc,
                                    &one,
                                    m_dst_desc,
                                    d_dst[i],
                                    m_dst_desc,
                                    d_grad_dst[i],
                                    m_src_desc,
                                    d_src[i],
                                    &zero,
                                    m_src_desc,
                                    d_grad_src[i]));

    // Transfer data from GPU
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(grad_src.Buffer(0,pos),
                                d_grad_src[i] + (pos-first_pos)*m_src_size,
                                m_src_size*sizeof(DataType),
                                cudaMemcpyDeviceToHost));
    }

  }

  // Free memory on GPU
  // Note: cudaFree is synchronous
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    checkCUDA(cudaFree(d_src[i]));
    checkCUDA(cudaFree(d_dst[i]));
    checkCUDA(cudaFree(d_grad_dst[i]));
    checkCUDA(cudaFree(d_grad_src[i]));
  }

}

#endif // __LIB_CUDNN
