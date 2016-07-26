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

#include <iostream>

#include "lbann/utils/cudnn_wrapper.hpp"

#ifdef __LIB_CUDNN

using namespace cudnn;

// Error utility macros
#define checkCUDA(status) {                                             \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  /* TODO: remove */ \
      cudaDeviceReset();                                                \
      exit(-1);                                                         \
    }                                                                   \
  }
#define checkCUDNN(status) {                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
      std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n"; /* TODO: remove */ \
      cudaDeviceReset();                                                \
      exit(-1);                                                         \
    }                                                                   \
  }

/// Determine number of GPUs
/** If num_gpus<0, then report total number of availabel GPUs */
int get_num_gpus(int num_gpus)
{
  if(num_gpus < 0)
    checkCUDA(cudaGetDeviceCount(&num_gpus));
  return num_gpus;
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
  default:
    std::cerr << "Error: invalid data type for cuDNN\n";
    exit(EXIT_FAILURE);
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
  default:
    std::cerr << "Warning: pooling mode " << pool_mode << " is unknown. Using max pooling instead.\n";
    return CUDNN_POOLING_MAX;
  }
}

cudnn_manager::cudnn_manager(const int num_gpus)
  : m_num_gpus(get_num_gpus(num_gpus))
{

  // Check that at least one GPU is allocated
  if(m_num_gpus < 1) {
    // TODO: consider throwing an exception instead
    std::cerr << "Error: no GPUs allocated or found for cuDNN\n";
    exit(EXIT_FAILURE);
  }

  // Initialize cuDNN on each GPU
  m_handles.resize(m_num_gpus, NULL);
  for(int dev=0; dev<m_num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDNN(cudnnCreate(&m_handles[dev]));
  }

}

cudnn_manager::~cudnn_manager()
{
  for(int dev=0; dev<m_handles.size(); ++dev) {
    if(m_handles[dev]) {
      checkCUDA(cudaSetDevice(dev));
      checkCUDNN(cudnnDestroy(m_handles[dev]));
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
                            cudnn_manager* cudnn)
  : m_num_dims(num_dims), m_cudnn(cudnn),
    m_cudnn_data_type(get_cudnn_data_type<DataType>()),
    m_src_desc(NULL), m_dst_desc(NULL),
    m_filter_desc(NULL), m_conv_desc(NULL)
{

  // Get input tensor dimensions
  m_src_dims.resize(m_num_dims+2);
  m_src_dims[0] = 1;
  m_src_dims[1] = src_channels;
  for(int i=0; i<m_num_dims; ++i)
    m_src_dims[i+2] = src_dims[i];
  m_src_size = 1;
  for(int i=0; i<m_src_dims.size(); ++i)
    m_src_size *= m_src_dims[i];

  // Get output tensor dimensions
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
  std::vector<int> src_strides(m_num_dims+2);
  src_strides[m_num_dims + 1]  = 1;
  for(int i=m_num_dims; i>=0; --i)
    src_strides[i] = src_strides[i+1] * m_src_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_src_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_src_dims.data(),
                                        src_strides.data()));

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
  for(int i=0; i<m_dst_dims.size(); ++i)
    m_dst_size *= m_dst_dims[i];
                                 
  // Set output tensor descriptor
  std::vector<int> dst_strides(m_num_dims+2);
  dst_strides[m_num_dims + 1]  = 1;
  for(int i=m_num_dims; i>=0; --i)
    dst_strides[i] = dst_strides[i+1] * m_dst_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_dst_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_dst_dims.data(),
                                        dst_strides.data()));

  // Choose forward pass algorithm
  // Note: assume all GPUs are identical to GPU 0
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
  // Note: assume all GPUs are identical to GPU 0
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
  // Note: assume all GPUs are identical to GPU 0
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
                                        Mat& dst)
{

  // Useful constants
  const DataType one = 1.0;
  const DataType zero = 0.0;
  const int num_gpus = m_cudnn->m_num_gpus;
  
  // Allocate memory on GPUs
  std::vector<DataType*> d_src(num_gpus, NULL);
  std::vector<DataType*> d_filter(num_gpus, NULL);
  std::vector<DataType*> d_dst(num_gpus, NULL);
  std::vector<DataType*> d_work_space(num_gpus, NULL);
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaMalloc(&d_src[dev], m_src_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_filter[dev], m_filter_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_dst[dev], m_dst_size*sizeof(DataType)));
    if(m_forward_work_space_size > 0)
      checkCUDA(cudaMalloc(&d_work_space[dev], m_forward_work_space_size));
  }

  // Transfer filter data to GPU
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaMemcpyAsync(d_filter[dev],
                              filter.LockedBuffer(),
                              m_filter_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));
  }

  // Perform convolution with each data sample
  for(int j=0; j<src.Width(); ++j) {
    
    // Determine GPU
    const int dev = j % num_gpus;
    checkCUDA(cudaSetDevice(dev));

    // Transfer input data to GPU
    checkCUDA(cudaMemcpyAsync(d_src[dev],
                              src.LockedBuffer(0,j),
                              m_src_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));

    // Perform convolution
    checkCUDNN(cudnnConvolutionForward(m_cudnn->m_handles[dev],
                                       &one,
                                       m_src_desc,
                                       d_src[dev],
                                       m_filter_desc,
                                       d_filter[dev],
                                       m_conv_desc,
                                       m_forward_algo,
                                       d_work_space[dev],
                                       m_forward_work_space_size,
                                       &zero,
                                       m_dst_desc,
                                       d_dst[dev]));
    
    // Transfer output data from GPU
    checkCUDA(cudaMemcpyAsync(dst.Buffer(0,j),
                              d_dst[dev],
                              m_dst_size*sizeof(DataType),
                              cudaMemcpyDeviceToHost));

  }

  // Free memory on GPU
  // Note: cudaFree is synchronous
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaFree(d_src[dev]));
    checkCUDA(cudaFree(d_filter[dev]));
    checkCUDA(cudaFree(d_dst[dev]));
    checkCUDA(cudaFree(d_work_space[dev]));
  }

}

void cudnn_convolutional_layer::backward(const Mat& src,
                                         const Mat& filter,
                                         const Mat& grad_dst,
                                         Mat& grad_filter,
                                         Mat& grad_src)
{

  // Useful constants
  const DataType one = 1.0;
  const DataType zero = 0.0;
  const int num_gpus = m_cudnn->m_num_gpus;

  // Allocate memory on GPUs
  std::vector<DataType*> d_src(num_gpus, NULL);
  std::vector<DataType*> d_filter(num_gpus, NULL);
  std::vector<DataType*> d_grad_dst(num_gpus, NULL);
  std::vector<DataType*> d_grad_filter(num_gpus, NULL);
  std::vector<DataType*> d_grad_src(num_gpus, NULL);
  std::vector<DataType*> d_filter_work_space(num_gpus, NULL);
  std::vector<DataType*> d_data_work_space(num_gpus, NULL);
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaMalloc(&d_src[dev], m_src_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_filter[dev], m_filter_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_dst[dev], m_dst_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_filter[dev], m_filter_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_src[dev], m_src_size*sizeof(DataType)));
    if(m_backward_filter_work_space_size > 0)
      checkCUDA(cudaMalloc(&d_filter_work_space[dev],
                           m_backward_filter_work_space_size));
    if(m_backward_data_work_space_size > 0)
      checkCUDA(cudaMalloc(&d_data_work_space[dev],
                           m_backward_data_work_space_size));
  }

  // Initialize filter and filter gradient on GPU
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaMemcpyAsync(d_filter[dev],
                              filter.LockedBuffer(),
                              m_filter_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));
    checkCUDA(cudaMemsetAsync(d_grad_filter[dev],
                              0,
                              m_filter_size*sizeof(DataType)));
  }

  // Compute gradients for each data sample
  for(int j=0; j<src.Width(); ++j) {
    
    // Determine GPU
    const int dev = j % num_gpus;
    checkCUDA(cudaSetDevice(dev));

    // Transfer data to GPU
    checkCUDA(cudaMemcpyAsync(d_src[dev],
                              src.LockedBuffer(0,j),
                              m_src_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyAsync(d_grad_dst[dev],
                              grad_dst.LockedBuffer(0,j),
                              m_dst_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));

    // Compute gradient w.r.t. filter
    checkCUDNN(cudnnConvolutionBackwardFilter(m_cudnn->m_handles[dev],
                                              &one,
                                              m_src_desc,
                                              d_src[dev],
                                              m_dst_desc,
                                              d_grad_dst[dev],
                                              m_conv_desc,
                                              m_backward_filter_algo,
                                              d_filter_work_space[dev],
                                              m_backward_filter_work_space_size,
                                              &one,
                                              m_filter_desc,
                                              d_grad_filter[dev]));

    // Compute gradient w.r.t. input
    checkCUDNN(cudnnConvolutionBackwardData(m_cudnn->m_handles[dev],
                                            &one,
                                            m_filter_desc,
                                            d_filter[dev],
                                            m_dst_desc,
                                            d_grad_dst[dev],
                                            m_conv_desc,
                                            m_backward_data_algo,
                                            d_data_work_space[dev],
                                            m_backward_data_work_space_size,
                                            &zero,
                                            m_src_desc,
                                            d_grad_src[dev]));
    
    // Transfer data from GPU
    checkCUDA(cudaMemcpyAsync(grad_src.Buffer(0,j),
                              d_grad_src[dev],
                              m_src_size*sizeof(DataType),
                              cudaMemcpyDeviceToHost));

  }
  
  // Transfer and accumulate filter gradients from GPUs
  Mat temp(grad_filter.Height(), grad_filter.Width());
  El::Zero(grad_filter);
  El::Zero(temp);
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaMemcpy(temp.Buffer(),
                         d_grad_filter[dev],
                         m_filter_size*sizeof(DataType),
                         cudaMemcpyDeviceToHost));
    grad_filter += temp;
  }

  // Free memory on GPU
  // Note: cudaFree is synchronous
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaFree(d_src[dev]));
    checkCUDA(cudaFree(d_filter[dev]));
    checkCUDA(cudaFree(d_grad_dst[dev]));
    checkCUDA(cudaFree(d_grad_filter[dev]));
    checkCUDA(cudaFree(d_grad_src[dev]));
    checkCUDA(cudaFree(d_filter_work_space[dev]));
    checkCUDA(cudaFree(d_data_work_space[dev]));
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
  std::vector<int> src_strides(m_num_dims+2);
  src_strides[m_num_dims + 1]  = 1;
  for(int i=m_num_dims; i>=0; --i)
    src_strides[i] = src_strides[i+1] * m_src_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_src_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_src_dims.data(),
                                        src_strides.data()));

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
  std::vector<int> dst_strides(m_num_dims+2);
  dst_strides[m_num_dims + 1]  = 1;
  for(int i=m_num_dims; i>=0; --i)
    dst_strides[i] = dst_strides[i+1] * m_dst_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_dst_desc,
                                        m_cudnn_data_type,
                                        m_num_dims+2,
                                        m_dst_dims.data(),
                                        dst_strides.data()));

}

void cudnn_pooling_layer::forward(const Mat& src, Mat& dst)
{

  // Useful constants
  const DataType one = 1.0;
  const DataType zero = 0.0;
  const int num_gpus = m_cudnn->m_num_gpus;
  
  // Allocate memory on GPUs
  std::vector<DataType*> d_src(num_gpus, NULL);
  std::vector<DataType*> d_dst(num_gpus, NULL);
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaMalloc(&d_src[dev], m_src_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_dst[dev], m_dst_size*sizeof(DataType)));
  }

  // Perform pooling with each data sample
  for(int j=0; j<src.Width(); ++j) {
    
    // Determine GPU
    const int dev = j % num_gpus;
    checkCUDA(cudaSetDevice(dev));

    // Transfer input data to GPU
    checkCUDA(cudaMemcpyAsync(d_src[dev],
                              src.LockedBuffer(0,j),
                              m_src_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));

    // Perform pooling
    checkCUDNN(cudnnPoolingForward(m_cudnn->m_handles[dev],
                                   m_pool_desc,
                                   &one,
                                   m_src_desc,
                                   d_src[dev],
                                   &zero,
                                   m_dst_desc,
                                   d_dst[dev]));
    
    // Transfer output data from GPU
    checkCUDA(cudaMemcpyAsync(dst.Buffer(0,j),
                              d_dst[dev],
                              m_dst_size*sizeof(DataType),
                              cudaMemcpyDeviceToHost));

  }

  // Free memory on GPU
  // Note: cudaFree is synchronous
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaFree(d_src[dev]));
    checkCUDA(cudaFree(d_dst[dev]));
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
  const int num_gpus = m_cudnn->m_num_gpus;

  // Allocate memory on GPUs
  std::vector<DataType*> d_src(num_gpus, NULL);
  std::vector<DataType*> d_dst(num_gpus, NULL);
  std::vector<DataType*> d_grad_dst(num_gpus, NULL);
  std::vector<DataType*> d_grad_src(num_gpus, NULL);
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaMalloc(&d_src[dev], m_src_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_dst[dev], m_dst_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_dst[dev], m_dst_size*sizeof(DataType)));
    checkCUDA(cudaMalloc(&d_grad_src[dev], m_src_size*sizeof(DataType)));
  }

  // Compute gradients for each data sample
  for(int j=0; j<src.Width(); ++j) {
    
    // Determine GPU
    const int dev = j % num_gpus;
    checkCUDA(cudaSetDevice(dev));

    // Transfer data to GPU
    checkCUDA(cudaMemcpyAsync(d_src[dev],
                              src.LockedBuffer(0,j),
                              m_src_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyAsync(d_dst[dev],
                              dst.LockedBuffer(0,j),
                              m_dst_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyAsync(d_grad_dst[dev],
                              grad_dst.LockedBuffer(0,j),
                              m_dst_size*sizeof(DataType),
                              cudaMemcpyHostToDevice));

    // Compute gradient w.r.t. input
    checkCUDNN(cudnnPoolingBackward(m_cudnn->m_handles[dev],
                                    m_pool_desc,
                                    &one,
                                    m_dst_desc,
                                    d_dst[dev],
                                    m_dst_desc,
                                    d_grad_dst[dev],
                                    m_src_desc,
                                    d_src[dev],
                                    &zero,
                                    m_src_desc,
                                    d_grad_src[dev]));
    
    // Transfer data from GPU
    checkCUDA(cudaMemcpyAsync(grad_src.Buffer(0,j),
                              d_grad_src[dev],
                              m_src_size*sizeof(DataType),
                              cudaMemcpyDeviceToHost));

  }

  // Free memory on GPU
  // Note: cudaFree is synchronous
  for(int dev=0; dev<num_gpus; ++dev) {
    checkCUDA(cudaSetDevice(dev));
    checkCUDA(cudaFree(d_src[dev]));
    checkCUDA(cudaFree(d_dst[dev]));
    checkCUDA(cudaFree(d_grad_dst[dev]));
    checkCUDA(cudaFree(d_grad_src[dev]));
  }

}

#endif // __LIB_CUDNN
