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

#define _ALLOC_DEVICE_MEM_ONCE_

// Error utility macros
#define checkCUDNN(status) {                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
      std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n"; /* TODO: remove */ \
      cudaDeviceReset();                                                \
      throw lbann::lbann_exception("cudnn_wrapper: cuDNN error");       \
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
cudnnPoolingMode_t get_cudnn_pool_mode(const pool_mode mode)
{
  switch(mode) {
  case pool_mode::max:
    return CUDNN_POOLING_MAX;
  case pool_mode::average:
    return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  case pool_mode::average_no_pad:
    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  default:
    throw lbann::lbann_exception("cudnn_wrapper: invalid pooling mode");
  }
}

cudnn_manager::cudnn_manager(lbann::lbann_comm* _comm, int max_num_gpus)
  : comm(_comm)
{

  // Initialize GPU memory pool
  m_gpu_memory = new cub::CachingDeviceAllocator(8u, 3u);

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
      m_streams.push_back(NULL);
      m_handles.push_back(NULL);
      cudaStream_t& stream = m_streams.back();
      cudnnHandle_t& handle = m_handles.back();
      checkCUDA(cudaStreamCreate(&stream));
      checkCUDNN(cudnnCreate(&handle));
      checkCUDNN(cudnnSetStream(handle, stream));
      gpu += procs_per_node;
    }
  }

  // Case where compute node has fewers GPUs than MPI ranks
  // TODO: smarter way to allocate GPUs to MPI ranks
  // TODO: we get CUDNN_STATUS_INTERNAL_ERROR when creating cuDNN handle
  else {
    const int gpu = rank_in_node % m_num_total_gpus;
    checkCUDA(cudaSetDevice(gpu));
    m_gpus.push_back(gpu);
    m_streams.push_back(NULL);
    m_handles.push_back(NULL);
    cudaStream_t& stream = m_streams.back();
    cudnnHandle_t& handle = m_handles.back();
    checkCUDA(cudaStreamCreate(&stream));
    checkCUDNN(cudnnCreate(&handle));
    checkCUDNN(cudnnSetStream(handle, stream));
  }

  // Get number of GPUs for current MPI rank
  m_num_gpus = m_gpus.size();

}

cudnn_manager::~cudnn_manager()
{
  // Destroy GPU memory pool
  if(m_gpu_memory) {
    delete m_gpu_memory;
  }

  // Destroy cuDNN handles
  for(int i=0; i<m_gpus.size(); ++i) {
    checkCUDA(cudaSetDevice(m_gpus[i]));
    if(m_streams[i]) {
      checkCUDA(cudaStreamDestroy(m_streams[i]));
    }
    if(m_handles[i]) {
      checkCUDNN(cudnnDestroy(m_handles[i]));
    }
  }
  unpin_ptrs();
}

void cudnn_manager::pin_ptr(void* ptr, size_t sz)
{
  if (!ptr) return;
  std::map<void*, size_t>::iterator it = pinned_ptr.find(ptr);
  if (it == pinned_ptr.end()) {
    //std::cout << "adding a new ptr " << reinterpret_cast<unsigned long long>(ptr) << std::endl;
    pinned_ptr[ptr] = sz;
    checkCUDA(cudaHostRegister(ptr, sz, cudaHostRegisterPortable));
  } else {
    // TODO: We can check here if the block defined by (ptr,sz) overlaps with an existing one.
  }
}

size_t cudnn_manager::pin_memory_block(ElMat *mat)
{
  if (!mat) return static_cast<size_t>(0u);
  const int w = (mat->Matrix()).Width();
  const int h = (mat->Matrix()).Height();
  const int sz = w*h*sizeof(DataType);
  void* ptr = (void*) (mat->Matrix()).Buffer();
  pin_ptr(ptr, sz);
  return static_cast<size_t>(sz);
}

void cudnn_manager::unpin_memory_block(ElMat *mat)
{
  if (!mat) return;
  unpin_ptr(reinterpret_cast<void*>((mat->Matrix()).Buffer()));
}

void cudnn_manager::unpin_ptr(void* const ptr)
{
  std::map<void*, size_t>::iterator it = pinned_ptr.find(ptr);
  if (it != pinned_ptr.end()) {
    checkCUDA(cudaHostUnregister(it->first));
    pinned_ptr.erase(it);
  }
}

void cudnn_manager::unpin_ptrs(void)
{
  std::map<void*, size_t>::iterator it = pinned_ptr.begin();
  std::map<void*, size_t>::iterator itend = pinned_ptr.end();

  for(; it != itend; ++it) {
    checkCUDA(cudaHostUnregister(it->first));
  }
  pinned_ptr.clear();
}

size_t cudnn_manager::get_total_size_of_pinned_blocks(void) const
{
  std::map<void*, size_t>::const_iterator it = pinned_ptr.begin();
  std::map<void*, size_t>::const_iterator itend = pinned_ptr.end();

  size_t total = 0u;
  for(; it != itend; ++it) total += it->second;
  return total;
}

void cudnn_manager::print_version() const {
  std::cout << "cudnnGetVersion() : " << (int)cudnnGetVersion() << " , "
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
                            const uint mini_batch_size,
                            cudnn_manager* cudnn)
  : m_num_dims(num_dims), m_cudnn(cudnn),
    m_cudnn_data_type(get_cudnn_data_type<DataType>()),
    m_src_desc(NULL), m_dst_desc(NULL),
    m_filter_desc(NULL), m_conv_desc(NULL),
    m_mini_batch_size(mini_batch_size)
{

  // Get number of GPUs
  const int num_gpus = m_cudnn->m_num_gpus;

  // Get number of samples per GPU
  const int num_processes = m_cudnn->comm->get_procs_per_model();
  const int local_mini_batch_size = (mini_batch_size + num_processes - 1) / num_processes;
  m_samples_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

  // Get input tensor dimensions
  m_src_dims.resize(m_num_dims+2);
  m_src_dims[0] = m_samples_per_gpu;
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
#ifdef _ALLOC_DEVICE_MEM_ONCE_
  device_deallocate();
#endif
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
#ifdef _ALLOC_DEVICE_MEM_ONCE_
  const int num_gpus = m_cudnn->m_num_gpus;
  m_dst_dims[0] = m_samples_per_gpu;
#endif

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
#ifdef _ALLOC_DEVICE_MEM_ONCE_
  device_allocate();
#endif
  temp.Resize(m_filter_size, m_cudnn->m_num_gpus);
  m_cudnn->pin_ptr((void*) temp.Buffer(), temp.Height()*temp.Width()*sizeof(DataType));
}

void cudnn_convolutional_layer::device_allocate(void)
{
    device_allocate_for_forward();
    device_allocate_for_backward();
}

void cudnn_convolutional_layer::device_allocate_for_forward(void)
{
  const int num_gpus = m_cudnn->m_num_gpus;

  d_src.clear();
  d_src.assign(num_gpus, NULL);
  d_filter.clear();
  d_filter.assign(num_gpus, NULL);
  d_bias.clear();
  d_bias.assign(num_gpus, NULL);
  d_dst.clear();
  d_dst.assign(num_gpus, NULL);
  d_work_space.clear();
  d_work_space.assign(num_gpus, NULL);

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_src[i],
                                        m_src_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_filter[i],
                                        m_filter_size*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_bias[i],
                                        m_dst_size*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_dst[i],
                                        m_dst_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
    if(m_forward_work_space_size > 0) {
      checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                          (void**) &d_work_space[i],
                                          m_forward_work_space_size,
                                          stream));
    }
  }
}

void cudnn_convolutional_layer::device_allocate_for_backward(void)
{
  const int num_gpus = m_cudnn->m_num_gpus;

#ifndef _ALLOC_DEVICE_MEM_ONCE_
  d_src.clear();
  d_src.assign(num_gpus, NULL);
  d_filter.clear();
  d_filter.assign(num_gpus, NULL);
  d_prev_error_signal.clear();
  d_prev_error_signal.assign(num_gpus, NULL);
#else // Otherwise reuse those blocks allocated for forward()
  d_prev_error_signal = d_dst;
#endif // end of ifndef _ALLOC_DEVICE_MEM_ONCE_
  d_filter_gradient.clear();
  d_filter_gradient.assign(num_gpus, NULL);
  d_error_signal.clear();
  d_error_signal.assign(num_gpus, NULL);
  d_filter_work_space.clear();
  d_filter_work_space.assign(num_gpus, NULL);
  d_data_work_space.clear();
  d_data_work_space.assign(num_gpus, NULL);

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
  #ifndef _ALLOC_DEVICE_MEM_ONCE_
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_src[i],
                                        m_src_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_filter[i],
                                        m_filter_size*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_prev_error_signal[i],
                                        m_dst_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
  #endif // end of ifndef _ALLOC_DEVICE_MEM_ONCE_
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_filter_gradient[i],
                                        m_filter_size*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_error_signal[i],
                                        m_src_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
    if(m_backward_filter_work_space_size > 0) {
      checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                          (void**) &d_filter_work_space[i],
                                          m_backward_filter_work_space_size,
                                          stream));
    }
    if(m_backward_data_work_space_size > 0) {
      checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                          (void**) &d_data_work_space[i],
                                          m_backward_data_work_space_size,
                                          stream));
    }
  }
}

void cudnn_convolutional_layer::device_deallocate(void)
{
  device_deallocate_for_forward();
  device_deallocate_for_backward();
}

void cudnn_convolutional_layer::device_deallocate_for_forward(void)
{
  const int num_gpus = m_cudnn->m_num_gpus;

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
  #ifndef _ALLOC_DEVICE_MEM_ONCE_
    checkCUDA(gpu_memory.DeviceFree(gpu, d_src[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_filter[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_dst[i]));
  #endif // end of ifndef _ALLOC_DEVICE_MEM_ONCE_
    checkCUDA(gpu_memory.DeviceFree(gpu, d_bias[i]));
    if(d_work_space[i]) {
      checkCUDA(gpu_memory.DeviceFree(gpu, d_work_space[i]));
    }
  }
  #ifndef _ALLOC_DEVICE_MEM_ONCE_
  d_src.clear();
  d_filter.clear();
  d_dst.clear();
  #endif // end of ifndef _ALLOC_DEVICE_MEM_ONCE_
  d_bias.clear();
  d_work_space.clear();
}

void cudnn_convolutional_layer::device_deallocate_for_backward(void)
{
  const int num_gpus = m_cudnn->m_num_gpus;

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
    checkCUDA(gpu_memory.DeviceFree(gpu, d_src[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_filter[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_prev_error_signal[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_filter_gradient[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_error_signal[i]));
    if(d_filter_work_space[i]) {
      checkCUDA(gpu_memory.DeviceFree(gpu, d_filter_work_space[i]));
    }
    if(d_data_work_space[i]) {
      checkCUDA(gpu_memory.DeviceFree(gpu, d_data_work_space[i]));
    }
  }
  d_src.clear();
  d_filter.clear();
  d_prev_error_signal.clear();
  d_filter_gradient.clear();
  d_error_signal.clear();
}

void cudnn_convolutional_layer::forward(const Mat& src,
                                        const Mat& filter,
                                        const Mat& bias,
                                        Mat& dst)
{

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Get number of GPUs
  const int num_gpus = m_cudnn->m_num_gpus;

#ifndef _ALLOC_DEVICE_MEM_ONCE_
  device_allocate_for_forward();
#endif

  // Perform convolution with each GPU
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cudnnHandle_t& handle = m_cudnn->m_handles[i];

    // Data samples assigned to GPU
    const int first_pos = Min(i * m_samples_per_gpu, src.Width());
    const int last_pos = Min((i+1) * m_samples_per_gpu, src.Width());
    if(first_pos >= last_pos) {
      continue;
    }

    // Transfer filters to GPU
    checkCUDA(cudaMemcpyAsync(d_filter[i],
                              filter.LockedBuffer(),
                              m_filter_size*sizeof(DataType),
                              cudaMemcpyHostToDevice,
                              stream));

    // Transfer inputs to GPU
    checkCUDA(cudaMemcpy2DAsync(d_src[i],
                                m_src_size*sizeof(DataType),
                                src.LockedBuffer(0,first_pos),
                                src.LDim()*sizeof(DataType),
                                m_src_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));

    // Transfer biases to GPU
    checkCUDA(cudaMemcpyAsync(d_bias[i],
                              bias.LockedBuffer(),
                              m_dst_size*sizeof(DataType),
                              cudaMemcpyHostToDevice,
                              stream));
    for(int pos=first_pos; pos<last_pos; ++pos) {
      checkCUDA(cudaMemcpyAsync(d_dst[i] + (pos-first_pos)*m_dst_size,
                                d_bias[i],
                                m_dst_size*sizeof(DataType),
                                cudaMemcpyDeviceToDevice,
                                stream));
    }

    // Perform convolution
    checkCUDNN(cudnnConvolutionForward(handle,
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
    checkCUDA(cudaMemcpy2DAsync(dst.Buffer(0,first_pos),
                                dst.LDim()*sizeof(DataType),
                                d_dst[i],
                                m_dst_size*sizeof(DataType),
                                m_dst_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyDeviceToHost,
                                stream));

  }

#ifndef _ALLOC_DEVICE_MEM_ONCE_
  device_deallocate_for_forward();
#endif

  // Synchronize CUDA streams
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaStreamSynchronize(m_cudnn->m_streams[i]));
  }

}

void cudnn_convolutional_layer::backward(const Mat& src,
                                         const Mat& filter,
                                         const Mat& prev_error_signal,
                                         Mat& filter_gradient,
                                         Mat& bias_gradient,
                                         Mat& error_signal)
{

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Get number of GPUs
  const int num_gpus = m_cudnn->m_num_gpus;

  // Get number of samples per GPU
  const DataType samples_per_gpu_float = m_samples_per_gpu;

  // Initialize filter gradient
  El::Zero(filter_gradient);

  // Compute bias gradient
  Mat ones;
  El::Ones(ones, prev_error_signal.Width(), El::Int(1));
  El::Gemv(El::NORMAL, DataType(1), prev_error_signal, ones,
           DataType(0), bias_gradient);

#ifndef _ALLOC_DEVICE_MEM_ONCE_
  device_allocate_for_backward();
#endif // _ALLOC_DEVICE_MEM_ONCE_

  // Perform back propagation on each GPU
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cudnnHandle_t& handle = m_cudnn->m_handles[i];

    // Data samples assigned to GPU
    const int first_pos = Min(i * m_samples_per_gpu, src.Width());
    const int last_pos = Min((i+1) * m_samples_per_gpu, src.Width());
    if(first_pos >= last_pos) {
      continue;
    }

    // Transfer filters to GPU
    checkCUDA(cudaMemcpyAsync(d_filter[i],
                              filter.LockedBuffer(),
                              m_filter_size*sizeof(DataType),
                              cudaMemcpyHostToDevice,
                              stream));

    // Transfer inputs and error signal to GPU
    checkCUDA(cudaMemcpy2DAsync(d_src[i],
                                m_src_size*sizeof(DataType),
                                src.LockedBuffer(0,first_pos),
                                src.LDim()*sizeof(DataType),
                                m_src_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));
    checkCUDA(cudaMemcpy2DAsync(d_prev_error_signal[i],
                                m_dst_size*sizeof(DataType),
                                prev_error_signal.LockedBuffer(0,first_pos),
                                prev_error_signal.LDim()*sizeof(DataType),
                                m_dst_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));
    if(last_pos - first_pos < m_samples_per_gpu) {
      checkCUDA(cudaMemsetAsync(d_prev_error_signal[i] + (last_pos-first_pos)*m_dst_size,
                                0,
                                m_dst_size*(m_samples_per_gpu-(last_pos-first_pos))*sizeof(DataType),
                                stream));
    }

    // Compute filter gradient
    checkCUDNN(cudnnConvolutionBackwardFilter(handle,
                                              &samples_per_gpu_float,
                                              m_src_desc,
                                              d_src[i],
                                              m_dst_desc,
                                              d_prev_error_signal[i],
                                              m_conv_desc,
                                              m_backward_filter_algo,
                                              d_filter_work_space[i],
                                              m_backward_filter_work_space_size,
                                              &zero,
                                              m_filter_desc,
                                              d_filter_gradient[i]));

    // Compute error signal to "next" layer
    checkCUDNN(cudnnConvolutionBackwardData(handle,
                                            &one,
                                            m_filter_desc,
                                            d_filter[i],
                                            m_dst_desc,
                                            d_prev_error_signal[i],
                                            m_conv_desc,
                                            m_backward_data_algo,
                                            d_data_work_space[i],
                                            m_backward_data_work_space_size,
                                            &zero,
                                            m_src_desc,
                                            d_error_signal[i]));

    // Transfer data from GPU
    checkCUDA(cudaMemcpy2DAsync(error_signal.Buffer(0,first_pos),
                                error_signal.LDim()*sizeof(DataType),
                                d_error_signal[i],
                                m_src_size*sizeof(DataType),
                                m_src_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyDeviceToHost,
                                stream));

  }
  
  // Transfer and accumulate filter gradients from GPUs
  El::Zero(filter_gradient);
  El::Zero(temp);

  for(int i=0; i<num_gpus; ++i) {
    if(i*m_samples_per_gpu >= src.Width()) {
      break;
    }
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    cudaStream_t& stream = m_cudnn->m_streams[i];
    checkCUDA(cudaMemcpyAsync(temp(ALL, IR(i,i+1)).Buffer(), // assuming 1 column otherwise filter_gradient.Width()*i with 2DAsync
                         d_filter_gradient[i],
                         m_filter_size*sizeof(DataType),
                         cudaMemcpyDeviceToHost,
                         stream));
  }

  // Synchronize CUDA streams
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaStreamSynchronize(m_cudnn->m_streams[i]));
    filter_gradient += temp(ALL, IR(i,i+1));
  }

#ifndef _ALLOC_DEVICE_MEM_ONCE_
  device_deallocate_for_backward();
#endif

}

cudnn_pooling_layer::cudnn_pooling_layer(const int num_dims,
                                         const int channels,
                                         const int* src_dims,
                                         const pool_mode _pool_mode,
                                         const int* pool_dims,
                                         const int* pool_pads,
                                         const int* pool_strides,
                                         const uint mini_batch_size,
                                         cudnn_manager* cudnn)
  : m_num_dims(num_dims), m_cudnn(cudnn),
    m_cudnn_data_type(get_cudnn_data_type<DataType>()),
    m_pool_mode(get_cudnn_pool_mode(_pool_mode)),
    m_src_desc(NULL), m_dst_desc(NULL), m_pool_desc(NULL),
    m_mini_batch_size(mini_batch_size)
{

  // Get number of GPUs
  const int num_gpus = m_cudnn->m_num_gpus;

  // Get number of samples per GPU
  const int num_processes = m_cudnn->comm->get_procs_per_model();
  const int local_mini_batch_size = (mini_batch_size + num_processes - 1) / num_processes;
  m_samples_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

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
#ifdef _ALLOC_DEVICE_MEM_ONCE_
  device_deallocate();
#endif
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

  m_src_dims[0] = m_samples_per_gpu;
  m_dst_dims[0] = m_samples_per_gpu;
#ifdef _ALLOC_DEVICE_MEM_ONCE_
  device_allocate();
#endif
}

void cudnn_pooling_layer::device_allocate(void)
{
  device_allocate_for_forward();
  device_allocate_for_backward();
}

void cudnn_pooling_layer::device_allocate_for_forward(void)
{
  const int num_gpus = m_cudnn->m_num_gpus;
  d_src.clear();
  d_src.assign(num_gpus, NULL);
  d_dst.clear();
  d_dst.assign(num_gpus, NULL);

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_src[i],
                                        m_src_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_dst[i],
                                        m_dst_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
  }
}

void cudnn_pooling_layer::device_allocate_for_backward(void)
{
  const int num_gpus = m_cudnn->m_num_gpus;
#ifndef _ALLOC_DEVICE_MEM_ONCE_ // Otherwise reuse the blocks allocated for forward path
  d_src.clear();
  d_src.assign(num_gpus, NULL);
  d_dst.clear();
  d_dst.assign(num_gpus, NULL);
#endif // ifndef _ALLOC_DEVICE_MEM_ONCE_
  d_prev_error_signal.clear();
  d_prev_error_signal.assign(num_gpus, NULL);
  d_error_signal.clear();
  d_error_signal.assign(num_gpus, NULL);

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
#ifndef _ALLOC_DEVICE_MEM_ONCE_
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_src[i],
                                        m_src_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_dst[i],
                                        m_dst_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
#endif // ifndef _ALLOC_DEVICE_MEM_ONCE_
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_prev_error_signal[i],
                                        m_dst_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_error_signal[i],
                                        m_src_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
  }
}

void cudnn_pooling_layer::device_deallocate(void)
{
  device_deallocate_for_forward();
  device_deallocate_for_backward();
}

void cudnn_pooling_layer::device_deallocate_for_forward(void)
{
#ifndef _ALLOC_DEVICE_MEM_ONCE_ // Otherwise, let backward deallocates blocks
  const int num_gpus = m_cudnn->m_num_gpus;

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
    checkCUDA(gpu_memory.DeviceFree(gpu, d_src[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_dst[i]));
  }
  d_src.clear();
  d_dst.clear();
#endif // ifndef _ALLOC_DEVICE_MEM_ONCE_
}

void cudnn_pooling_layer::device_deallocate_for_backward(void)
{
  const int num_gpus = m_cudnn->m_num_gpus;

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
    checkCUDA(gpu_memory.DeviceFree(gpu, d_src[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_dst[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_prev_error_signal[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_error_signal[i]));
  }

  d_src.clear();
  d_dst.clear();
  d_prev_error_signal.clear();
  d_error_signal.clear();
}

void cudnn_pooling_layer::forward(const Mat& src, Mat& dst)
{

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Return immediately if mini-batch size is zero
  if(src.Width() <= 0) {
    return;
  }

  const int num_gpus = m_cudnn->m_num_gpus;

//  m_src_dims[0] = m_samples_per_gpu;
//  m_dst_dims[0] = m_samples_per_gpu;
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


#ifndef _ALLOC_DEVICE_MEM_ONCE_
  device_allocate_for_forward();
#endif

  // Iterate through GPUs
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cudnnHandle_t& handle = m_cudnn->m_handles[i];
 
    // Data samples assigned to GPU
    const int first_pos = Min(i * m_samples_per_gpu, src.Width());
    const int last_pos = Min((i+1) * m_samples_per_gpu, src.Width());
    if(first_pos >= last_pos) {
      continue;
    }

    // Transfer inputs to GPU
    checkCUDA(cudaMemcpy2DAsync(d_src[i],
                                m_src_size*sizeof(DataType),
                                src.LockedBuffer(0,first_pos),
                                src.LDim()*sizeof(DataType),
                                m_src_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));

    // Perform pooling
    checkCUDNN(cudnnPoolingForward(handle,
                                   m_pool_desc,
                                   &one,
                                   m_src_desc,
                                   d_src[i],
                                   &zero,
                                   m_dst_desc,
                                   d_dst[i]));

    // Transfer outputs from GPU
    checkCUDA(cudaMemcpy2DAsync(dst.Buffer(0,first_pos),
                                dst.LDim()*sizeof(DataType),
                                d_dst[i],
                                m_dst_size*sizeof(DataType),
                                m_dst_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyDeviceToHost,
                                stream));

  }

#ifndef _ALLOC_DEVICE_MEM_ONCE_
  device_deallocate_for_forward();
#endif

  // Synchronize CUDA streams
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaStreamSynchronize(m_cudnn->m_streams[i]));
  }

}

void cudnn_pooling_layer::backward(const Mat& src,
                                   const Mat& dst,
                                   const Mat& prev_error_signal,
                                   Mat& error_signal)
{

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Return immediately if mini-batch size is zero
  if(src.Width() <= 0) {
    return;
  }

  const int num_gpus = m_cudnn->m_num_gpus;

/*
  m_src_dims[0] = m_samples_per_gpu;
  m_dst_dims[0] = m_samples_per_gpu;
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
*/


#ifndef _ALLOC_DEVICE_MEM_ONCE_
  device_allocate_for_backward();
#endif

  // Iterate through GPUs
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cudnnHandle_t& handle = m_cudnn->m_handles[i];

    // Data samples assigned to GPU
    const int first_pos = Min(i * m_samples_per_gpu, src.Width());
    const int last_pos = Min((i+1) * m_samples_per_gpu, src.Width());
    if(first_pos >= last_pos) {
      continue;
    }

    // Transfer data to GPU
    checkCUDA(cudaMemcpy2DAsync(d_src[i],
                                m_src_size*sizeof(DataType),
                                src.LockedBuffer(0,first_pos),
                                src.LDim()*sizeof(DataType),
                                m_src_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));
    checkCUDA(cudaMemcpy2DAsync(d_dst[i],
                                m_dst_size*sizeof(DataType),
                                dst.LockedBuffer(0,first_pos),
                                dst.LDim()*sizeof(DataType),
                                m_dst_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));
    checkCUDA(cudaMemcpy2DAsync(d_prev_error_signal[i],
                                m_dst_size*sizeof(DataType),
                                prev_error_signal.LockedBuffer(0,first_pos),
                                prev_error_signal.LDim()*sizeof(DataType),
                                m_dst_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));
    if(last_pos - first_pos < m_samples_per_gpu) {
      checkCUDA(cudaMemsetAsync(d_prev_error_signal[i] + (last_pos-first_pos)*m_dst_size,
                                0,
                                m_dst_size*(m_samples_per_gpu-(last_pos-first_pos))*sizeof(DataType),
                                stream));
    }

    // Compute error signal
    checkCUDNN(cudnnPoolingBackward(handle,
                                    m_pool_desc,
                                    &one,
                                    m_dst_desc,
                                    d_dst[i],
                                    m_dst_desc,
                                    d_prev_error_signal[i],
                                    m_src_desc,
                                    d_src[i],
                                    &zero,
                                    m_src_desc,
                                    d_error_signal[i]));

    // Transfer data from GPU
    checkCUDA(cudaMemcpy2DAsync(error_signal.Buffer(0,first_pos),
                                error_signal.LDim()*sizeof(DataType),
                                d_error_signal[i],
                                m_src_size*sizeof(DataType),
                                m_src_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyDeviceToHost,
                                stream));

  }

#ifndef _ALLOC_DEVICE_MEM_ONCE_
  device_deallocate_for_backward();
#endif

  // Synchronize CUDA streams
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaStreamSynchronize(m_cudnn->m_streams[i]));
  }

}

#endif // __LIB_CUDNN
