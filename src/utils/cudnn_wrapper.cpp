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
using namespace lbann;

#define _ALLOC_DEVICE_MEM_ONCE_

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

void cudnn_manager::cudnn_manager::pin_ptr(void* ptr, size_t sz)
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

void cudnn_manager::pin_memory_block(ElMat *mat)
{
    if (!mat) return;
    const int w = (mat->Matrix()).Width();
    const int h = (mat->Matrix()).Height();
    const int sz = w*h*sizeof(DataType);
    void* ptr = (void*) (mat->Matrix()).Buffer();
    pin_ptr(ptr, w*h*sizeof(DataType));
}

void cudnn_manager::cudnn_manager::unpin_ptr(void* const ptr)
{
  std::map<void*, size_t>::iterator it = pinned_ptr.find(ptr);
  if (it != pinned_ptr.end()) {
    checkCUDA(cudaHostUnregister(it->first));
    pinned_ptr.erase(it);
  }
}

void cudnn_manager::cudnn_manager::unpin_ptrs(void)
{
  std::map<void*, size_t>::iterator it = pinned_ptr.begin();
  std::map<void*, size_t>::iterator itend = pinned_ptr.end();

  for(; it != itend; ++it) {
    checkCUDA(cudaHostUnregister(it->first));
  }
  pinned_ptr.clear();
}

void cudnn_manager::print_version() const {
  std::cout << "cudnnGetVersion() : " << (int)cudnnGetVersion() << " , "
            << "CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION
            << std::endl;
}

cudnnDataType_t cudnn_manager::get_cudnn_data_type() const {
  switch(sizeof(DataType)) {
  case 2: return CUDNN_DATA_HALF;
  case 4: return CUDNN_DATA_FLOAT;
  case 8: return CUDNN_DATA_DOUBLE;
  default: throw lbann::lbann_exception("cudnn_wrapper: invalid data type for cuDNN");
  }
}

int cudnn_manager::get_num_gpus() const {
  return m_num_gpus;
}

int cudnn_manager::get_num_total_gpus() const {
  return m_num_total_gpus;
}

cub::CachingDeviceAllocator* cudnn_manager::get_gpu_memory() {
  return m_gpu_memory;
}

std::vector<cudaStream_t>* cudnn_manager::get_streams() {
  return &m_streams;
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
    m_cudnn_data_type(cudnn->get_cudnn_data_type()),
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
  d_prev_activations.clear();
  d_prev_activations.assign(num_gpus, NULL);
  d_activations.clear();
  d_activations.assign(num_gpus, NULL);

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_prev_activations[i],
                                        m_src_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_activations[i],
                                        m_dst_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
  }
}

void cudnn_pooling_layer::device_allocate_for_backward(void)
{
  const int num_gpus = m_cudnn->m_num_gpus;
#ifndef _ALLOC_DEVICE_MEM_ONCE_ // Otherwise reuse the blocks allocated for forward path
  d_prev_activations.clear();
  d_prev_activations.assign(num_gpus, NULL);
  d_activations.clear();
  d_activations.assign(num_gpus, NULL);
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
                                        (void**) &d_prev_activations[i],
                                        m_src_size*m_samples_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &d_activations[i],
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
    checkCUDA(gpu_memory.DeviceFree(gpu, d_prev_activations[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_activations[i]));
  }
  d_prev_activations.clear();
  d_activations.clear();
#endif // ifndef _ALLOC_DEVICE_MEM_ONCE_
}

void cudnn_pooling_layer::device_deallocate_for_backward(void)
{
  const int num_gpus = m_cudnn->m_num_gpus;

#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    const int gpu = m_cudnn->m_gpus[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
    checkCUDA(gpu_memory.DeviceFree(gpu, d_prev_activations[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_activations[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_prev_error_signal[i]));
    checkCUDA(gpu_memory.DeviceFree(gpu, d_error_signal[i]));
  }

  d_prev_activations.clear();
  d_activations.clear();
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
    checkCUDA(cudaMemcpy2DAsync(d_prev_activations[i],
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
                                   d_prev_activations[i],
                                   &zero,
                                   m_dst_desc,
                                   d_activations[i]));

    // Transfer outputs from GPU
    checkCUDA(cudaMemcpy2DAsync(dst.Buffer(0,first_pos),
                                dst.LDim()*sizeof(DataType),
                                d_activations[i],
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
    checkCUDA(cudaMemcpy2DAsync(d_prev_activations[i],
                                m_src_size*sizeof(DataType),
                                src.LockedBuffer(0,first_pos),
                                src.LDim()*sizeof(DataType),
                                m_src_size*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));
    checkCUDA(cudaMemcpy2DAsync(d_activations[i],
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
                                    d_activations[i],
                                    m_dst_desc,
                                    d_prev_error_signal[i],
                                    m_src_desc,
                                    d_prev_activations[i],
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
