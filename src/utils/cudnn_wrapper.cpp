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

#endif // #ifdef __LIB_CUDNN
