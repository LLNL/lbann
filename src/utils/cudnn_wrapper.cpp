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

cudnn_manager::cudnn_manager(lbann::lbann_comm *_comm, int max_num_gpus)
  : comm(_comm) {

  // Determine number of available GPUs
  int num_total_gpus;
  CHECK_CUDA(cudaGetDeviceCount(&num_total_gpus));
  m_num_total_gpus = num_total_gpus;
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
  if(m_num_total_gpus >= procs_per_node) {
    int gpu = rank_in_node;
    while(gpu < m_num_total_gpus) {
      CHECK_CUDA(cudaSetDevice(gpu));
      m_gpus.push_back(gpu);
      m_streams.push_back(NULL);
      m_handles.push_back(NULL);
      cudaStream_t& stream = m_streams.back();
      cudnnHandle_t& handle = m_handles.back();
      CHECK_CUDA(cudaStreamCreate(&stream));
      CHECK_CUDNN(cudnnCreate(&handle));
      CHECK_CUDNN(cudnnSetStream(handle, stream));
      gpu += procs_per_node;
    }
  }

  // Case where compute node has fewers GPUs than MPI ranks
  else {
    const int gpu = rank_in_node % m_num_total_gpus;
    CHECK_CUDA(cudaSetDevice(gpu));
    m_gpus.push_back(gpu);
    m_streams.push_back(NULL);
    m_handles.push_back(NULL);
    cudaStream_t& stream = m_streams.back();
    cudnnHandle_t& handle = m_handles.back();
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDNN(cudnnCreate(&handle));
    CHECK_CUDNN(cudnnSetStream(handle, stream));
  }

  // Get number of GPUs for current MPI rank
  m_num_gpus = m_gpus.size();

  // Initialize work spaces
  m_work_spaces = std::vector<void *>(m_num_gpus, NULL);
  m_work_space_sizes = std::vector<size_t>(m_num_gpus, 0);

}

cudnn_manager::~cudnn_manager() {
  // Free work spaces
  for(size_t i=0u; i<m_gpus.size(); ++i) {
    if(m_work_space_sizes[i]) {
      CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      CHECK_CUDA(cudaFree(m_work_spaces[i]));
    }
  }

  // Destroy cuDNN handles
  for(size_t i=0u; i<m_gpus.size(); ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    if(m_streams[i]) {
      CHECK_CUDA(cudaStreamDestroy(m_streams[i]));
    }
    if(m_handles[i]) {
      CHECK_CUDNN(cudnnDestroy(m_handles[i]));
    }
  }
}

void cudnn_manager::cudnn_manager::allocate_on_gpus(std::vector<DataType *>& gpu_data,
                                                    int height,
                                                    int width_per_gpu) {

#ifdef LBANN_DEBUG
  if(gpu_data.size() != 0) {
    // Check that list of pointers has valid number of entries
    if(gpu_data.size() != m_num_gpus) {
      throw lbann_exception("cudnn_wrapper: number of GPU memory pointers doesn't match number of GPUs");
    }
    // Check that list of pointers only contains null pointers
    for(int i=0; i<m_num_gpus; ++i) {
      if(gpu_data[i] != NULL) {
        throw lbann_exception("cudnn_wrapper: overwriting non-null pointer with newly allocated GPU memory");
      }
    }
  }
#endif // #ifdef LBANN_DEBUG

  // Allocate GPU memory
  gpu_data.resize(m_num_gpus, NULL);
  for(int i=0; i<m_num_gpus; ++i) {
    if(height*width_per_gpu > 0) {
      CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      CHECK_CUDA(cudaMalloc((void **) &gpu_data[i],
                            height*width_per_gpu*sizeof(DataType)));
    }

  }

}

void cudnn_manager::cudnn_manager::deallocate_on_gpus(std::vector<DataType *>& gpu_data) {

  // Stop if list of pointers is empty
  if(gpu_data.size() == 0) {
    return;
  }

#ifdef LBANN_DEBUG
  // Ensure that gpu_data has right dimensions
  if(gpu_data.size() != m_num_gpus) {
    throw lbann_exception("cudnn_wrapper: number of GPU memory pointers doesn't match number of GPUs");
  }
#endif // #ifdef LBANN_DEBUG

  // Deallocate GPU memory
  for(int i=0; i<m_num_gpus; ++i) {
    if(gpu_data[i] != NULL) {
      CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      CHECK_CUDA(cudaFree(gpu_data[i]));
    }
  }

  // Clear list of GPU memory pointers
  gpu_data.clear();

}

void cudnn_manager::cudnn_manager::clear_on_gpus(std::vector<DataType *>& gpu_data,
                                                 int height,
                                                 int width_per_gpu) {

  // Clear memory on each GPU
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    CHECK_CUDA(cudaMemsetAsync(gpu_data[i],
                               0,
                               height*width_per_gpu*sizeof(DataType),
                               m_streams[i]));
  }

}

void cudnn_manager::cudnn_manager::clear_unused_columns_on_gpus(std::vector<DataType *>& gpu_data,
                                                                int height,
                                                                int width,
                                                                int width_per_gpu) {

  // Iterate through GPUs
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));

    // Find number of columns on each GPU
    const int first_pos = std::min(i * width_per_gpu, width);
    const int last_pos = std::min((i+1) * width_per_gpu, width);
    const int current_width = last_pos - first_pos;

    // Set unused GPU memory to zero
    if(current_width < width_per_gpu) {
      CHECK_CUDA(cudaMemsetAsync(gpu_data[i] + height*current_width,
                                 0,
                                 height*(width_per_gpu-current_width)*sizeof(DataType),
                                 m_streams[i]));
    }

  }

}

void cudnn_manager::cudnn_manager::copy_on_gpus(std::vector<DataType *>& gpu_dst_data,
                                                const std::vector<DataType *>& gpu_src_data,
                                                int height,
                                                int width_per_gpu) {

  // Perform memory transfer on each GPU
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    CHECK_CUDA(cudaMemcpyAsync(gpu_dst_data[i],
                               gpu_src_data[i],
                               height*width_per_gpu*sizeof(DataType),
                               cudaMemcpyDeviceToDevice,
                               m_streams[i]));
  }

}

void cudnn_manager::cudnn_manager::scatter_to_gpus(std::vector<DataType *>& gpu_data,
                                                   const Mat& cpu_data,
                                                   int width_per_gpu) {

  // Get matrix properties
  const int height = cpu_data.Height();
  const int width = cpu_data.Width();
  const int cpu_ldim = cpu_data.LDim();

  // Perform memory transfer on each GPU
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));

    // Find number of columns to transfer to current GPU
    const int first_pos = std::min(i * width_per_gpu, width);
    const int last_pos = std::min((i+1) * width_per_gpu, width);
    const int current_width = last_pos - first_pos;

    // Transfer data to current GPU
    if(current_width > 0) {
      if(cpu_ldim > height) {
        CHECK_CUDA(cudaMemcpy2DAsync(gpu_data[i],
                                     height*sizeof(DataType),
                                     cpu_data.LockedBuffer(0,first_pos),
                                     cpu_ldim*sizeof(DataType),
                                     height*sizeof(DataType),
                                     current_width,
                                     cudaMemcpyHostToDevice,
                                     m_streams[i]));
      } else {
        CHECK_CUDA(cudaMemcpyAsync(gpu_data[i],
                                   cpu_data.LockedBuffer(0,first_pos),
                                   height*current_width*sizeof(DataType),
                                   cudaMemcpyHostToDevice,
                                   m_streams[i]));
      }
    }

    // Set unused GPU memory to zero
    if(current_width < width_per_gpu) {
      CHECK_CUDA(cudaMemsetAsync(gpu_data[i] + height*current_width,
                                 0,
                                 height*(width_per_gpu-current_width)*sizeof(DataType),
                                 m_streams[i]));
    }

  }

}

void cudnn_manager::cudnn_manager::gather_from_gpus(Mat& cpu_data,
                                                    const std::vector<DataType *>& gpu_data,
                                                    int width_per_gpu) {

  // Get matrix properties
  const int height = cpu_data.Height();
  const int width = cpu_data.Width();
  const int cpu_ldim = cpu_data.LDim();

  // Perform memory transfer on each GPU
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));

    // Find number of columns to transfer to current GPU
    const int first_pos = std::min(i * width_per_gpu, width);
    const int last_pos = std::min((i+1) * width_per_gpu, width);
    const int current_width = last_pos - first_pos;

    // Transfer data from current GPU
    if(current_width > 0) {
      if(cpu_ldim > height) {
        CHECK_CUDA(cudaMemcpy2DAsync(cpu_data.Buffer(0,first_pos),
                                     cpu_ldim*sizeof(DataType),
                                     gpu_data[i],
                                     height*sizeof(DataType),
                                     height*sizeof(DataType),
                                     current_width,
                                     cudaMemcpyDeviceToHost,
                                     m_streams[i]));
      } else {
        CHECK_CUDA(cudaMemcpyAsync(cpu_data.Buffer(0,first_pos),
                                   gpu_data[i],
                                   height*current_width*sizeof(DataType),
                                   cudaMemcpyDeviceToHost,
                                   m_streams[i]));
      }
    }

  }

}

void cudnn_manager::cudnn_manager::broadcast_to_gpus(std::vector<DataType *>& gpu_data,
                                                     const Mat& cpu_data) {

  // Get matrix properties
  const int height = cpu_data.Height();
  const int width = cpu_data.Width();
  const int cpu_ldim = cpu_data.LDim();

  // Perform memory transfer on each GPU
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));

    // Transfer data to current GPU
    if(cpu_ldim > height) {
      CHECK_CUDA(cudaMemcpy2DAsync(gpu_data[i],
                                   height*sizeof(DataType),
                                   cpu_data.LockedBuffer(),
                                   cpu_ldim*sizeof(DataType),
                                   height*sizeof(DataType),
                                   width,
                                   cudaMemcpyHostToDevice,
                                   m_streams[i]));
    } else {
      CHECK_CUDA(cudaMemcpyAsync(gpu_data[i],
                                 cpu_data.LockedBuffer(),
                                 height*width*sizeof(DataType),
                                 cudaMemcpyHostToDevice,
                                 m_streams[i]));
    }

  }

}

void cudnn_manager::cudnn_manager::reduce_from_gpus(Mat& cpu_data,
                                                    const std::vector<DataType *>& gpu_data) {

  // Get matrix properties
  const int height = cpu_data.Height();
  const int width = cpu_data.Width();

  // Copy data from GPUs to CPU
  Mat temp;
  Zeros(temp, height, m_num_gpus*width);
  gather_from_gpus(temp, gpu_data, width);

  // Reduce data from different GPUs
  Zero(cpu_data);
  synchronize();
  for(int i=0; i<m_num_gpus; ++i) {
    cpu_data += temp(ALL, IR(i*width, (i+1)*width));
  }

}

void cudnn_manager::cudnn_manager::synchronize() {
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    CHECK_CUDA(cudaStreamSynchronize(m_streams[i]));
  }
}

void cudnn_manager::print_version() const {
  std::cout << "cudnnGetVersion() : " << (int)cudnnGetVersion() << " , "
            << "CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION
            << std::endl;
}

cudnnDataType_t cudnn_manager::get_cudnn_data_type() const {
  switch(sizeof(DataType)) {
  case 2:
    return CUDNN_DATA_HALF;
  case 4:
    return CUDNN_DATA_FLOAT;
  case 8:
    return CUDNN_DATA_DOUBLE;
  default:
    throw lbann::lbann_exception("cudnn_wrapper: invalid data type for cuDNN");
  }
}

int cudnn_manager::get_num_gpus() const {
  return m_num_gpus;
}

int cudnn_manager::get_num_total_gpus() const {
  return m_num_total_gpus;
}

std::vector<int>& cudnn_manager::get_gpus() {
  return m_gpus;
}

const std::vector<int>& cudnn_manager::get_gpus() const {
  return m_gpus;
}

int cudnn_manager::get_gpu(int i) const {
  return m_gpus[i];
}

std::vector<cudaStream_t>& cudnn_manager::get_streams() {
  return m_streams;
}

const std::vector<cudaStream_t>& cudnn_manager::get_streams() const {
  return m_streams;
}

cudaStream_t& cudnn_manager::get_stream(int i) {
  return m_streams[i];
}

const cudaStream_t& cudnn_manager::get_stream(int i) const {
  return m_streams[i];
}

std::vector<cudnnHandle_t>& cudnn_manager::get_handles() {
  return m_handles;
}

const std::vector<cudnnHandle_t>& cudnn_manager::get_handles() const {
  return m_handles;
}

cudnnHandle_t& cudnn_manager::get_handle(int i) {
  return m_handles[i];
}

const cudnnHandle_t& cudnn_manager::get_handle(int i) const {
  return m_handles[i];
}

std::vector<void *> cudnn_manager::get_work_spaces() {
  for(int i=0; i<m_num_gpus; ++i) {
    if(m_work_spaces[i] == NULL && m_work_space_sizes[i] > 0) {
      CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      CHECK_CUDA(cudaMalloc((void **) &m_work_spaces[i],
                            m_work_space_sizes[i]));
    }
  }
  return m_work_spaces;
}

void *cudnn_manager::get_work_space(int i) {
  if(m_work_spaces[i] == NULL && m_work_space_sizes[i] > 0) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    CHECK_CUDA(cudaMalloc((void **) &m_work_spaces[i],
                          m_work_space_sizes[i]));
  }
  return m_work_spaces[i];
}

std::vector<size_t> cudnn_manager::get_work_space_sizes() {
  return m_work_space_sizes;
};

size_t cudnn_manager::get_work_space_size(int i) const {
  return m_work_space_sizes[i];
}

void cudnn_manager::set_work_space_size(int i, size_t size) {
  if(m_work_space_sizes[i] != size) {
    m_work_space_sizes[i] = size;
    if(m_work_spaces[i] != NULL) {
      CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      CHECK_CUDA(cudaFree(m_work_spaces[i]));
      m_work_spaces[i] = NULL;
    }
  }
}

void cudnn_manager::pin_matrix(ElMat& mat) {
  
  // Get local matrix
  const Mat& mat_local = mat.LockedMatrix();
  
  // Allocate pinned memory on host
  DataType* pinned_buffer;
  CHECK_CUDA(cudaMallocHost((void**) &pinned_buffer,
                            mat_local.Height()*mat_local.Width()*sizeof(DataType)));

  // Reconfigure matrix around pinned memory
  mat.Attach(mat.Height(),
             mat.Width(),
             mat.Grid(),
             mat.ColAlign(),
             mat.RowAlign(),
             pinned_buffer,
             mat_local.Height(),
             mat.Root());

  // Copy data
  Copy(mat_local, mat.Matrix());

}

void cudnn_manager::unpin_matrix(ElMat& mat) {
  
  // Get local matrix
  Mat& mat_local = mat.Matrix();
  Mat new_mat_local(mat_local);
  
  // Reconfigure matrix around unpinned memory
  mat.Attach(mat.Height(),
             mat.Width(),
             mat.Grid(),
             mat.ColAlign(),
             mat.RowAlign(),
             new_mat_local,
             mat.Root());

  // Deallocate pinned memory
  CHECK_CUDA(cudaFreeHost(mat_local.Buffer()));

}

#endif // #ifdef __LIB_CUDNN
