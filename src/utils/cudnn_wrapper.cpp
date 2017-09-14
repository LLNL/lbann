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
#include "lbann/utils/exception.hpp"

#include <iostream>

#include "El.hpp"

#ifdef __LIB_CUDNN

using namespace cudnn;
using namespace lbann;

cudnn_manager::cudnn_manager(lbann::lbann_comm *_comm, int max_num_gpus)
  : comm(_comm) {

  // Determine number of available GPUs
  CHECK_CUDA(cudaGetDeviceCount(&m_num_total_gpus));
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
    const int min_gpus_per_proc = m_num_total_gpus / procs_per_node;
    const int num_gpus_remainder = m_num_total_gpus % procs_per_node;
    int gpu_start = rank_in_node * min_gpus_per_proc;
    int gpu_end = (rank_in_node + 1) * min_gpus_per_proc;
    if(rank_in_node < num_gpus_remainder) {
      gpu_start += rank_in_node;
      gpu_end += rank_in_node + 1;
    }
    else {
      gpu_start += num_gpus_remainder;
      gpu_end += num_gpus_remainder;
    }
    for(int gpu = gpu_start; gpu < gpu_end; ++gpu) {
      FORCE_CHECK_CUDA(cudaSetDevice(gpu));
      m_gpus.push_back(gpu);
      m_streams.push_back(nullptr);
      m_handles.push_back(nullptr);
      m_cublas_handles.push_back(nullptr);
      FORCE_CHECK_CUDA(cudaStreamCreate(&m_streams.back()));
      FORCE_CHECK_CUDNN(cudnnCreate(&m_handles.back()));
      FORCE_CHECK_CUDNN(cudnnSetStream(m_handles.back(), m_streams.back()));
      FORCE_CHECK_CUBLAS(cublasCreate(&m_cublas_handles.back()));
    }
  }

  // Case where compute node has fewers GPUs than MPI ranks
  else {
    const int min_procs_per_gpu = procs_per_node / m_num_total_gpus;
    const int procs_remainder = procs_per_node % m_num_total_gpus;
    int gpu = -1;
    int proc_end = 0;
    do {
      gpu++;
      if(gpu < procs_remainder) {
        proc_end += min_procs_per_gpu + 1;
      }
      else {
        proc_end += min_procs_per_gpu;
      }
    } while(rank_in_node >= proc_end);
    FORCE_CHECK_CUDA(cudaSetDevice(gpu));
    m_gpus.push_back(gpu);
    m_streams.push_back(nullptr);
    m_handles.push_back(nullptr);
    m_cublas_handles.push_back(nullptr);    
    FORCE_CHECK_CUDA(cudaStreamCreate(&m_streams.back()));
    FORCE_CHECK_CUDNN(cudnnCreate(&m_handles.back()));
    FORCE_CHECK_CUDNN(cudnnSetStream(m_handles.back(), m_streams.back()));
    FORCE_CHECK_CUBLAS(cublasCreate(&m_cublas_handles.back()));
  }

  // Get number of GPUs for current MPI rank
  m_num_gpus = m_gpus.size();

  // Initialize work spaces
  m_work_spaces = std::vector<void *>(m_num_gpus, nullptr);
  m_work_space_sizes = std::vector<size_t>(m_num_gpus, 0);

}

cudnn_manager::~cudnn_manager() {
  // Free work spaces
  for(size_t i=0u; i<m_gpus.size(); ++i) {
    if(m_work_space_sizes[i]) {
      FORCE_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      FORCE_CHECK_CUDA(cudaFree(m_work_spaces[i]));
    }
  }

  // Destroy cuDNN handles
  for(size_t i=0u; i<m_gpus.size(); ++i) {
    FORCE_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    if(m_streams[i]) {
      FORCE_CHECK_CUDA(cudaStreamDestroy(m_streams[i]));
    }
    if(m_handles[i]) {
      FORCE_CHECK_CUDNN(cudnnDestroy(m_handles[i]));
    }
    if(m_cublas_handles[i]) {
      FORCE_CHECK_CUBLAS(cublasDestroy(m_cublas_handles[i]));
    }
  }
}

void cudnn_manager::cudnn_manager::allocate_on_gpus(std::vector<DataType *>& gpu_data,
                                                    int height,
                                                    int width_per_gpu) {

  if(!gpu_data.empty()) {
    // Check that list of pointers has valid number of entries
    if((int) gpu_data.size() != m_num_gpus) {
      throw lbann_exception("cudnn_wrapper: number of GPU memory pointers doesn't match number of GPUs");
    }
    // Check that list of pointers only contains null pointers
    for(int i=0; i<m_num_gpus; ++i) {
      if(gpu_data[i] != nullptr) {
        throw lbann_exception("cudnn_wrapper: overwriting non-null pointer with newly allocated GPU memory");
      }
    }
  }

  // Allocate GPU memory
  gpu_data.resize(m_num_gpus, nullptr);
  for(int i=0; i<m_num_gpus; ++i) {
    if(height*width_per_gpu > 0) {
      FORCE_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      FORCE_CHECK_CUDA(cudaMalloc((void **) &gpu_data[i],
                                  height*width_per_gpu*sizeof(DataType)));
    }

  }

}

void cudnn_manager::cudnn_manager::deallocate_on_gpus(std::vector<DataType *>& gpu_data) {

  // Stop if list of pointers is empty
  if(gpu_data.empty()) {
    return;
  }

#ifdef LBANN_DEBUG
  // Ensure that gpu_data has right dimensions
  if((int) gpu_data.size() != m_num_gpus) {
    throw lbann_exception("cudnn_wrapper: number of GPU memory pointers doesn't match number of GPUs");
  }
#endif // #ifdef LBANN_DEBUG

  // Deallocate GPU memory
  for(int i=0; i<m_num_gpus; ++i) {
    if(gpu_data[i] != nullptr) {
      FORCE_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      FORCE_CHECK_CUDA(cudaFree(gpu_data[i]));
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
                                                int width_per_gpu,
                                                int src_leading_dim,
                                                int dst_leading_dim) {

  // Default leading dimension
  src_leading_dim = std::max(src_leading_dim, height);
  dst_leading_dim = std::max(dst_leading_dim, height);

  // Perform memory transfer on each GPU
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    CHECK_CUDA(cudaMemcpy2DAsync(gpu_dst_data[i],
                                 dst_leading_dim*sizeof(DataType),
                                 gpu_src_data[i],
                                 src_leading_dim*sizeof(DataType),
                                 height*sizeof(DataType),
                                 width_per_gpu,
                                 cudaMemcpyDeviceToDevice,
                                 m_streams[i]));
  }

}

void cudnn_manager::cudnn_manager::scatter_to_gpus(std::vector<DataType *>& gpu_data,
                                                   const Mat& cpu_data,
                                                   int width_per_gpu,
                                                   int gpu_data_leading_dim) {

  // Get matrix properties
  const int height = cpu_data.Height();
  const int width = cpu_data.Width();
  const int cpu_data_leading_dim = cpu_data.LDim();
  gpu_data_leading_dim = std::max(gpu_data_leading_dim, height);

  // Perform memory transfer on each GPU
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));

    // Find number of columns to transfer to current GPU
    const int first_pos = std::min(i * width_per_gpu, width);
    const int last_pos = std::min((i+1) * width_per_gpu, width);
    const int current_width = last_pos - first_pos;

    // Transfer data to current GPU
    if(current_width > 0) {
      CHECK_CUDA(cudaMemcpy2DAsync(gpu_data[i],
                                   gpu_data_leading_dim*sizeof(DataType),
                                   cpu_data.LockedBuffer(0,first_pos),
                                   cpu_data_leading_dim*sizeof(DataType),
                                   height*sizeof(DataType),
                                   current_width,
                                   cudaMemcpyHostToDevice,
                                   m_streams[i]));
    }

    // Set unused GPU memory to zero
    if(current_width < width_per_gpu) {
      CHECK_CUDA(cudaMemset2DAsync(gpu_data[i] + gpu_data_leading_dim*current_width,
                                   gpu_data_leading_dim*sizeof(DataType),
                                   0,
                                   height*sizeof(DataType),
                                   width_per_gpu-current_width,
                                   m_streams[i]));
    }

  }

}

void cudnn_manager::cudnn_manager::gather_from_gpus(Mat& cpu_data,
                                                    const std::vector<DataType *>& gpu_data,
                                                    int width_per_gpu,
                                                    int gpu_data_leading_dim) {

  // Get matrix properties
  const int height = cpu_data.Height();
  const int width = cpu_data.Width();
  const int cpu_data_leading_dim = cpu_data.LDim();
  gpu_data_leading_dim = std::max(gpu_data_leading_dim, height);

  // Perform memory transfer on each GPU
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));

    // Find number of columns to transfer to current GPU
    const int first_pos = std::min(i * width_per_gpu, width);
    const int last_pos = std::min((i+1) * width_per_gpu, width);
    const int current_width = last_pos - first_pos;

    // Transfer data from current GPU
    if(current_width > 0) {
      CHECK_CUDA(cudaMemcpy2DAsync(cpu_data.Buffer(0,first_pos),
                                   cpu_data_leading_dim*sizeof(DataType),
                                   gpu_data[i],
                                   gpu_data_leading_dim*sizeof(DataType),
                                   height*sizeof(DataType),
                                   current_width,
                                   cudaMemcpyDeviceToHost,
                                   m_streams[i]));
    }

  }

}

void cudnn_manager::cudnn_manager::broadcast_to_gpus(std::vector<DataType *>& gpu_data,
                                                     const Mat& cpu_data,
                                                     int gpu_data_leading_dim) {

  // Get matrix properties
  const int height = cpu_data.Height();
  const int width = cpu_data.Width();
  const int cpu_data_leading_dim = cpu_data.LDim();
  gpu_data_leading_dim = std::max(gpu_data_leading_dim, height);

  // Perform memory transfer on each GPU
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    CHECK_CUDA(cudaMemcpy2DAsync(gpu_data[i],
                                 gpu_data_leading_dim*sizeof(DataType),
                                 cpu_data.LockedBuffer(),
                                 cpu_data_leading_dim*sizeof(DataType),
                                 height*sizeof(DataType),
                                 width,
                                 cudaMemcpyHostToDevice,
                                 m_streams[i]));
  }

}

void cudnn_manager::cudnn_manager::reduce_from_gpus(Mat& cpu_data,
                                                    const std::vector<DataType *>& gpu_data,
                                                    int gpu_data_leading_dim) {

  // Get matrix properties
  const int height = cpu_data.Height();
  const int width = cpu_data.Width();

  // Copy data from GPUs to CPU
  Mat temp;
  El::Zeros(temp, height, m_num_gpus*width);
  gather_from_gpus(temp, gpu_data, width, gpu_data_leading_dim);

  // Reduce data from different GPUs
  El::Zero(cpu_data);
  synchronize();
  for(int i=0; i<m_num_gpus; ++i) {
    cpu_data += temp(El::ALL, El::IR(i*width, (i+1)*width));
  }

}

void cudnn_manager::cudnn_manager::synchronize() {
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    CHECK_CUDA(cudaStreamSynchronize(m_streams[i]));
  }
}

void cudnn_manager::cudnn_manager::synchronize_all() {
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    CHECK_CUDA(cudaDeviceSynchronize());
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

std::vector<cublasHandle_t>& cudnn_manager::get_cublas_handles() {
  return m_cublas_handles;
}

const std::vector<cublasHandle_t>& cudnn_manager::get_cublas_handles() const {
  return m_cublas_handles;
}

cublasHandle_t& cudnn_manager::get_cublas_handle(int i) {
  return m_cublas_handles[i];
}

const cublasHandle_t& cudnn_manager::get_cublas_handle(int i) const {
  return m_cublas_handles[i];
}

std::vector<void *> cudnn_manager::get_work_spaces() {
  for(int i=0; i<m_num_gpus; ++i) {
    if(m_work_spaces[i] == nullptr && m_work_space_sizes[i] > 0) {
      CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      FORCE_CHECK_CUDA(cudaMalloc((void **) &m_work_spaces[i],
                                  m_work_space_sizes[i]));
    }
  }
  return m_work_spaces;
}

void *cudnn_manager::get_work_space(int i) {
  if(m_work_spaces[i] == nullptr && m_work_space_sizes[i] > 0) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    FORCE_CHECK_CUDA(cudaMalloc((void **) &m_work_spaces[i],
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
  if(m_work_spaces.empty()) {
    m_work_spaces.assign(m_num_gpus, nullptr);
  }
  if(m_work_space_sizes.empty()) {
    m_work_space_sizes.assign(m_num_gpus, 0);
  }
  if(m_work_space_sizes[i] != size) {
    m_work_space_sizes[i] = size;
    if(m_work_spaces[i] != nullptr) {
      CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      CHECK_CUDA(cudaFree(m_work_spaces[i]));
      m_work_spaces[i] = nullptr;
    }
  }
}

void cudnn_manager::pin_matrix(AbsDistMat& mat) {  

  // Get local matrix
  Mat& mat_local = mat.Matrix();
  const El::Int local_height = mat.LocalHeight();
  const El::Int local_width = mat.LocalWidth();
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::DistData dist_data(mat);
  const DataType* buffer = mat.LockedBuffer();

  // Check that data buffer is unpinned memory
  cudaPointerAttributes buffer_attributes;
  cudaError_t status = cudaPointerGetAttributes(&buffer_attributes, buffer);
  if(status != cudaErrorInvalidValue) {
    FORCE_CHECK_CUDA(status);
    return;
  }

  // clear the error status
  cudaGetLastError();
  
  // Allocate pinned memory on host
  const size_t buffer_size = local_height * local_width * sizeof(DataType);
  DataType* pinned_buffer;
  FORCE_CHECK_CUDA(cudaMallocHost((void**) &pinned_buffer, buffer_size));
  Mat pinned_mat(local_height, local_width, pinned_buffer, local_height);

  // Copy data to pinned memory
  Copy(mat_local, pinned_mat);
  mat.Empty();

  // Reconfigure matrix around pinned memory
  ElMat* elemental_mat = dynamic_cast<ElMat*>(&mat);
  BlockMat* block_mat = dynamic_cast<BlockMat*>(&mat);
  if(elemental_mat != nullptr) {
    elemental_mat->Attach(height,
                          width,
                          mat.Grid(),
                          dist_data.colAlign,
                          dist_data.rowAlign,
                          pinned_mat,
                          dist_data.root);
  } else if(block_mat != nullptr) {
    block_mat->Attach(height,
                      width,
                      mat.Grid(),
                      dist_data.blockHeight,
                      dist_data.blockWidth,
                      dist_data.colAlign,
                      dist_data.rowAlign,
                      dist_data.colCut,
                      dist_data.rowCut,
                      pinned_mat,
                      dist_data.root);
  } else {
    throw lbann::lbann_exception("cudnn_manager: could not cast AbsDistMat to ElMat or BlockMat");
  }

}

void cudnn_manager::unpin_matrix(AbsDistMat& mat) {

  // Matrix parameters
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::DistData dist_data(mat);
  DataType *buffer = mat.Buffer();

  // Check that data buffer is pinned memory on host
  cudaPointerAttributes buffer_attributes;
  cudaError_t status = cudaPointerGetAttributes(&buffer_attributes, buffer);
  if(status == cudaErrorInvalidValue) {
    return;
  }
  if(status != cudaErrorInvalidDevice) {
    FORCE_CHECK_CUDA(status);
  }
  if(buffer_attributes.memoryType != cudaMemoryTypeHost) {
    throw lbann::lbann_exception("cudnn_wrapper: can only unpin host memory");
  }

  // Copy data to unpinned memory
  const Mat mat_local_copy(mat.LockedMatrix());

  // Allocate new memory owned by matrix
  mat.Empty();
  mat.Resize(height, width);
  mat.AlignWith(dist_data);

  // Copy data to new memory
  Mat& mat_local = mat.Matrix();
  El::Copy(mat_local_copy, mat_local);

  // Deallocate pinned memory
  FORCE_CHECK_CUDA(cudaFreeHost(buffer));

}

void cudnn_manager::check_error() {
  synchronize();
  for(int i=0; i<m_num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
      cudaDeviceReset();
      throw lbann::lbann_exception("CUDA error");
    }
  }
}

#endif // #ifdef __LIB_CUDNN
