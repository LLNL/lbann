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
#include <unistd.h>

#ifdef __LIB_CUDNN

using namespace cudnn;
using namespace lbann;

cudnn_manager::cudnn_manager(lbann::lbann_comm *_comm, int max_num_gpus, bool nccl_used)
  : comm(_comm) {

  // Indicate whether NCCL is used 
  m_nccl_used = nccl_used;

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

    // NCCL setup
    if(m_nccl_used){
      nccl_setup();
    }
  }

  // Case where compute node has fewer GPUs than MPI ranks
  else {
    if(m_nccl_used){
      throw lbann_exception("cudnn_wrapper: the number of MPI ranks on compute node cannot exceed that of GPUs when NCCL is used");
    }

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
  free_work_spaces();

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

  // Free work spaces
  free_work_spaces();

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

  // Free work spaces
  free_work_spaces();

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
  std::vector<void *> work_spaces;
  for(int i=0; i<m_num_gpus; ++i) {
    work_spaces.push_back(get_work_space(i));
  }
  return work_spaces;
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
  if(m_work_space_sizes[i] != size) {
    m_work_space_sizes[i] = size;
    if(m_work_spaces[i] != nullptr) {
      CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      CHECK_CUDA(cudaFree(m_work_spaces[i]));
    }
    m_work_spaces[i] = nullptr;
  }
}

void cudnn_manager::set_maximum_work_space_size(int i) {

  // Search parameters for work space size
  const size_t min_work_space_size = 1024;
  const double decay_factor = 0.8;
  size_t free_memory, total_memory;
  CHECK_CUDA(cudaMemGetInfo(&free_memory, &total_memory));

  // Clear work space
  set_work_space_size(i, 0);

  // Try allocating work spaces until we find a valid size
  size_t work_space_size = free_memory;
  void* work_space = nullptr;
  while(work_space_size > min_work_space_size && work_space == nullptr) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    cudaError_t status = cudaMalloc(&work_space, work_space_size);
    if(status != cudaErrorMemoryAllocation) {
      FORCE_CHECK_CUDA(status);
      m_work_spaces[i] = work_space;
      m_work_space_sizes[i] = work_space_size;
    }
    else {
      work_space = nullptr;
    }
    work_space_size = decay_factor * work_space_size;
  }

}

void cudnn_manager::free_work_spaces() {
  for(int i=0; i<m_num_gpus; ++i) {
    set_work_space_size(i, 0);
  }
}

std::vector<DataType*> cudnn_manager::copy(const std::vector<DataType*>& gpu_data,
                                           int height,
                                           int width_per_gpu,
                                           int leading_dim) {
  free_work_spaces();
  leading_dim = std::max(leading_dim, height);
  std::vector<DataType*> output_gpu_data;
  if(!gpu_data.empty()) {
    allocate_on_gpus(output_gpu_data, leading_dim, width_per_gpu);
    copy_on_gpus(output_gpu_data, gpu_data, height, width_per_gpu,
                 leading_dim, leading_dim);
  }
  return output_gpu_data;
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

#ifdef __LIB_NCCL
uint64_t cudnn_manager::getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}
#endif


void cudnn_manager::nccl_setup() {
#ifdef __LIB_NCCL

  int nProcs = comm->get_procs_per_model();
  int myid = comm->get_model_rank();
  int localRank = comm->get_rank_in_node();

  ncclUniqueId ncclId;
  if (myid == 0) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
  }
  El::mpi::Comm model_comm = comm->get_model_comm();
  MPI_Comm mpicomm = model_comm.comm;

  /**
  Not sure if we can use Elemental's broadcast for new date type 'ncclUniqeId'. 
  For that reason, raw MPI_Bcast is used instead.
  
  El::mpi::Broadcast(&ncclId, 1, 0, model_comm); */

  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, mpicomm);

  if (nProcs == 1) {
    int gpuArray = 0;
    NCCLCHECK(ncclCommInitAll(&m_nccl_comm, 1, &gpuArray));
  } 
  else {
    NCCLCHECK(ncclGroupStart());
    FORCE_CHECK_CUDA(cudaSetDevice(localRank));
    NCCLCHECK(ncclCommInitRank(&m_nccl_comm, nProcs, ncclId, myid)); 
    NCCLCHECK(ncclGroupEnd());
  }
}

#endif
}

void cudnn_manager::nccl_destroy() {
#ifdef __LIB_NCCL
  ncclCommDestroy(m_nccl_comm);
#endif
}

void cudnn::print_version() {
  std::cout << "cudnnGetVersion() : " << (int)cudnnGetVersion() << " , "
            << "CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION
            << std::endl;
}

cudnnDataType_t cudnn::get_cudnn_data_type() {
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

void cudnn::set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                                  int num_samples,
                                  const std::vector<int>& sample_dims) {

  // Create tensor descriptor if needed
  if(desc == nullptr) {
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
  }

  // Determine tensor dimensions
  // Note: cuDNN tensors should have at least 4 dimension
  std::vector<int> dims = sample_dims;
  dims.insert(dims.begin(), num_samples);
  while(dims.size() < 4) {
    dims.push_back(1);
  }

  // Determine tensor strides
  std::vector<int> strides(dims.size());
  strides.back() = 1;
  for(int i=dims.size()-1; i>0; --i) {
    strides[i-1] = strides[i] * dims[i];
  }

  // Set cuDNN tensor descriptor
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                         cudnn::get_cudnn_data_type(),
                                         dims.size(),
                                         dims.data(),
                                         strides.data()));

}

void cudnn::copy_tensor_cudnn_desc(const cudnnTensorDescriptor_t& src,
                                   cudnnTensorDescriptor_t& dst) {

  // Create or destroy descriptor if needed
  if(src != nullptr && dst == nullptr) {
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&dst));
  }
  else if(src == nullptr && dst != nullptr) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(dst));
    dst = nullptr;
  }

  // Copy descriptor data if needed
  if(src != nullptr) {
      cudnnDataType_t data_type;
      int num_dims;
      CHECK_CUDNN(cudnnGetTensorNdDescriptor(src,
                                             0,
                                             &data_type,
                                             &num_dims,
                                             nullptr,
                                             nullptr));
      std::vector<int> dims(num_dims), strides(num_dims);
      CHECK_CUDNN(cudnnGetTensorNdDescriptor(src,
                                             num_dims,
                                             &data_type,
                                             &num_dims,
                                             dims.data(),
                                             strides.data()));
      CHECK_CUDNN(cudnnSetTensorNdDescriptor(dst,
                                             data_type,
                                             num_dims,
                                             dims.data(),
                                             strides.data()));
  }

}

void cudnn::copy_kernel_cudnn_desc(const cudnnFilterDescriptor_t& src,
                                   cudnnFilterDescriptor_t& dst) {

  // Create or destroy descriptor if needed
  if(src != nullptr && dst == nullptr) {
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&dst));
  }
  else if(src == nullptr && dst != nullptr) {
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(dst));
    dst = nullptr;
  }

  // Copy descriptor data if needed
  if(src != nullptr) {
    cudnnDataType_t data_type;
    cudnnTensorFormat_t format;
    int num_dims;
    CHECK_CUDNN(cudnnGetFilterNdDescriptor(src,
                                           0,
                                           &data_type,
                                           &format,
                                           &num_dims,
                                           nullptr));
    std::vector<int> dims(num_dims);
    CHECK_CUDNN(cudnnGetFilterNdDescriptor(src,
                                           num_dims,
                                           &data_type,
                                           &format,
                                           &num_dims,
                                           dims.data()));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(dst,
                                           data_type,
                                           format,
                                           num_dims,
                                           dims.data()));
  }

}

void cudnn::copy_convolution_cudnn_desc(const cudnnConvolutionDescriptor_t& src,
                                        cudnnConvolutionDescriptor_t& dst) {

  // Create or destroy descriptor if needed
  if(src != nullptr && dst == nullptr) {
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&dst));
  }
  else if(src == nullptr && dst != nullptr) {
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(dst));
    dst = nullptr;
  }

  // Copy descriptor data if needed
  if(src != nullptr) {
    cudnnConvolutionMode_t mode;
    cudnnDataType_t data_type;
    int num_dims;
    CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(src,
                                                0,
                                                &num_dims,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                &mode,
                                                &data_type));
    std::vector<int> pads(num_dims), strides(num_dims), upscales(num_dims);
    CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(src,
                                                num_dims,
                                                &num_dims,
                                                pads.data(),
                                                strides.data(),
                                                upscales.data(),
                                                &mode,
                                                &data_type));
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(dst,
                                                num_dims,
                                                pads.data(),
                                                strides.data(),
                                                upscales.data(),
                                                mode,
                                                data_type));
  }

}

void cudnn::copy_pooling_cudnn_desc(const cudnnPoolingDescriptor_t& src,
                                    cudnnPoolingDescriptor_t& dst) {

  // Create or destroy descriptor if needed
  if(src != nullptr && dst == nullptr) {
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&dst));
  }
  else if(src == nullptr && dst != nullptr) {
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(dst));
    dst = nullptr;
  }

  // Copy descriptor data if needed
  if(src != nullptr) {
    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t nan_propagation;
    int num_dims;
    CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                            0,
                                            &mode,
                                            &nan_propagation,
                                            &num_dims,
                                            nullptr,
                                            nullptr,
                                            nullptr));
    std::vector<int> dims(num_dims), pads(num_dims), strides(num_dims);
    CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                            0,
                                            &mode,
                                            &nan_propagation,
                                            &num_dims,
                                            dims.data(),
                                            pads.data(),
                                            strides.data()));
    CHECK_CUDNN(cudnnSetPoolingNdDescriptor(dst,
                                            mode,
                                            nan_propagation,
                                            num_dims,
                                            dims.data(),
                                            pads.data(),
                                            strides.data()));
  }

}

void cudnn::copy_activation_cudnn_desc(const cudnnActivationDescriptor_t& src,
                                       cudnnActivationDescriptor_t& dst) {

  // Create or destroy descriptor if needed
  if(src != nullptr && dst == nullptr) {
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&dst));
  }
  else if(src == nullptr && dst != nullptr) {
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(dst));
    dst = nullptr;
  }

  // Copy descriptor data if needed
  if(src != nullptr) {
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t nan_propagation;
    double relu_ceiling;
    CHECK_CUDNN(cudnnGetActivationDescriptor(src,
                                             &mode,
                                             &nan_propagation,
                                             &relu_ceiling));
    CHECK_CUDNN(cudnnSetActivationDescriptor(dst,
                                             mode,
                                             nan_propagation,
                                             relu_ceiling));
  }

}

void cudnn::copy_lrn_cudnn_desc(const cudnnLRNDescriptor_t& src,
                                cudnnLRNDescriptor_t& dst) {

  // Create or destroy descriptor if needed
  if(src != nullptr && dst == nullptr) {
    CHECK_CUDNN(cudnnCreateLRNDescriptor(&dst));
  }
  else if(src == nullptr && dst != nullptr) {
    CHECK_CUDNN(cudnnDestroyLRNDescriptor(dst));
    dst = nullptr;
  }

  // Copy descriptor data if needed
  if(src != nullptr) {
    unsigned n;
    double alpha, beta, k;
    CHECK_CUDNN(cudnnGetLRNDescriptor(src, &n, &alpha, &beta, &k));
    CHECK_CUDNN(cudnnSetLRNDescriptor(dst, n, alpha, beta, k));
  }

}

#endif // #ifdef __LIB_CUDNN
