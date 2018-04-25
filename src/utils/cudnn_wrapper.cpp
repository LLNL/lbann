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
#include "lbann/utils/cublas_wrapper.hpp"
#include "lbann/utils/exception.hpp"

#include <iostream>

#include "El.hpp"
#include <unistd.h>

#ifdef LBANN_HAS_CUDNN

namespace lbann
{
namespace cudnn
{

matrix::matrix(cudnn_manager *cudnn, int height, int width_per_gpu)
  : m_cudnn(cudnn),
    m_height(0),
    m_width_per_gpu(0),
    m_leading_dim(0),
    m_is_view(false),
    m_is_locked(false) {
  if (m_cudnn != nullptr) {
    m_data.assign(m_cudnn->get_num_gpus(), nullptr);
    resize(height, width_per_gpu);
  }
}

matrix::matrix(const matrix& other)
  : m_cudnn(other.m_cudnn),
    m_height(0),
    m_width_per_gpu(0),
    m_leading_dim(0),
    m_is_view(false),
    m_is_locked(false) {
  if (m_cudnn != nullptr) {
    m_data.assign(m_cudnn->get_num_gpus(), nullptr);
    copy(other);
  }
}

matrix::matrix(matrix&& other)
  : m_cudnn(other.m_cudnn),
    m_data(other.m_data),
    m_height(other.m_height),
    m_width_per_gpu(other.m_width_per_gpu),
    m_leading_dim(other.m_leading_dim),
    m_is_view(other.m_is_view),
    m_is_locked(other.m_is_locked) {
  other.m_is_view = true;
}

matrix& matrix::operator=(const matrix& other) {
  clear();
  m_cudnn = other.m_cudnn;
  if (m_cudnn != nullptr) {
    copy(other);
  }
  return *this;
}

matrix& matrix::operator=(matrix&& other) {
  clear();
  m_cudnn = other.m_cudnn;
  m_data = other.m_data;
  m_height = other.m_height;
  m_width_per_gpu = other.m_width_per_gpu;
  m_leading_dim = other.m_leading_dim;
  m_is_view = other.m_is_view;
  m_is_locked = other.m_is_locked;
  other.m_is_view = true;
  return *this;
}

matrix::~matrix() {
  clear();
}

void matrix::clear() {
  if (m_cudnn != nullptr && !m_is_view) {
    m_cudnn->deallocate_on_gpus(m_data);
  }
  const int num_gpus = m_cudnn != nullptr ? m_cudnn->get_num_gpus() : 0;
  m_data.assign(num_gpus, nullptr);
  m_height = 0;
  m_width_per_gpu = 0;
  m_leading_dim = 0;
  m_is_view = false;
  m_is_locked = false;
}

void matrix::resize(int height, int width_per_gpu) {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted to resize matrix without cuDNN manager");
  }
  if (m_height != height || m_width_per_gpu != width_per_gpu) {
    clear();
    if (height > 0 && width_per_gpu > 0) {
      m_cudnn->allocate_on_gpus(m_data, height, width_per_gpu);
      m_height = height;
      m_width_per_gpu = width_per_gpu;
      m_leading_dim = height;
      m_is_view = false;
      m_is_locked = false;
    }
  }
}

void matrix::copy(const matrix& other) {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted to copy into matrix without cuDNN manager");
  } else if (m_is_locked) {
    throw lbann::lbann_exception("cudnn::matrix: attempted to copy into a locked matrix");
  }
  resize(other.m_height, other.m_width_per_gpu);
  if (m_height > 0 && m_width_per_gpu > 0) {
    m_cudnn->copy_on_gpus(m_data, other.m_data,
                          m_height, m_width_per_gpu,
                          other.m_leading_dim, m_leading_dim);
  }
}

void matrix::view(matrix& other) {
  attach(other.get_data(),
         other.get_height(),
         other.get_width_per_gpu(),
         other.get_leading_dim());
}

void matrix::locked_view(const matrix& other) {
  locked_attach(other.get_locked_data(),
                other.get_height(),
                other.get_width_per_gpu(),
                other.get_leading_dim());
}

void matrix::zero() {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted to zero out matrix without cuDNN manager");
  } else if (m_is_locked) {
    throw lbann::lbann_exception("cudnn::matrix: attempted to zero out a locked matrix");
  }
  if (m_height > 0 && m_width_per_gpu > 0) {
    m_cudnn->clear_on_gpus(m_data, m_height, m_width_per_gpu, m_leading_dim);
  }
}

void matrix::attach(std::vector<DataType*>& data,
                    int height,
                    int width_per_gpu,
                    int leading_dim) {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted to attach matrix without cuDNN manager");
  }
  clear();
  m_data = data;
  m_height = height;
  m_width_per_gpu = width_per_gpu;
  m_leading_dim = std::max(leading_dim, height);
  m_is_view = true;
  m_is_locked = false;
}

void matrix::locked_attach(const std::vector<DataType*>& data,
                           int height,
                           int width_per_gpu,
                           int leading_dim) {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted to attach matrix without cuDNN manager");
  }
  clear();
  m_data = data;
  m_height = height;
  m_width_per_gpu = width_per_gpu;
  m_leading_dim = std::max(leading_dim, height);
  m_is_view = true;
  m_is_locked = true;
}

void matrix::attach_to_work_spaces(int height,
                                   int width_per_gpu,
                                   int leading_dim) {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted to attach matrix without cuDNN manager");
  }
  clear();

  // Check if work space size is valid
  leading_dim = std::max(leading_dim, height);
  const size_t required_size = leading_dim * width_per_gpu * sizeof(DataType);
  const size_t work_space_size = m_cudnn->get_minimum_work_space_size();
  if (work_space_size < required_size) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "insufficient GPU work space "
        << "(requires " << required_size << " bytes on each GPU, "
        << "but only have " << work_space_size << " bytes)";
    throw lbann_exception(err.str());
  }

  // Attach matrix to work spaces
  std::vector<DataType*> work_spaces;
  for (auto&& ptr : m_cudnn->get_work_spaces()) {
    work_spaces.push_back(static_cast<DataType*>(ptr));
  }
  attach(work_spaces, height, width_per_gpu, leading_dim);

}

std::vector<DataType*>& matrix::get_data() {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted access data of matrix without cuDNN manager");
  } else if (m_is_locked) {
    throw lbann::lbann_exception("cudnn::matrix: attempted access mutable data of locked matrix");
  }
  return m_data;
}

const std::vector<DataType*>& matrix::get_locked_data() const {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted access data of matrix without cuDNN manager");
  }
  return m_data;
}

DataType* matrix::get_data(int i) {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted access data of matrix without cuDNN manager");
  } else if (i < 0 || i >= m_cudnn->get_num_gpus()) {
    throw lbann::lbann_exception("cudnn::matrix: attempted access data on invalid GPU");
  } else if (m_is_locked) {
    throw lbann::lbann_exception("cudnn::matrix: attempted access mutable data of locked matrix");
  }
  return m_data[i];
}

const DataType* matrix::get_locked_data(int i) const {
  if (m_cudnn == nullptr) {
    throw lbann::lbann_exception("cudnn::matrix: attempted access data of matrix without cuDNN manager");
  } else if (i < 0 || i >= m_cudnn->get_num_gpus()) {
    throw lbann::lbann_exception("cudnn::matrix: attempted access data on invalid GPU");
  }
  return m_data[i];
}

/// It is assumed the number of processes and the number of GPUs on a compute node are equal
cudnn_manager::cudnn_manager(lbann::lbann_comm *_comm,
                             size_t work_space_size,
                             int max_num_gpus,
                             bool nccl_used)
    : comm(_comm) {

    // Indicate whether NCCL is used
#ifdef LBANN_HAS_NCCL2
    m_nccl_used = nccl_used;
#else
    if (nccl_used) {
        throw lbann::lbann_exception("cudnn_wrapper: NCCL is requested, but not enabled");
    }
    m_nccl_used = false;
#endif

    // Determine number of MPI ranks on current compute node
    const int rank_in_node = comm->get_rank_in_node();
    const int procs_per_node = comm->get_procs_per_node();

    // Determine number of visible GPUs
    CHECK_CUDA(cudaGetDeviceCount(&m_num_visible_gpus));
    if(max_num_gpus >= 0 && max_num_gpus < m_num_visible_gpus) {
        m_num_visible_gpus = max_num_gpus;
    }
    if(m_num_visible_gpus < 1) {
        throw lbann::lbann_exception("cudnn_wrapper: no GPUs found");
    }
    /// It is assumed that the number of processes on this node is equal to the total number of GPUs available
/*
  if(procs_per_node != m_num_visible_gpus){
  throw lbann::lbann_exception("cudnn_wrapper: the number of MPI ranks is different from than the number of GPUs available on this node");
  }
*/

    // Assign GPUs to process
    int gpu_start, gpu_end;
    const char* visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    if(visible_devices != nullptr && strlen(visible_devices) > 0) {
        // Use all visible GPUs if specified with an environment variable
        gpu_start = 0;
        gpu_end = m_num_visible_gpus;
    }
    else if(m_num_visible_gpus >= procs_per_node) {
        // Case where compute node has more GPUs than MPI ranks
        const int gpus_per_proc = m_num_visible_gpus / procs_per_node;
        const int num_leftover_gpus = m_num_visible_gpus % procs_per_node;
        gpu_start = rank_in_node * gpus_per_proc;
        gpu_end = (rank_in_node + 1) * gpus_per_proc;
        if(rank_in_node < num_leftover_gpus) {
            gpu_start += rank_in_node;
            gpu_end += rank_in_node + 1;
        }
        else {
            gpu_start += num_leftover_gpus;
            gpu_end += num_leftover_gpus;
        }
    }
    else {
        // Case where compute node has fewer GPUs than MPI ranks
        // TODO: Support case where MPI ranks have to share GPUs
        std::stringstream err;
        err << "cudnn_wrapper: cannot have " << procs_per_node << " processes "
            << "on a node with " << m_num_visible_gpus << " GPUs";
        throw lbann_exception(err.str());
        gpu_start = rank_in_node % m_num_visible_gpus;
        gpu_end = gpu_start + 1;
    }

    // Construct GPU objects
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
        FORCE_CHECK_CUBLAS(cublasSetStream(m_cublas_handles.back(), m_streams.back()));
        FORCE_CHECK_CUBLAS(cublasSetPointerMode(m_cublas_handles.back(), CUBLAS_POINTER_MODE_HOST));
    }

    // Get number of GPUs for current MPI rank
    m_num_gpus = m_gpus.size();

    // Make sure LBANN communicator knows GPUs and CUDA streams
    /**  @todo This is a kludge. A better solution would be to
     *   refactor the cuDNN manager and make the LBANN communicator
     *   responsible for GPU management.
     */
    comm->get_gpus() = m_gpus;
    comm->get_cuda_streams() = m_streams;

    // Initialize work spaces
    m_work_spaces = std::vector<void *>(m_num_gpus, nullptr);
    m_work_space_sizes = std::vector<size_t>(m_num_gpus, 0);
    for (int i = 0; i < m_num_gpus; ++i) {
      set_work_space_size(work_space_size);
    }

    /// Setting up for NCCL collective calls
    /// NOTE: For whoever makes changes in this file, please make sure following if statement comes last.
    if(m_nccl_used){
        nccl_setup();
    }
}

cudnn_manager::~cudnn_manager() {
  // Free work spaces
  free_work_spaces();

  // Destroy cuDNN handles
  // Use a try-catch block for FORCE_CHECK_{CUDA |CUDNN | CUBLAS} in the
  // destructor -- these could thrown an exception and destructors are
  // considered to be noexcept by default
  try
  {
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
  catch(const std::exception& e)
  {
    std::cerr << "~cudnn_manager: try ... catch " << e.what() << std::endl;
    std::terminate();
  }

  /// NCCL clear
  if(m_nccl_used)
  {
      nccl_destroy();
  }
}

void cudnn_manager::cudnn_manager::allocate_on_gpus(std::vector<DataType *>& gpu_data,
                                                    int height,
                                                    int width_per_gpu) {

  // Check that list of pointers is valid
  if(!gpu_data.empty()) {
    if((int) gpu_data.size() != m_num_gpus) {
      throw lbann_exception("cudnn_wrapper: number of GPU memory pointers doesn't match number of GPUs");
    }
    for(int i=0; i<m_num_gpus; ++i) {
      if(gpu_data[i] != nullptr) {
        throw lbann_exception("cudnn_wrapper: overwriting non-null pointer with newly allocated GPU memory");
      }
    }
  }

  // Allocate GPU memory
  gpu_data.assign(m_num_gpus, nullptr);
  if(height > 0 && width_per_gpu > 0) {

    // Free work spaces
    free_work_spaces();

    // Allocate memory on GPUs
    const size_t size = height * width_per_gpu * sizeof(DataType);
    for(int i=0; i<m_num_gpus; ++i) {
      FORCE_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      const cudaError_t status = cudaMalloc((void **) &gpu_data[i], size);

      // Check that allocation is successful
      if(status == cudaErrorMemoryAllocation) {
        size_t free_memory, total_memory;
        CHECK_CUDA(cudaMemGetInfo(&free_memory, &total_memory));
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
            << "could not allocate GPU memory on GPU " << m_gpus[i] << " "
            << "(" << size << " bytes requested, "
            << free_memory << " bytes available, "
            << total_memory << " bytes total)";
        throw lbann_exception(err.str());
      } else {
        FORCE_CHECK_CUDA(status);
      }

    }
  }

  // Set entries to zero
  clear_on_gpus(gpu_data, height, width_per_gpu);

}

void cudnn_manager::cudnn_manager::deallocate_on_gpus(std::vector<DataType *>& gpu_data) {

  // Stop early if deallocation is not needed
  if (gpu_data.empty()) { return; }
  if ((int) gpu_data.size() != m_num_gpus) {
    throw lbann_exception("cudnn_wrapper: number of GPU memory pointers doesn't match number of GPUs");
  }
  if (std::count(gpu_data.begin(), gpu_data.end(), nullptr)
      == m_num_gpus) {
    gpu_data.clear();
    return;
  }

  // Free work spaces
  free_work_spaces();

  // Deallocate GPU memory
  for(int i=0; i<m_num_gpus; ++i) {
    FORCE_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    FORCE_CHECK_CUDA(cudaFree(gpu_data[i]));
  }
  gpu_data.clear();

}

void cudnn_manager::cudnn_manager::clear_on_gpu(int i,
                                                DataType* gpu_data,
                                                int height,
                                                int width,
                                                int leading_dim) {
  CHECK_CUDA(cudaSetDevice(m_gpus[i]));
  leading_dim = std::max(leading_dim, height);
  if (height <= 0 || width <= 0) { return; }
  else if (leading_dim == height) {
    CHECK_CUDA(cudaMemsetAsync(gpu_data,
                               0,
                               height * width * sizeof(DataType),
                               m_streams[i]));
  }
  else {
    CHECK_CUDA(cudaMemset2DAsync(gpu_data,
                                 leading_dim * sizeof(DataType),
                                 0,
                                 height * sizeof(DataType),
                                 width,
                                 m_streams[i]));
  }
}

void cudnn_manager::cudnn_manager::copy_to_gpu(int i,
                                               DataType* gpu_data,
                                               const Mat& cpu_data,
                                               int gpu_data_leading_dim) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    const int height = cpu_data.Height();
    const int width = cpu_data.Width();
    gpu_data_leading_dim = std::max(gpu_data_leading_dim, height);
    const int cpu_data_leading_dim = cpu_data.LDim();
    if (height == 0 || width == 0) { return; }
    else if (gpu_data_leading_dim == height
             && cpu_data_leading_dim == height) {
        CHECK_CUDA(cudaMemcpyAsync(gpu_data,
                                   cpu_data.LockedBuffer(),
                                   height * width * sizeof(DataType),
                                   cudaMemcpyHostToDevice,
                                   m_streams[i]));
    }
    else {
        CHECK_CUDA(cudaMemcpy2DAsync(gpu_data,
                                     gpu_data_leading_dim * sizeof(DataType),
                                     cpu_data.LockedBuffer(),
                                     cpu_data.LDim() * sizeof(DataType),
                                     height * sizeof(DataType),
                                     width,
                                     cudaMemcpyHostToDevice,
                                     m_streams[i]));
    }
}

void cudnn_manager::cudnn_manager::copy_from_gpu(int i,
                                                 Mat& cpu_data,
                                                 const DataType* gpu_data,
                                                 int gpu_data_leading_dim) {

  CHECK_CUDA(cudaSetDevice(m_gpus[i]));
  const int height = cpu_data.Height();
  const int width = cpu_data.Width();
  gpu_data_leading_dim = std::max(gpu_data_leading_dim, height);
  const int cpu_data_leading_dim = cpu_data.LDim();
  if (height <= 0 || width <= 0) { return; }
  else if (gpu_data_leading_dim == height
           && cpu_data_leading_dim == height) {
    CHECK_CUDA(cudaMemcpyAsync(cpu_data.Buffer(),
                               gpu_data,
                               height * width * sizeof(DataType),
                               cudaMemcpyDeviceToHost,
                               m_streams[i]));
  }
  else {
    CHECK_CUDA(cudaMemcpy2DAsync(cpu_data.Buffer(),
                                 cpu_data_leading_dim * sizeof(DataType),
                                 gpu_data,
                                 gpu_data_leading_dim * sizeof(DataType),
                                 height * sizeof(DataType),
                                 width,
                                 cudaMemcpyDeviceToHost,
                                 m_streams[i]));
  }
}

void cudnn_manager::cudnn_manager::clear_on_gpus(std::vector<DataType *>& gpu_data,
                                                 int height,
                                                 int width_per_gpu,
                                                 int leading_dim) {
    if(!gpu_data.empty()) {
        for(int i=0; i<m_num_gpus; ++i) {
            clear_on_gpu(i, gpu_data[i], height, width_per_gpu, leading_dim);
        }
    }
}

void cudnn_manager::cudnn_manager::clear_unused_columns_on_gpus(std::vector<DataType *>& gpu_data,
                                                                int height,
                                                                int width,
                                                                int width_per_gpu,
                                                                int leading_dim) {
    if(!gpu_data.empty()) {
        leading_dim = std::max(leading_dim, height);
        for(int i=0; i<m_num_gpus; ++i) {
            const int first_pos = std::min(i * width_per_gpu, width);
            const int last_pos = std::min((i+1) * width_per_gpu, width);
            const int current_width = last_pos - first_pos;
            clear_on_gpu(i,
                         gpu_data[i] + leading_dim * current_width,
                         height,
                         width_per_gpu - current_width,
                         leading_dim);
        }
    }
}

void cudnn_manager::cudnn_manager::set_on_gpus(std::vector<DataType *>& gpu_data,
                                               DataType val,
                                               int height,
                                               int width_per_gpu) {
  if(!gpu_data.empty()) {
    for(int i=0; i<m_num_gpus; ++i) {
      set_on_gpu(i, gpu_data[i], height, width_per_gpu);
    }
  }
}

void cudnn_manager::cudnn_manager::copy_on_gpus(std::vector<DataType *>& gpu_dst_data,
                                                const std::vector<DataType *>& gpu_src_data,
                                                int height,
                                                int width_per_gpu,
                                                int src_leading_dim,
                                                int dst_leading_dim) {

    // Check inputs
    if (gpu_dst_data.empty() || gpu_src_data.empty()) {
        throw lbann_exception("cudnn_wrapper: attempted to copy on GPUs before allocating GPU memory");
    }

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

    // Check inputs
    if (gpu_data.empty()) {
        throw lbann_exception("cudnn_wrapper: attempted to scatter to GPUs before allocating GPU memory");
    }

    // Get matrix properties
    const int height = cpu_data.Height();
    const int width = cpu_data.Width();

    // Perform memory transfer on each GPU
    #pragma omp parallel for
    for(int i=0; i<m_num_gpus; ++i) {
        const int first_pos = std::min(i * width_per_gpu, width);
        const int last_pos = std::min((i+1) * width_per_gpu, width);
        if (first_pos < last_pos) {
            const auto& cpu_data_view = El::LockedView(cpu_data, El::ALL, El::IR(first_pos, last_pos));
            copy_to_gpu(i, gpu_data[i], cpu_data_view, gpu_data_leading_dim);
        }
    }

    // Clear unused columns
    clear_unused_columns_on_gpus(gpu_data,
                                 height,
                                 width,
                                 width_per_gpu,
                                 gpu_data_leading_dim);

}


void cudnn_manager::cudnn_manager::gather_from_gpus(Mat& cpu_data,
                                                    const std::vector<DataType *>& gpu_data,
                                                    int width_per_gpu,
                                                    int gpu_data_leading_dim) {
    if (gpu_data.empty()) {
        throw lbann_exception("cudnn_wrapper: attempted to gather from GPUs before allocating GPU memory");
    }

    const int width = cpu_data.Width();
    #pragma omp parallel for
    for(int i=0; i<m_num_gpus; ++i) {
        const int first_pos = std::min(i * width_per_gpu, width);
        const int last_pos = std::min((i+1) * width_per_gpu, width);
        if (first_pos < last_pos) {
            auto&& cpu_data_view = El::View(cpu_data, El::ALL, El::IR(first_pos, last_pos));
            copy_from_gpu(i, cpu_data_view, gpu_data[i], gpu_data_leading_dim);
        }
    }
}

void cudnn_manager::cudnn_manager::broadcast_to_gpus(std::vector<DataType *>& gpu_data,
                                                     const Mat& cpu_data,
                                                     int gpu_data_leading_dim) {
    if (gpu_data.empty()) {
        throw lbann_exception("cudnn_wrapper: attempted to broadcast to GPUs before allocating GPU memory");
    }
    #pragma omp parallel for
    for(int i=0; i<m_num_gpus; ++i) {
        copy_to_gpu(i, gpu_data[i], cpu_data, gpu_data_leading_dim);
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

int cudnn_manager::get_num_visible_gpus() const {
    return m_num_visible_gpus;
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
  if(i >= m_num_gpus) {
    throw lbann_exception("cudnn_wrapper: tried to access invalid work space");
  }
  m_work_spaces.resize(m_num_gpus, nullptr);
  get_work_space_size(i); // Reallocate work space if needed
  return m_work_spaces[i];
}

size_t cudnn_manager::get_minimum_work_space_size() {
  if (m_num_gpus <= 0) { return 0; }
  size_t size = get_work_space_size(0);
  for(int i=1; i<m_num_gpus; ++i) {
    size = std::min(size, get_work_space_size(i));
  }
  return size;
}

std::vector<size_t> cudnn_manager::get_work_space_sizes() {
  std::vector<size_t> work_space_sizes;
  for(int i=0; i<m_num_gpus; ++i) {
    work_space_sizes.push_back(get_work_space_size(i));
  }
  return work_space_sizes;
};

size_t cudnn_manager::get_work_space_size(int i) {
  if(i >= m_num_gpus) {
    throw lbann_exception("cudnn_wrapper: tried to access invalid work space size");
  }
  m_work_space_sizes.resize(m_num_gpus, 0);
  return m_work_space_sizes[i];
}

void cudnn_manager::set_work_space_size(size_t size, int i) {
  free_work_space(i);
  CHECK_CUDA(cudaSetDevice(m_gpus[i]));
  FORCE_CHECK_CUDA(cudaMalloc(&m_work_spaces[i], size));
  m_work_space_sizes[i] = size;
}

void cudnn_manager::free_work_space(int i) {
  if(i < 0 || i >= m_num_gpus) {
    throw lbann_exception("cudnn_wrapper: tried to access size of invalid work space");
  }
  if (m_work_spaces[i] != nullptr) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    CHECK_CUDA(cudaFree(m_work_spaces[i]));
  }
  m_work_spaces[i] = nullptr;
  m_work_space_sizes[i] = 0;
}

void cudnn_manager::free_work_spaces() {
    for(int i=0; i<m_num_gpus; ++i) {
        free_work_space(i);
    }
}

std::vector<DataType*> cudnn_manager::copy(const std::vector<DataType*>& gpu_data,
                                           int height,
                                           int width_per_gpu,
                                           int leading_dim) {
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
    Mat& mat_local = static_cast<CPUMat&>(mat.Matrix());
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
    const GPUMat mat_local_copy(static_cast<const GPUMat&>(mat.LockedMatrix()));

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

void cudnn_manager::nccl_setup() {

#ifdef LBANN_HAS_NCCL2
    if(m_num_gpus != 1){
        char line[1024];
        sprintf(line, "cudnn_manager: the number of GPUs assigned to process is %d; should be 1", m_num_gpus);
        throw lbann::lbann_exception(line);
    }

    /// Create nccl communicators
    int num_gpus_assigned = m_gpus.size();
    m_nccl_comm.resize(num_gpus_assigned);


    int nProcs = comm->get_procs_per_model();
    int myid = comm->get_rank_in_model();
    int total_num_comms = nProcs*num_gpus_assigned;

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

    /// todo@ check if we can use Elemental's broadcast
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, mpicomm);

    if (nProcs == 1) {
        int gpuArray = 0;
        NCCLCHECK(ncclCommInitAll(&(m_nccl_comm[0]), 1, &gpuArray));
    }
    else {
        if(num_gpus_assigned > 1) NCCLCHECK(ncclGroupStart());
        for(int i=0; i<num_gpus_assigned; i++){
            FORCE_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
            NCCLCHECK(ncclCommInitRank(&(m_nccl_comm[i]), total_num_comms, ncclId, num_gpus_assigned*myid+i));
        }
        if(num_gpus_assigned > 1) NCCLCHECK(ncclGroupEnd());

    }

#endif // #ifdef LBANN_HAS_NCCL2
}

void cudnn_manager::nccl_destroy() {
#ifdef LBANN_HAS_NCCL2
    int num_gpus_assigned = m_gpus.size();
    for(int i=0; i<num_gpus_assigned; i++){
        ncclCommDestroy(m_nccl_comm[i]);
    }
#endif // #ifdef LBANN_HAS_NCCL2
}

void print_version() {
    std::cout << "cudnnGetVersion() : " << (int)cudnnGetVersion() << " , "
              << "CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION
              << std::endl;
}

cudnnDataType_t get_cudnn_data_type() {
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

void set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                           int num_samples,
                           const std::vector<int>& sample_dims,
                           int sample_stride) {

    // Create tensor descriptor if needed
    if (desc == nullptr) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    }

    // Determine tensor dimensions
    // Note: cuDNN tensors should have at least 4 dimension
    std::vector<int> dims = sample_dims;
    while (dims.size() < 3) {
        dims.insert(dims.begin(), 1);
    }
    dims.insert(dims.begin(), num_samples);

    // Determine tensor strides
    std::vector<int> strides(dims.size());
    strides.back() = 1;
    for(int i = dims.size() - 1; i > 0; --i) {
        strides[i-1] = strides[i] * dims[i];
    }
    strides.front() = std::max(strides.front(), sample_stride);

    // Set cuDNN tensor descriptor
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                           get_cudnn_data_type(),
                                           dims.size(),
                                           dims.data(),
                                           strides.data()));

}

void set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                           int height,
                           int width,
                           int leading_dim) {

    // Create tensor descriptor if needed
    if (desc == nullptr) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    }

    // Determine tensor dimensions and strides
    // Note: cuDNN tensors should have at least 4 dimension
    leading_dim = std::max(height, leading_dim);
    const std::vector<int> dims = {1, 1, width, height};
    const std::vector<int> strides = {width * leading_dim,
                                      width * leading_dim,
                                      leading_dim,
                                      1};

    // Set cuDNN tensor descriptor
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                           get_cudnn_data_type(),
                                           dims.size(),
                                           dims.data(),
                                           strides.data()));

}

void copy_tensor_cudnn_desc(const cudnnTensorDescriptor_t& src,
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

void copy_kernel_cudnn_desc(const cudnnFilterDescriptor_t& src,
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

void copy_convolution_cudnn_desc(const cudnnConvolutionDescriptor_t& src,
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

void copy_pooling_cudnn_desc(const cudnnPoolingDescriptor_t& src,
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

void copy_activation_cudnn_desc(const cudnnActivationDescriptor_t& src,
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

void copy_lrn_cudnn_desc(const cudnnLRNDescriptor_t& src,
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

}// namespace cudnn
}// namespace lbann
#endif // #ifdef LBANN_HAS_CUDNN
