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

#ifndef CUDNN_WRAPPER_HPP_INCLUDED
#define CUDNN_WRAPPER_HPP_INCLUDED

#include <vector>
#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/layers/lbann_layer_activations.hpp"

#ifdef __LIB_CUDNN
#include <cuda.h>
#include <cudnn.h>
#include <cub/util_allocator.cuh>
#endif // #ifdef __LIB_CUDNN

// Error utility macros
#ifdef __LIB_CUDNN
#ifdef LBANN_DEBUG
#define checkCUDA(status) {                                             \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      throw lbann::lbann_exception("CUDA error");                       \
    }                                                                   \
  }
#define checkCUDNN(status) {                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
      std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      throw lbann::lbann_exception("cuDNN error");                      \
    }                                                                   \
  }
#else
#define checkCUDA(status)  status
#define checkCUDNN(status) status
#endif // #ifdef LBANN_DEBUG
#endif // #ifdef __LIB_CUDNN

namespace cudnn
{

  /** cuDNN manager class */
  class cudnn_manager
  {
#ifdef __LIB_CUDNN

  public:
    /** Constructor
     *  @param _comm         Pointer to LBANN communicator
     *  @param max_num_gpus  Maximum Number of available GPUs. If
     *                       negative, then use all available GPUs.
     */
    cudnn_manager(lbann::lbann_comm* _comm, int max_num_gpus = -1);

    /** Destructor */
    ~cudnn_manager();

    /** Print cuDNN version information to standard output. */
    void print_version() const;
    /** Get cuDNN data type associated with C++ data type. */
    cudnnDataType_t get_cudnn_data_type() const;

    /** Get number of GPUs assigned to current MPI rank. */
    int get_num_gpus() const;
    /** Get number of GPUs on current node. */
    int get_num_total_gpus() const;
    /** Get GPU memory allocator. */
    cub::CachingDeviceAllocator* get_gpu_memory();
    /** Get CUDA streams for current MPI rank. */
    std::vector<cudaStream_t>* get_streams();

    /// Register a block of memory to pin
    void pin_ptr(void* ptr, size_t sz);
    /// Pin the memory block of a matrix
    void pin_memory_block(ElMat *mat);
    /// Unregister a block of pinnedmemory
    void unpin_ptr(void* ptr);
    /// Unregister all the memories registered to pin
    void unpin_ptrs(void);

  public:

    /** LBANN communicator */
    lbann::lbann_comm* comm;

    /** Number of GPUs for current MPI rank */
    int m_num_gpus;
    /** Number of available GPUs */
    int m_num_total_gpus;

    /** GPU memory allocator
     *  Faster than cudaMalloc/cudaFree since it uses a memory pool */
    cub::CachingDeviceAllocator* m_gpu_memory;

    /** GPUs for current MPI rank */
    std::vector<int> m_gpus;
    /** CUDA streams for current MPI rank */
    std::vector<cudaStream_t> m_streams;
    /** cuDNN handles for current MPI rank */
    std::vector<cudnnHandle_t> m_handles;
    /// pinned memory addresses
    std::map<void*, size_t> pinned_ptr;

#endif // #ifdef __LIB_CUDNN
  };

}

#endif // CUDNN_WRAPPER_HPP_INCLUDED
