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
#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_CUDNN
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#ifdef LBANN_HAS_NCCL2
#include "nccl.h"

#define NCCLCHECK(cmd)                                                  \
    {                                                                   \
        ncclResult_t result__ = cmd;                                    \
        if (result__ != ncclSuccess)                                    \
        {                                                               \
            std::ostringstream oss;                                     \
            oss << "NCCL failure in " << __FILE__ << " at line "        \
                << __LINE__ << ": " << ncclGetErrorString(result__)     \
                << std::endl;                                           \
            throw lbann::lbann_exception(oss.str());                    \
        }                                                               \
    }

//#include "nccl1_compat.h"
//#include "common.h"
#endif // #ifdef LBANN_HAS_NCCL2

#endif // #ifdef LBANN_HAS_CUDNN

// Error utility macros
#ifdef LBANN_HAS_CUDNN
#define FORCE_CHECK_CUDA(cuda_call)                                     \
  do {                                                                  \
    const cudaError_t cuda_status = cuda_call;                          \
    if (cuda_status != cudaSuccess) {                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      throw lbann::lbann_exception("CUDA error");                       \
    }                                                                   \
  } while (0)
#define FORCE_CHECK_CUDNN(cudnn_call)                                   \
  do {                                                                  \
    const cudnnStatus_t cudnn_status = cudnn_call;                      \
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {                         \
      std::cerr << "cuDNN error: " << cudnnGetErrorString(cudnn_status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      throw lbann::lbann_exception("cuDNN error");                      \
    }                                                                   \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUDA(cuda_call)     FORCE_CHECK_CUDA(cuda_call)
#define CHECK_CUDNN(cudnn_call)   FORCE_CHECK_CUDNN(cudnn_call)
#else
#define CHECK_CUDA(cuda_call)     cuda_call
#define CHECK_CUDNN(cudnn_call)   cudnn_call
#endif // #ifdef LBANN_DEBUG
#endif // #ifdef LBANN_HAS_CUDNN

namespace lbann
{
namespace cudnn
{

// Forward declaration
class cudnn_manager;
class matrix;

/** GPU matrix class.
 *  This will soon be deprecated by native GPU support in Hydrogen.
 */
class matrix {
#ifdef LBANN_HAS_CUDNN
  
public:

  /** Constructor. */
  matrix(cudnn_manager *cudnn = nullptr,
         int height = 0,
         int width_per_gpu = 0);
  /** Copy constructor. */
  matrix(const matrix& other);
  /** Copy assignment operator. */
  matrix& operator=(const matrix& other);
  /** Move constructor. */
  matrix(matrix&& other);
  /** Move assignment operator. */
  matrix& operator=(matrix&& other);
  /** Destructor. */
  virtual ~matrix();

  /** Clear data. */
  void clear();
  /** Resize matrix. */
  void resize(int height, int width_per_gpu = 1);
  /** Copy matrix entries.
   *  The matrix is resized if needed.
   */
  void copy(const matrix& other);
  /** Make a view of another matrix.
   *  There is no check whether the original matrix is still valid.
   */
  void view(matrix& other);
  /** Make a view of another matrix.
   *  There is no check whether the original matrix is still valid.
   */
  void locked_view(const matrix& other);
  /** Set matrix entries to zero. */
  void zero();

  /** Attach matrix to GPU data. */
  void attach(std::vector<DataType*>& data,
              int height,
              int width_per_gpu = 1,
              int leading_dim = 0);
  /** Attach matrix to GPU data. */
  void locked_attach(const std::vector<DataType*>& data,
                     int height,
                     int width_per_gpu = 1,
                     int leading_dim = 0);

  /** Get matrix height. */
  inline int get_height() const { return m_height; }
  /** Get matrix width per GPU. */
  inline int get_width_per_gpu() const { return m_width_per_gpu; }
  /** Get matrix leading dimension. */
  inline int get_leading_dim() const { return m_leading_dim; }
  /** Whether the matrix is responsible for managing its memory. */
  inline bool is_view() const { return m_is_view; }
  /** Whether the matrix can modify its entries. */
  inline bool is_locked() const { return m_is_locked; }
  /** Get GPU data pointers. */
  std::vector<DataType*>& get_data();
  /** Get GPU data pointers (const). */
  const std::vector<DataType*>& get_locked_data() const;
  /** Get GPU data pointer on ith GPU. */
  DataType* get_data(int i);
  /** Get GPU data pointer on ith GPU (const). */
  const DataType* get_locked_data(int i) const;

private:

  /** cuDNN manager. */
  cudnn_manager *m_cudnn;
  /** GPU data pointers. */
  std::vector<DataType*> m_data;
  /** Matrix height. */
  int m_height;
  /** Matrix width per GPU. */
  int m_width_per_gpu;
  /** Matrix leading dimension. */
  int m_leading_dim;
  /** Whether the matrix is responsible for managing its memory. */
  bool m_is_view;
  /** Whether the matrix can modify its entries. */
  bool m_is_locked;

#endif
};

/** cuDNN manager class */
class cudnn_manager {
#ifdef LBANN_HAS_CUDNN

 public:
  /** Constructor
   *  @param _comm         Pointer to LBANN communicator
   *  @param max_num_gpus  Maximum Number of available GPUs. If
   *                       negative, then use all available GPUs.
   */
  cudnn_manager(lbann::lbann_comm *_comm, int max_num_gpus = -1, bool nccl_used = false);

  /** Destructor */
  ~cudnn_manager();

  /** Get number of GPUs assigned to current process. */
  int get_num_gpus() const;
  /** Get number of visible GPUs on current node. */
  int get_num_visible_gpus() const;
  /** Get GPUs. */
  std::vector<int>& get_gpus();
  /** Get GPUs (const). */
  const std::vector<int>& get_gpus() const;
  /** Get ith GPU. */
  int get_gpu(int i) const;
  /** Get CUDA streams. */
  std::vector<cudaStream_t>& get_streams();
  /** Get CUDA streams (const). */
  const std::vector<cudaStream_t>& get_streams() const;
  /** Get ith CUDA stream. */
  cudaStream_t& get_stream(int i);
  /** Get ith CUDA stream (const). */
  const cudaStream_t& get_stream(int i) const;
  /** Get cuDNN handles. */
  std::vector<cudnnHandle_t>& get_handles();
  /** Get cuDNN handles (const). */
  const std::vector<cudnnHandle_t>& get_handles() const;
  /** Get ith cuDNN handle. */
  cudnnHandle_t& get_handle(int i);
  /** Get ith cuDNN handle (const). */
  const cudnnHandle_t& get_handle(int i) const;
  /** Get CUBLAS handles. */
  std::vector<cublasHandle_t>& get_cublas_handles();
  /** Get CUBLAS handles (const). */
  const std::vector<cublasHandle_t>& get_cublas_handles() const;
  /** Get ith CUBLAS handle. */
  cublasHandle_t& get_cublas_handle(int i);
  /** Get ith CUBLAS handle (const). */
  const cublasHandle_t& get_cublas_handle(int i) const;
  /** Get GPU work spaces. */
  std::vector<void*> get_work_spaces();
  /** Get ith GPU work space. */
  void *get_work_space(int i);
  /** Get a lower bound on GPU work space sizes (in bytes). */
  size_t get_minimum_work_space_size();
  /** Get GPU work space sizes (in bytes). */
  std::vector<size_t> get_work_space_sizes();
  /** Get ith GPU work space size (in bytes). */
  size_t get_work_space_size(int i);
  /** Set ith GPU work space to occupy all available GPU memory. */
  void set_maximum_work_space_size(int i);
  /** Free ith GPU work space. */
  void free_work_space(int i);
  /** Free all GPU work spaces. */
  void free_work_spaces();

  /** Allocate memory on GPUs. */
  void allocate_on_gpus(std::vector<DataType*>& gpu_data,
                        int height,
                        int width_per_gpu);
  /** Deallocate memory on GPUs. */
  void deallocate_on_gpus(std::vector<DataType*>& gpu_data);

  /** Zero out memory on ith GPU. */
  void clear_on_gpu(int i,
                    DataType* gpu_data,
                    int height,
                    int width,
                    int leading_dim = 0);
  /** Copy data from CPU to ith GPU. */
  void copy_to_gpu(int i,
                   DataType* gpu_data,
                   const Mat& cpu_data,
                   int gpu_data_leading_dim = 0);
  /** Copy data from ith GPU to CPU. */
  void copy_from_gpu(int i,
                     Mat& cpu_data,
                     const DataType* gpu_data,
                     int gpu_data_leading_dim = 0);

  /** Zero out memory on GPUs. */
  void clear_on_gpus(std::vector<DataType*>& gpu_data,
                     int height,
                     int width_per_gpu,
                     int leading_dim = 0);
  /** Zero out memory corresponding to unused columns on GPUs. */
  void clear_unused_columns_on_gpus(std::vector<DataType*>& gpu_data,
                                    int height,
                                    int width,
                                    int width_per_gpu,
                                    int leading_dim = 0);

  /** Set memory on ith GPU to a constant. */
  void set_on_gpu(int i,
                  DataType* gpu_data,
                  DataType val,
                  int height,
                  int width = 1);
  /** Set memory on GPU to a constant. */
  void set_on_gpus(std::vector<DataType*>& gpu_data,
                   DataType val,
                   int height,
                   int width_per_gpu = 1);

  /** Copy data on GPUs. */
  void copy_on_gpus(std::vector<DataType*>& gpu_dst_data,
                    const std::vector<DataType*>& gpu_src_data,
                    int height,
                    int width_per_gpu,
                    int src_leading_dim = 0,
                    int dst_leading_dim = 0);
  /** Copy data from CPU to GPUs.
   *  Matrix columns are scattered amongst GPUs.
   */
  void scatter_to_gpus(std::vector<DataType*>& gpu_data,
                       const Mat& cpu_data,
                       int width_per_gpu,
                       int gpu_data_leading_dim = 0);
  /** Copy data from GPUs to CPU.
   *  Matrix columns are gathered from GPUs.
   */
  void gather_from_gpus(Mat& cpu_data,
                        const std::vector<DataType*>& gpu_data,
                        int width_per_gpu,
                        int gpu_data_leading_dim = 0);
  /** Copy data from CPU to GPUs.
   *  Data is duplicated across GPUs.
   */
  void broadcast_to_gpus(std::vector<DataType*>& gpu_data,
                         const Mat& cpu_data,
                         int gpu_data_leading_dim = 0);
  /** Copy data from GPUs to CPU and reduce. */
  void reduce_from_gpus(Mat& cpu_data,
                        const std::vector<DataType*>& gpu_data,
                        int gpu_data_leading_dim = 0);
  /** Allreduce within local GPUs. */
  void allreduce_on_gpus(std::vector<DataType*>& gpu_data,
                         El::Int height,
                         El::Int width);

  /** Allreduce within all GPUs in MPI communicator. */
  void global_allreduce_on_gpus(std::vector<DataType*>& gpu_data,
                                El::Int height,
                                El::Int width,
                                El::mpi::Comm comm);

  /** Allreduce within local GPUs using NCCL. */
  void global_allreduce_on_gpus_nccl(std::vector<DataType*>& gpu_data,
                 El::Int height,
                 El::Int width,
                 DataType scale = DataType(1));

  /** Synchronize the default stream. */
  void synchronize();

  /** Synchronize all streams. */
  void synchronize_all();

  /** Create copy of GPU data.
   *  The GPU memory allocated in this function must be deallocated
   *  elsewhere.
   */
  std::vector<DataType*> copy(const std::vector<DataType*>& gpu_data,
                              int height,
                              int width_per_gpu,
                              int leading_dim = 0);

  /** Pin matrix memory.
   *  Pinned memory accelerates memory transfers with GPU, but may
   *  degrade system performance. This function assumes that the
   *  matrix memory was previously allocated within Elemental.
   */
  void pin_matrix(AbsDistMat& mat);

  /** Unpin matrix memory.
   *  Pinned memory accelerates memory transfers with GPU, but may
   *  degrade system performance.
   */
  void unpin_matrix(AbsDistMat& mat);

  void check_error();

  bool is_nccl_used() { return m_nccl_used; }

 private:

  /** LBANN communicator. */
  lbann::lbann_comm *comm;

  /** Number of GPUs for current process. */
  int m_num_gpus;
  /** Number of visible GPUs. */
  int m_num_visible_gpus;

  /** List of GPUs. */
  std::vector<int> m_gpus;
  /** List of CUDA streams. */
  std::vector<cudaStream_t> m_streams;
  /** List of cuDNN handles. */
  std::vector<cudnnHandle_t> m_handles;
  /** List of cuDNN handles. */
  std::vector<cublasHandle_t> m_cublas_handles;

  /** List of GPU work spaces. */
  std::vector<void *> m_work_spaces;
  /** List of GPU work space sizes. */
  std::vector<size_t> m_work_space_sizes;

  bool m_nccl_used;
  void nccl_setup();
  void nccl_destroy();

  /** List of NCCL 2 related variables. */
#ifdef LBANN_HAS_NCCL2
  // One GPU per single thread of one MPI rank is assumed
  std::vector<ncclComm_t> m_nccl_comm;
  ncclDataType_t nccl_datatype();
#endif // #ifdef LBANN_HAS_NCCL2

#endif // #ifdef LBANN_HAS_CUDNN
};

#ifdef LBANN_HAS_CUDNN

/** Print cuDNN version information to standard output. */
void print_version();

/** Get cuDNN data type associated with C++ data type. */
cudnnDataType_t get_cudnn_data_type();

/** Set cuDNN tensor descriptor.
 *  num_samples is interpreted as the first tensor dimension, followed
 *  by the entries in sample_dims. desc is created or destroyed if
 *  needed.
 */
void set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                           int num_samples,
                           const std::vector<int>& sample_dims,
                           int sample_stride = 0);

/** Copy cuDNN tensor descriptor.
 *  dst is created or destroyed if needed.
 */
void copy_tensor_cudnn_desc(const cudnnTensorDescriptor_t& src,
                            cudnnTensorDescriptor_t& dst);

/** Copy cuDNN convolution kernel descriptor.
 *  dst is created or destroyed if needed.
 */
void copy_kernel_cudnn_desc(const cudnnFilterDescriptor_t& src,
                            cudnnFilterDescriptor_t& dst);

/** Copy cuDNN convolution descriptor.
 *  dst is created or destroyed if needed.
 */
void copy_convolution_cudnn_desc(const cudnnConvolutionDescriptor_t& src,
                                 cudnnConvolutionDescriptor_t& dst);

/** Copy cuDNN pooling descriptor.
 *  dst is created or destroyed if needed.
 */
void copy_pooling_cudnn_desc(const cudnnPoolingDescriptor_t& src,
                             cudnnPoolingDescriptor_t& dst);

/** Copy cuDNN activation descriptor.
 *  dst is created or destroyed if needed.
 */
void copy_activation_cudnn_desc(const cudnnActivationDescriptor_t& src,
                                cudnnActivationDescriptor_t& dst);

/** Copy cuDNN local response normalization descriptor.
 *  dst is created or destroyed if needed.
 */
void copy_lrn_cudnn_desc(const cudnnLRNDescriptor_t& src,
                         cudnnLRNDescriptor_t& dst);

#endif // #ifdef LBANN_HAS_CUDNN

}// namespace cudnn
}// namespace lbann

#endif // CUDNN_WRAPPER_HPP_INCLUDED
