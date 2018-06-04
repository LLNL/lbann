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
////////////////////////////////////////////////////////////////////////////////

#ifndef CUDNN_WRAPPER_HPP_INCLUDED
#define CUDNN_WRAPPER_HPP_INCLUDED

#include <vector>
#include "lbann/base.hpp"
#include "lbann/comm.hpp"

#ifdef LBANN_HAS_CUDNN
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#endif // #ifdef LBANN_HAS_CUDNN

// Error utility macros
#ifdef LBANN_HAS_CUDNN
#define FORCE_CHECK_CUDA(cuda_call)                                     \
  do {                                                                  \
    {                                                                   \
      /* Check for earlier asynchronous errors. */                      \
      cudaError_t status_FORCE_CHECK_CUDA = cudaDeviceSynchronize();    \
      if (status_FORCE_CHECK_CUDA == cudaSuccess)                       \
        status_FORCE_CHECK_CUDA = cudaGetLastError();                   \
      if (status_FORCE_CHECK_CUDA != cudaSuccess) {                     \
        cudaDeviceReset();                                              \
        LBANN_ERROR(std::string("Asynchronous CUDA error: ")            \
                    + cudaGetErrorString(status_FORCE_CHECK_CUDA));     \
      }                                                                 \
    }                                                                   \
    {                                                                   \
      /* Make CUDA call and check for errors. */                        \
      cudaError_t status_FORCE_CHECK_CUDA = (cuda_call);                \
      if (status_FORCE_CHECK_CUDA == cudaSuccess)                       \
        status_FORCE_CHECK_CUDA = cudaDeviceSynchronize();              \
      if (status_FORCE_CHECK_CUDA == cudaSuccess)                       \
        status_FORCE_CHECK_CUDA = cudaGetLastError();                   \
      if (status_FORCE_CHECK_CUDA != cudaSuccess) {                     \
        cudaDeviceReset();                                              \
        LBANN_ERROR(std::string("CUDA error: ")                         \
                    + cudaGetErrorString(status_FORCE_CHECK_CUDA));     \
      }                                                                 \
    }                                                                   \
  } while (0)
#define FORCE_CHECK_CUDNN(cudnn_call)                                   \
  do {                                                                  \
    /* Check for earlier asynchronous errors. */                        \
    FORCE_CHECK_CUDA(cudaSuccess);                                      \
    {                                                                   \
      /* Make cuDNN call and check for errors. */                       \
      const cudnnStatus_t status_FORCE_CHECK_CUDNN = (cudnn_call);      \
      if (status_FORCE_CHECK_CUDNN != CUDNN_STATUS_SUCCESS) {           \
        cudaDeviceReset();                                              \
        LBANN_ERROR(std::string("cuDNN error: ")                        \
                    + cudnnGetErrorString(status_FORCE_CHECK_CUDNN));   \
      }                                                                 \
    }                                                                   \
    {                                                                   \
      /* Check for CUDA errors. */                                      \
      cudaError_t status_FORCE_CHECK_CUDNN = cudaDeviceSynchronize();   \
      if (status_FORCE_CHECK_CUDNN == cudaSuccess)                      \
        status_FORCE_CHECK_CUDNN = cudaGetLastError();                  \
      if (status_FORCE_CHECK_CUDNN != cudaSuccess) {                    \
        cudaDeviceReset();                                              \
        LBANN_ERROR(std::string("CUDA error: ")                         \
                    + cudaGetErrorString(status_FORCE_CHECK_CUDNN));    \
      }                                                                 \
    }                                                                   \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUDA(cuda_call)   FORCE_CHECK_CUDA(cuda_call);
#define CHECK_CUDNN(cudnn_call) FORCE_CHECK_CUDNN(cudnn_call);
#else
#define CHECK_CUDA(cuda_call)   (cuda_call)
#define CHECK_CUDNN(cudnn_call) (cudnn_call)
#endif // #ifdef LBANN_DEBUG
#endif // #ifdef LBANN_HAS_CUDNN

namespace lbann
{
namespace cudnn
{

// Forward declaration
class cudnn_manager;

/** cuDNN manager class */
class cudnn_manager {
#ifdef LBANN_HAS_CUDNN

 public:
  /** Constructor
   *  @param _comm           Pointer to LBANN communicator
   *  @param workspace_size  Recommendation for workspace size.
   */
  cudnn_manager(lbann::lbann_comm *_comm,
                size_t workspace_size = 1 << 9);

  /** Destructor */
  ~cudnn_manager();

  /** Get cuDNN handle.
   *  This sets the CUDA device and stream to those used by Hydrogen.
   */
  cudnnHandle_t& get_handle();

  /** Get a recommended GPU workspace size (in bytes). */
  size_t get_workspace_size() const { return m_workspace_size; }
  /** Set a recommended GPU workspace size (in bytes). */
  void set_workspace_size(size_t size) { m_workspace_size = size; }

  /** Synchronize the primary Hydrogen CUDA stream. */
  void synchronize();
  /** Synchronize GPU. */
  void synchronize_all();

  /** Check for errors from asynchronous CUDA kernels. */
  void check_error();

 private:

  /** LBANN communicator. */
  lbann::lbann_comm *comm;

  /** cuDNN handle. */
  cudnnHandle_t m_handle;

  /** Recommendation for workspace size (in bytes). */
  size_t m_workspace_size;

#endif // #ifdef LBANN_HAS_CUDNN
};

#ifdef LBANN_HAS_CUDNN

/** Print cuDNN version information to standard output. */
void print_version();

/** Get cuDNN data type associated with C++ data type. */
cudnnDataType_t get_cudnn_data_type();

/** Set cuDNN tensor descriptor.
 *  num_samples is interpreted as the first tensor dimension, followed
 *  by the entries in sample_dims. desc is created if needed.
 */
void set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                           int num_samples,
                           const std::vector<int>& sample_dims,
                           int sample_stride = 0);

/** Set cuDNN tensor descriptor for a matrix.
 *  desc is created if needed.
 */
void set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                           int height,
                           int width = 1,
                           int leading_dim = 0);

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
