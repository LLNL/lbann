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
#include "lbann/layers/layer.hpp"

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

// Forward declaration
class Layer;

namespace cudnn
{

#ifdef LBANN_HAS_CUDNN

/** Print cuDNN version information to standard output. */
void print_version();

/** Get cuDNN data type associated with DataType. */
cudnnDataType_t get_data_type();

/** Set cuDNN tensor descriptor. 
 *  desc is created if necessary.
 */
void set_tensor_desc(cudnnTensorDescriptor_t& desc,
                     std::vector<int> dims,
                     std::vector<int> strides = {});
/** Copy cuDNN tensor descriptor.
 *  dst is created or destroyed if needed.
 */
void copy_tensor_desc(const cudnnTensorDescriptor_t& src,
                      cudnnTensorDescriptor_t& dst);

/** Copy cuDNN activation descriptor.
 *  dst is created or destroyed if needed.
 */
void copy_activation_desc(const cudnnActivationDescriptor_t& src,
                          cudnnActivationDescriptor_t& dst);

#endif // LBANN_HAS_CUDNN

/** cuDNN manager. */
class cudnn_manager {
#ifdef LBANN_HAS_CUDNN

 public:
  cudnn_manager(size_t workspace_size = 1 << 30);
  ~cudnn_manager();

  /** Get cuDNN handle.
   *  This resets the active CUDA device and stream to the Hydrogen
   *  defaults.
   */
  cudnnHandle_t& get_handle();

  /** Get a recommended GPU workspace size (in bytes). */
  size_t get_workspace_size() const { return m_workspace_size; }
  /** Set a recommended GPU workspace size (in bytes). */
  void set_workspace_size(size_t size) { m_workspace_size = size; }

 private:

  /** cuDNN handle. */
  cudnnHandle_t m_handle;

  /** Recommendation for workspace size (in bytes). */
  size_t m_workspace_size;

#endif // #ifdef LBANN_HAS_CUDNN
};

#ifdef LBANN_HAS_CUDNN

/** Manager for a layer's cuDNN tensor descriptors. */
class layer_tensor_manager {
public:
  layer_tensor_manager(const Layer* l = nullptr);
  layer_tensor_manager(const layer_tensor_manager& other);
  layer_tensor_manager& operator=(const layer_tensor_manager& other);
  virtual ~layer_tensor_manager();

  /** Get the layer being managed. */
  const Layer* get_layer() const { return m_layer; }
  /** Set the layer being managed. */
  void set_layer(const Layer* l);

  /** Get cuDNN tensor descriptor for layer input. */
  virtual cudnnTensorDescriptor_t& get_prev_activations(int parent_index = 0) = 0;
  /** Get cuDNN tensor descriptor for layer output. */
  virtual cudnnTensorDescriptor_t& get_activations(int child_index = 0) = 0;
  /** Get cuDNN tensor descriptor for gradient w.r.t. layer output. */
  virtual cudnnTensorDescriptor_t& get_prev_error_signals(int child_index = 0) = 0;
  /** Get cuDNN tensor descriptor for gradient w.r.t. layer input. */
  virtual cudnnTensorDescriptor_t& get_error_signals(int parent_index = 0) = 0;

protected:

  /** Set number of tensor descriptors corresponding to layer inputs. */
  void set_num_parents(int num_parents);
  /** Set number of tensor descriptors corresponding to layer outputs. */
  void set_num_children(int num_children);

  /** Layer being managed. */
  const Layer* m_layer;
  /** cuDNN tensor descriptors for layer inputs. */
  std::vector<cudnnTensorDescriptor_t> m_prev_activations;
  /** cuDNN tensor descriptors for layer outputs. */
  std::vector<cudnnTensorDescriptor_t> m_activations;
  /** cuDNN tensor descriptors for gradients w.r.t. layer outputs. */
  std::vector<cudnnTensorDescriptor_t> m_prev_error_signals;
  /** cuDNN tensor descriptors for gradients w.r.t. layer inputs. */
  std::vector<cudnnTensorDescriptor_t> m_error_signals;

};

/** Manager for a data-parallel layer's cuDNN tensor descriptors. */
class data_parallel_layer_tensor_manager : public layer_tensor_manager {
public:
  data_parallel_layer_tensor_manager(const Layer* l = nullptr);
  data_parallel_layer_tensor_manager(
    const data_parallel_layer_tensor_manager& other) = default;
  data_parallel_layer_tensor_manager&
    operator=(const data_parallel_layer_tensor_manager& other) = default;
  ~data_parallel_layer_tensor_manager() = default;
  cudnnTensorDescriptor_t& get_prev_activations(int parent_index = 0) override;
  cudnnTensorDescriptor_t& get_activations(int child_index = 0) override;
  cudnnTensorDescriptor_t& get_prev_error_signals(int child_index = 0) override;
  cudnnTensorDescriptor_t& get_error_signals(int parent_index = 0) override;
};

/** Manager for an entry-wise layer's cuDNN tensor descriptors. */
class entrywise_layer_tensor_manager : public layer_tensor_manager {
public:
  entrywise_layer_tensor_manager(const Layer* l = nullptr);
  entrywise_layer_tensor_manager(
    const entrywise_layer_tensor_manager& other) = default;
  entrywise_layer_tensor_manager&
    operator=(const entrywise_layer_tensor_manager& other) = default;
  ~entrywise_layer_tensor_manager() = default;
  cudnnTensorDescriptor_t& get_prev_activations(int parent_index = 0) override;
  cudnnTensorDescriptor_t& get_activations(int child_index = 0) override;
  cudnnTensorDescriptor_t& get_prev_error_signals(int child_index = 0) override;
  cudnnTensorDescriptor_t& get_error_signals(int parent_index = 0) override;
};

#endif // #ifdef LBANN_HAS_CUDNN

}// namespace cudnn
}// namespace lbann

#endif // CUDNN_WRAPPER_HPP_INCLUDED
