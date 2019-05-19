////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_UTILS_CUDNN_HPP
#define LBANN_UTILS_CUDNN_HPP

#include "lbann/base.hpp"
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/layers/layer.hpp"
#include <vector>

#ifdef LBANN_HAS_CUDNN

#include <cudnn.h>

// Error utility macros
#define CHECK_CUDNN_NODEBUG(cudnn_call)                         \
  do {                                                          \
    const cudnnStatus_t status_CHECK_CUDNN = (cudnn_call);      \
    if (status_CHECK_CUDNN != CUDNN_STATUS_SUCCESS) {           \
      cudaDeviceReset();                                        \
      LBANN_ERROR(std::string("cuDNN error (")                  \
                  + cudnnGetErrorString(status_CHECK_CUDNN)     \
                  + std::string(")"));                          \
    }                                                           \
  } while (0)
#define CHECK_CUDNN_DEBUG(cudnn_call)                           \
  do {                                                          \
    LBANN_CUDA_CHECK_LAST_ERROR(true);                          \
    CHECK_CUDNN_NODEBUG(cudnn_call);                            \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUDNN(cudnn_call) CHECK_CUDNN_DEBUG(cudnn_call)
#else
#define CHECK_CUDNN(cudnn_call) CHECK_CUDNN_NODEBUG(cudnn_call)
#endif // #ifdef LBANN_DEBUG

#define CHECK_CUDNN_DTOR(cudnn_call)            \
  try {                                         \
    CHECK_CUDNN(cudnn_call);                                            \
  }                                                                     \
  catch (std::exception const& e) {                                     \
    std::cerr << "Caught exception:\n\n    what(): "                    \
              << e.what() << "\n\nCalling std::terminate() now."        \
              <<  std::endl;                                            \
    std::terminate();                                                   \
  }                                                                     \
  catch (...) {                                                         \
    std::cerr << "Caught something that isn't an std::exception.\n\n"   \
              << "Calling std::terminate() now." << std::endl;          \
    std::terminate();                                                   \
  }


namespace lbann {

// Forward declaration
class Layer;

namespace cudnn {

////////////////////////////////////////////////////////////
// Global cuDNN objects
////////////////////////////////////////////////////////////

/** Initialize global cuDNN objects. */
void initialize();
/** Destroy global cuDNN objects. */
void destroy();
/** Get cuDNN handle.
 *  This resets the active CUDA device and stream to the Hydrogen
 *  defaults. The cuDNN handle is initialized if needed.
 */
cudnnHandle_t& get_handle();

////////////////////////////////////////////////////////////
// Helper functions for cuDNN types
////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////
// cuDNN tensor managers
////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////
// cuDNN algorithm selection
////////////////////////////////////////////////////////////

/**
 * Select a forward convolution algorithm.
 *
 * If autotuning, memory for cuDNN algorithm runs is needed and should be
 * provided via the pointer arguments.
 *
 * @param autotune True to attempt all cuDNN algorithms and select the fastest.
 * @param deterministic True to require deterministic algorithms.
 */
cudnnConvolutionFwdAlgo_t get_fwd_algorithm(
  bool autotune,
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const void* input,
  const cudnnFilterDescriptor_t& kernel_desc,
  const void* kernel,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& output_desc,
  void* output,
  size_t ws_size,
  void* ws);

/** Select a backward data convolution algorithm.
 *
 * If autotuning, memory for cuDNN algorithm runs is needed and should be
 * provided via the pointer arguments.
 *
 * @param autotune True to attempt all cuDNN algorithms and select the fastest.
 * @param deterministic True to require deterministic algorithms.
 */
cudnnConvolutionBwdDataAlgo_t get_bwd_data_algorithm(
  bool autotune,
  bool deterministic,
  const cudnnFilterDescriptor_t& kernel_desc,
  const void* kernel,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& error_signal_desc,
  void* error_signal,
  size_t ws_size,
  void* ws);

/** Select a backward filter convolution algorithm.
 *
 * If autotuning, memory for cuDNN algorithm runs is needed and should be
 * provided via the pointer arguments.
 *
 * @param autotune True to attempt all cuDNN algorithms and select the fastest.
 * @param deterministic True to require deterministic algorithms.
 */
cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algorithm(
  bool autotune,
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const void* input,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnFilterDescriptor_t& kernel_gradient_desc,
  void* kernel_gradient,
  size_t ws_size,
  void* ws);

} // namespace cudnn
} // namespace lbann

#endif // LBANN_HAS_CUDNN
#endif // LBANN_UTILS_CUDNN_HPP
