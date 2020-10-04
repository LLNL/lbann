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
#include "lbann/layers/data_type_layer.hpp"
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

template <typename T>
struct ScalingParameterT
{
  using type = T;
};

template <typename T>
using ScalingParamType = typename ScalingParameterT<T>::type;

#ifdef LBANN_HAS_GPU_FP16
template <>
struct ScalingParameterT<fp16>
{
  using type = float;
};
#endif // LBANN_USE_GPU_FP16

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
template <typename TensorDataType>
cudnnDataType_t get_data_type();

/** Set cuDNN tensor descriptor.
 *  desc is created if necessary.
 */
template <typename TensorDataType>
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
// Wrapper classes for cuDNN types
////////////////////////////////////////////////////////////

/** @brief Wrapper around @c cudnnTensorDescriptor_t */
class TensorDescriptor {

public:

  TensorDescriptor(cudnnTensorDescriptor_t desc=nullptr);
  template <typename... ArgTs>
  TensorDescriptor(ArgTs... args) {
    set(args...);
  }

  ~TensorDescriptor();

  // Copy-and-swap idiom
  TensorDescriptor(const TensorDescriptor&);
  TensorDescriptor(TensorDescriptor&&);
  TensorDescriptor& operator=(TensorDescriptor);
  friend void swap(TensorDescriptor& first, TensorDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(cudnnTensorDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  cudnnTensorDescriptor_t release();
  /** @brief Return cuDNN object without releasing ownership */
  cudnnTensorDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator cudnnTensorDescriptor_t() const noexcept;

  /** @brief Create cuDNN object
   *
   *  Does nothing if already created.
   */
  void create();
  /** @brief Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    cudnnDataType_t data_type,
    const std::vector<int>& dims,
    std::vector<int> strides = {});
  /** @brief Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  template <typename... IntTs>
  void set(
    cudnnDataType_t data_type,
    IntTs... dims) {
    set(data_type, {static_cast<int>(dims)...});
  }

private:

  cudnnTensorDescriptor_t desc_{nullptr};

};

/** Wrapper around @c cudnnFilterDescriptor_t */
class FilterDescriptor {

public:

  FilterDescriptor(cudnnFilterDescriptor_t desc=nullptr);
  template <typename... ArgTs>
  FilterDescriptor(ArgTs... args) {
    set(args...);
  }

  ~FilterDescriptor();

  // Copy-and-swap idiom
  FilterDescriptor(const FilterDescriptor&);
  FilterDescriptor(FilterDescriptor&&);
  FilterDescriptor& operator=(FilterDescriptor);
  friend void swap(FilterDescriptor& first, FilterDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(cudnnFilterDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  cudnnFilterDescriptor_t release();
  /** @brief Return cuDNN object without releasing ownership */
  cudnnFilterDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator cudnnFilterDescriptor_t() const noexcept;

  /** Create cuDNN object
   *
   *  Does nothing if already created.
   */
  void create();
  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    cudnnDataType_t data_type,
    cudnnTensorFormat_t format,
    const std::vector<int>& dims);
  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  template <typename... IntTs>
  void set(
    cudnnDataType_t data_type,
    cudnnTensorFormat_t format,
    IntTs... dims) {
    set(data_type, format, {static_cast<int>(dims)...});
  }

private:

  cudnnFilterDescriptor_t desc_{nullptr};

};

/** Wrapper around @c cudnnDropoutDescriptor_t */
class DropoutDescriptor {

public:

  DropoutDescriptor(cudnnDropoutDescriptor_t desc=nullptr);
  template <typename... ArgTs>
  DropoutDescriptor(ArgTs... args) {
    set(args...);
  }

  ~DropoutDescriptor();

  // Copy-and-swap idiom
  DropoutDescriptor(const DropoutDescriptor&);
  DropoutDescriptor(DropoutDescriptor&&);
  DropoutDescriptor& operator=(DropoutDescriptor);
  friend void swap(DropoutDescriptor& first, DropoutDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(cudnnDropoutDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  cudnnDropoutDescriptor_t release();
  /** @brief Return cuDNN object without releasing ownership */
  cudnnDropoutDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator cudnnDropoutDescriptor_t() const noexcept;

  /** Create cuDNN object
   *
   *  Does nothing if already created.
   */
  void create();
  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    float dropout,
    void* states,
    size_t states_size,
    unsigned long long seed);

private:

  cudnnDropoutDescriptor_t desc_{nullptr};

};

/** Wrapper around @c cudnnRNNDescriptor_t */
class RNNDescriptor {

public:

  RNNDescriptor(cudnnRNNDescriptor_t desc=nullptr);
  template <typename... ArgTs>
  RNNDescriptor(ArgTs... args) {
    set(args...);
  }

  RNNDescriptor(const RNNDescriptor&) = delete;
  ~RNNDescriptor();

  // Copy-and-swap idiom
  RNNDescriptor(RNNDescriptor&&);
  RNNDescriptor& operator=(RNNDescriptor);
  friend void swap(RNNDescriptor& first, RNNDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(cudnnRNNDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  cudnnRNNDescriptor_t release();
  /** @brief Return cuDNN object without releasing ownership */
  cudnnRNNDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator cudnnRNNDescriptor_t() const noexcept;

  /** Create cuDNN object
   *
   *  Does nothing if already created.
   */
  void create();
  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    cudnnRNNAlgo_t algorithm,
    cudnnRNNMode_t cell_mode,
    cudnnRNNBiasMode_t bias_mode,
    cudnnDirectionMode_t direction_mode,
    cudnnRNNInputMode_t input_mode,
    cudnnDataType_t data_type,
    cudnnDataType_t math_precision,
    cudnnMathType_t math_type,
    size_t input_size,
    size_t hidden_size,
    size_t proj_size,
    size_t num_layers,
    cudnnDropoutDescriptor_t dropout_desc,
    uint32_t aux_flags);

private:

  cudnnRNNDescriptor_t desc_{nullptr};

};

/** Wrapper around @c cudnnRNNDataDescriptor_t */
class RNNDataDescriptor {

public:

  RNNDataDescriptor(cudnnRNNDataDescriptor_t desc=nullptr);

  ~RNNDataDescriptor();

  // Copy-and-swap idiom
  RNNDataDescriptor(const RNNDataDescriptor&) = delete; /// @todo Implement
  RNNDataDescriptor(RNNDataDescriptor&&);
  RNNDataDescriptor& operator=(RNNDataDescriptor);
  friend void swap(RNNDataDescriptor& first, RNNDataDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(cudnnRNNDataDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  cudnnRNNDataDescriptor_t release();
  /** @brief Return cuDNN object without releasing ownership */
  cudnnRNNDataDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator cudnnRNNDataDescriptor_t() const noexcept;

  /** Create cuDNN object
   *
   *  Does nothing if already created.
   */
  void create();
  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    cudnnDataType_t data_type,
    cudnnRNNDataLayout_t layout,
    size_t max_seq_length,
    size_t batch_size,
    size_t vector_size,
    const int seq_length_array[],
    void* padding_fill);

private:

  cudnnRNNDataDescriptor_t desc_{nullptr};

};

////////////////////////////////////////////////////////////
// cuDNN tensor managers
////////////////////////////////////////////////////////////

/** Manager for a layer's cuDNN tensor descriptors. */
template <typename TensorDataType>
class layer_tensor_manager {
public:
  using LayerType = data_type_layer<TensorDataType>;
public:
  layer_tensor_manager(const LayerType* l = nullptr);
  layer_tensor_manager(const layer_tensor_manager& other);
  layer_tensor_manager& operator=(const layer_tensor_manager& other);
  virtual ~layer_tensor_manager();

  /** Get the layer being managed. */
  const LayerType* get_layer() const { return m_layer; }
  /** Set the layer being managed. */
  void set_layer(const LayerType* l);

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
  const LayerType* m_layer;
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
template <typename TensorDataType>
class data_parallel_layer_tensor_manager : public layer_tensor_manager<TensorDataType> {
public:
  using LayerType = data_type_layer<TensorDataType>;
public:
  data_parallel_layer_tensor_manager(const LayerType* l = nullptr);
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
template <typename TensorDataType>
class entrywise_layer_tensor_manager : public layer_tensor_manager<TensorDataType> {
public:
  using LayerType = data_type_layer<TensorDataType>;
public:
  entrywise_layer_tensor_manager(const LayerType* l = nullptr);
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

/** @brief Set the default to use tensor core operations, allowing
 *         FP32->FP16 conversions.
 */
void default_to_tensor_ops() noexcept;

/** @brief Get the default math type.
 *
 *  Will query the command-line args.
 */
cudnnMathType_t get_default_convolution_math_type() noexcept;

} // namespace cudnn
} // namespace lbann

#endif // LBANN_HAS_CUDNN
#endif // LBANN_UTILS_CUDNN_HPP
