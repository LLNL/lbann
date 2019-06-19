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

#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/number_theory.hpp"

#include "El.hpp"
#include <iostream>
#include <map>
#include <unordered_map>
#include <tuple>

#ifdef LBANN_HAS_CUDNN

namespace lbann {
namespace cudnn {

////////////////////////////////////////////////////////////
// Global cuDNN objects
////////////////////////////////////////////////////////////

namespace {

/** Wrapper for cuDNN handle. */
struct handle_wrapper {
  cudnnHandle_t handle;
  handle_wrapper() : handle(nullptr) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    if (handle == nullptr) { CHECK_CUDNN(cudnnCreate(&handle)); }
    if (handle == nullptr) { LBANN_ERROR("failed to create cuDNN handle"); }
    CHECK_CUDNN(cudnnSetStream(handle, El::GPUManager::Stream()));
  }
  handle_wrapper(const handle_wrapper&) = delete;
  handle_wrapper& operator=(const handle_wrapper&) = delete;
  ~handle_wrapper() {
    if (handle != nullptr) { cudnnDestroy(handle); }
  }
};

/** Global instance of cuDNN handle. */
std::unique_ptr<handle_wrapper> handle_instance;

} // namespace

void initialize() {
  handle_instance.reset(new handle_wrapper());
}

void destroy() {
  handle_instance.reset();
}

cudnnHandle_t& get_handle() {
  if (!handle_instance) { initialize(); }
  CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
  CHECK_CUDNN(cudnnSetStream(handle_instance->handle,
                             El::GPUManager::Stream()));
  return handle_instance->handle;
}

////////////////////////////////////////////////////////////
// Helper functions for cuDNN types
////////////////////////////////////////////////////////////

cudnnDataType_t get_data_type() {
  switch (sizeof(DataType)) {
  case 2: return CUDNN_DATA_HALF;
  case 4: return CUDNN_DATA_FLOAT;
  case 8: return CUDNN_DATA_DOUBLE;
  default: LBANN_ERROR("invalid data type for cuDNN");
  }
  return CUDNN_DATA_FLOAT;
}

void set_tensor_desc(cudnnTensorDescriptor_t& desc,
                     std::vector<int> dims,
                     std::vector<int> strides) {
  std::stringstream err;
  if (dims.empty()) {
    LBANN_ERROR("attempted to set cuDNN tensor descriptor with no dimensions");
  }

  // Assume data is contiguous if no strides are provided
  if (strides.empty()) {
    strides.resize(dims.size());
    strides.back() = 1;
    for(int i = strides.size() - 1; i > 0; --i) {
        strides[i-1] = strides[i] * dims[i];
    }
  }

#ifdef LBANN_DEBUG
  // Check that dimensions and strides are valid
  if (strides.size() != dims.size()) {
    err << "attempted to set cuDNN tensor descriptor "
        << "with invalid strides (";
    for (size_t i = 0; i < strides.size(); ++i) {
      err << (i == 0 ? "" : ",") << strides[i];
    }
    err << ") for dimensions (";
    for (size_t i = 0; i < dims.size(); ++i) {
      err << (i == 0 ? "" : "x") << dims[i];
    }
    err << ")";
    LBANN_ERROR(err.str());
  }
  for (size_t j = 0; j < dims.size(); ++j) {
    if (dims[j] <= 0) {
      err << "attempted to set cuDNN tensor descriptor "
          << "with invalid dimensions (";
      for (size_t i = 0; i < dims.size(); ++i) {
        err << (i == 0 ? "" : "x") << dims[i];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }
    if (j > 0 && strides[j-1] < dims[j] * strides[j]) {
      err << "attempted to set cuDNN tensor descriptor "
          << "with invalid strides (";
      for (size_t i = 0; i < strides.size(); ++i) {
        err << (i == 0 ? "" : ",") << strides[i];
      }
      err << ") for dimensions (";
      for (size_t i = 0; i < dims.size(); ++i) {
        err << (i == 0 ? "" : "x") << dims[i];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }
  }
#endif // LBANN_DEBUG

  // Set cuDNN tensor descriptor
  // Note: cuDNN tensors should have at least 4 dimensions
  /// @todo Think about 1D convolution
  while (dims.size() < 4) {
    dims.push_back(1);
    strides.push_back(1);
  }
  if (desc == nullptr) {
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
  }
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                         get_data_type(),
                                         dims.size(),
                                         dims.data(),
                                         strides.data()));

}

void copy_tensor_desc(const cudnnTensorDescriptor_t& src,
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

void copy_activation_desc(const cudnnActivationDescriptor_t& src,
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

////////////////////////////////////////////////////////////
// Base cuDNN tensor manager
////////////////////////////////////////////////////////////

layer_tensor_manager::layer_tensor_manager(const Layer* l)
  : m_layer(nullptr) {
  set_layer(l);
}

layer_tensor_manager::layer_tensor_manager(const layer_tensor_manager& other)
  : m_layer(other.m_layer),
    m_prev_activations(other.m_prev_activations.size(), nullptr),
    m_activations(other.m_activations.size(), nullptr),
    m_prev_error_signals(other.m_prev_error_signals.size(), nullptr),
    m_error_signals(other.m_error_signals.size(), nullptr) {
  for (size_t i = 0; i < m_prev_activations.size(); ++i) {
    copy_tensor_desc(other.m_prev_activations[i], m_prev_activations[i]);
  }
  for (size_t i = 0; i < m_activations.size(); ++i) {
    copy_tensor_desc(other.m_activations[i], m_activations[i]);
  }
  for (size_t i = 0; i < m_prev_error_signals.size(); ++i) {
    copy_tensor_desc(other.m_prev_error_signals[i], m_prev_error_signals[i]);
  }
  for (size_t i = 0; i < m_error_signals.size(); ++i) {
    copy_tensor_desc(other.m_error_signals[i], m_error_signals[i]);
  }
}

layer_tensor_manager& layer_tensor_manager::operator=(const layer_tensor_manager& other) {

  // Set layer being managed
  m_layer = other.m_layer;

  // Destroy tensor descriptors
  set_num_parents(0);
  set_num_children(0);

  // Create copies of tensor descriptors
  m_prev_activations.resize(other.m_prev_activations.size(), nullptr);
  m_activations.resize(other.m_activations.size(), nullptr);
  m_prev_error_signals.resize(other.m_prev_error_signals.size(), nullptr);
  m_error_signals.resize(other.m_error_signals.size(), nullptr);
  for (size_t i = 0; i < m_prev_activations.size(); ++i) {
    copy_tensor_desc(other.m_prev_activations[i], m_prev_activations[i]);
  }
  for (size_t i = 0; i < m_activations.size(); ++i) {
    copy_tensor_desc(other.m_activations[i], m_activations[i]);
  }
  for (size_t i = 0; i < m_prev_error_signals.size(); ++i) {
    copy_tensor_desc(other.m_prev_error_signals[i], m_prev_error_signals[i]);
  }
  for (size_t i = 0; i < m_error_signals.size(); ++i) {
    copy_tensor_desc(other.m_error_signals[i], m_error_signals[i]);
  }

  return *this;
}

layer_tensor_manager::~layer_tensor_manager() {
  for (auto&& desc : m_prev_activations) {
    if (desc != nullptr) { cudnnDestroyTensorDescriptor(desc); }
  }
  for (auto&& desc : m_activations) {
    if (desc != nullptr) { cudnnDestroyTensorDescriptor(desc); }
  }
  for (auto&& desc : m_prev_error_signals) {
    if (desc != nullptr) { cudnnDestroyTensorDescriptor(desc); }
  }
  for (auto&& desc : m_error_signals) {
    if (desc != nullptr) { cudnnDestroyTensorDescriptor(desc); }
  }
}

void layer_tensor_manager::set_layer(const Layer* new_layer) {
  m_layer = new_layer;
  set_num_parents(m_layer == nullptr ? 0 : m_layer->get_num_parents());
  set_num_children(m_layer == nullptr ? 0 : m_layer->get_num_children());
}

void layer_tensor_manager::set_num_parents(int num_parents) {
#ifdef LBANN_DEBUG
  if (num_parents < 0) { LBANN_ERROR("negative number of parents"); }
#endif // LBANN_DEBUG
  for (size_t i = num_parents; i < m_prev_activations.size(); ++i) {
    if (m_prev_activations[i] != nullptr) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_prev_activations[i]));
      m_prev_activations[i] = nullptr;
    }
  }
  for (size_t i = num_parents; i < m_error_signals.size(); ++i) {
    if (m_error_signals[i] != nullptr) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_error_signals[i]));
      m_error_signals[i] = nullptr;
    }
  }
  m_prev_activations.resize(num_parents, nullptr);
  m_error_signals.resize(num_parents, nullptr);
}

void layer_tensor_manager::set_num_children(int num_children) {
#ifdef LBANN_DEBUG
  if (num_children < 0) { LBANN_ERROR("negative number of children"); }
#endif // LBANN_DEBUG
  for (size_t i = num_children; i < m_activations.size(); ++i) {
    if (m_activations[i] != nullptr) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_activations[i]));
      m_activations[i] = nullptr;
    }
  }
  for (size_t i = num_children; i < m_prev_error_signals.size(); ++i) {
    if (m_prev_error_signals[i] != nullptr) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_prev_error_signals[i]));
      m_prev_error_signals[i] = nullptr;
    }
  }
  m_activations.resize(num_children, nullptr);
  m_prev_error_signals.resize(num_children, nullptr);
}

////////////////////////////////////////////////////////////
// Data-parallel cuDNN tensor manager
////////////////////////////////////////////////////////////

data_parallel_layer_tensor_manager
::data_parallel_layer_tensor_manager(const Layer* l)
  : layer_tensor_manager(l) {}

namespace {

/** Set a cuDNN tensor descriptor for a data-parallel data layout.
 */
void set_data_parallel_tensor_desc(cudnnTensorDescriptor_t& desc,
                                   std::vector<int> dims,
                                   const AbsMat& local_data) {
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("attempted to setup cuDNN tensor with non-GPU data");
  }
#endif // LBANN_DEBUG
  if (local_data.Height() > 0 && local_data.Width() > 0) {
    std::vector<int> strides(dims.size(), 1);
    for(int i = strides.size() - 1; i > 0; --i) {
      strides[i-1] = strides[i] * dims[i];
    }
    dims.insert(dims.begin(), local_data.Width());
    strides.insert(strides.begin(), local_data.LDim());
    set_tensor_desc(desc, dims, strides);
  }
}

} // namespace

cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager::get_prev_activations(int parent_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = m_layer->get_local_prev_activations(parent_index);
  const auto& dims = m_layer->get_input_dims(parent_index);
  set_num_parents(m_layer->get_num_parents());
  auto& desc = m_prev_activations[parent_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager::get_activations(int child_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = m_layer->get_local_activations(child_index);
  const auto& dims = m_layer->get_output_dims(child_index);
  set_num_children(m_layer->get_num_children());
  auto& desc = m_activations[child_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager::get_prev_error_signals(int child_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = m_layer->get_local_prev_error_signals(child_index);
  const auto& dims = m_layer->get_output_dims(child_index);
  set_num_children(m_layer->get_num_children());
  auto& desc = m_prev_error_signals[child_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager::get_error_signals(int parent_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = m_layer->get_local_error_signals(parent_index);
  const auto& dims = m_layer->get_input_dims(parent_index);
  set_num_parents(m_layer->get_num_parents());
  auto& desc = m_error_signals[parent_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

////////////////////////////////////////////////////////////
// Entry-wise cuDNN tensor manager
////////////////////////////////////////////////////////////

entrywise_layer_tensor_manager
::entrywise_layer_tensor_manager(const Layer* l)
  : layer_tensor_manager(l) {}

namespace {

/** Set a cuDNN tensor descriptor for an entrywise tensor operation.
 *  Given local data in a (height x width) matrix, the tensor is
 *  initialized with dimensions (width, a, b, c), where
 *  a*b*c=height. This is because cuDNN is optimized for 4D tensors
 *  and gets poor performance with 1D tensors and 2D tensors.
 */
void set_entrywise_tensor_desc(cudnnTensorDescriptor_t& desc,
                               const AbsMat& local_data) {
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("attempted to setup cuDNN tensor with non-GPU data");
  }
#endif // LBANN_DEBUG
  const int height = local_data.Height();
  const int width = local_data.Width();
  const int ldim = local_data.LDim();
  if (height > 0 && width > 0) {

    // Factorize height into three factors
    // Note: factorization is memoized
    static std::unordered_map<int,std::vector<int>> cache;
    auto& factors = cache[height];
    if (factors.empty()) {
      factors = number_theory::balanced_factors(height, 3);
    }

    // Set cuDNN tensor descriptor with 4D tensor
    set_tensor_desc(desc,
                    {width, factors[2], factors[1], factors[0]},
                    {ldim, factors[1]*factors[0], factors[0], 1});

  }
}

} // namespace

cudnnTensorDescriptor_t& entrywise_layer_tensor_manager::get_prev_activations(int parent_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = m_layer->get_local_prev_activations(parent_index);
  set_num_parents(m_layer->get_num_parents());
  auto& desc = m_prev_activations[parent_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

cudnnTensorDescriptor_t& entrywise_layer_tensor_manager::get_activations(int child_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = m_layer->get_local_activations(child_index);
  set_num_children(m_layer->get_num_children());
  auto& desc = m_activations[child_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

cudnnTensorDescriptor_t& entrywise_layer_tensor_manager::get_prev_error_signals(int child_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = m_layer->get_local_prev_error_signals(child_index);
  set_num_children(m_layer->get_num_children());
  auto& desc = m_prev_error_signals[child_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

cudnnTensorDescriptor_t& entrywise_layer_tensor_manager::get_error_signals(int parent_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = m_layer->get_local_error_signals(parent_index);
  set_num_parents(m_layer->get_num_parents());
  auto& desc = m_error_signals[parent_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

////////////////////////////////////////////////////////////
// cuDNN algorithm selection
////////////////////////////////////////////////////////////

namespace {

// Non-deterministic algorithms.
std::vector<cudnnConvolutionFwdAlgo_t> nondet_fwd_algos = {};
std::vector<cudnnConvolutionBwdDataAlgo_t> nondet_bwd_data_algos = {
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
};
std::vector<cudnnConvolutionBwdFilterAlgo_t> nondet_bwd_filter_algos = {
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3
};

template <typename AlgoType, typename PerfType>
AlgoType find_best_heuristic_algorithm(
  const std::vector<PerfType>& perf_results,
  const std::vector<AlgoType>& nondeterministic_algos,
  bool deterministic,
  size_t max_ws_size) {
  std::vector<AlgoType> algos;
  for (const auto& p : perf_results) {
    if (p.status != CUDNN_STATUS_SUCCESS) {
      continue;
    }
    if (deterministic &&
        std::find(nondeterministic_algos.begin(), nondeterministic_algos.end(),
                  p.algo) != nondeterministic_algos.end()) {
      continue;
    }
    if (p.memory > max_ws_size) {
      continue;
    }
    algos.push_back(p.algo);
  }
  if (algos.empty()) {
    LBANN_ERROR("No valid convolution algorithms.");
  }
  return algos[0];
}

template <typename AlgoType, typename PerfType>
AlgoType find_best_algorithm(
  const std::vector<PerfType>& perf_results,
  const std::vector<AlgoType>& nondeterministic_algos,
  bool deterministic,
  size_t max_ws_size) {
  std::map<AlgoType, float> time_map;
  for (const auto& p : perf_results) {
    if (p.status != CUDNN_STATUS_SUCCESS) {
      // If an algorithm fails, we still add it in case the failure is
      // nondeterministic.
      time_map[p.algo] = std::numeric_limits<float>::max();
      continue;
    }
    if (deterministic &&
        std::find(nondeterministic_algos.begin(), nondeterministic_algos.end(),
                  p.algo) != nondeterministic_algos.end()) {
      continue;
    }
    if (p.memory > max_ws_size) {
      continue;
    }
    if (time_map.count(p.algo) == 0) {
      time_map[p.algo] = p.time;
    } else {
      time_map[p.algo] += p.time;
    }
  }
  if (time_map.empty()) {
    LBANN_ERROR("No valid convolution algorithms.");
  }
  AlgoType best_algo = time_map.begin()->first;
  float min_time = std::numeric_limits<float>::max();
  for (const auto& x : time_map) {
    AlgoType algo = x.first;
    float time = x.second;
    if (time < min_time) {
      min_time = time;
      best_algo = algo;
    }
  }
  if (min_time == std::numeric_limits<float>::max()) {
    LBANN_ERROR("No valid convolution algorithms.");
  }
  return best_algo;
}

cudnnConvolutionFwdAlgo_t get_fwd_algo_heuristic(
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const cudnnFilterDescriptor_t& kernel_desc,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& output_desc,
  size_t ws_size) {
  int num_algos;
  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
                get_handle(), input_desc, kernel_desc, conv_desc, output_desc,
                num_algos, &num_tested_algos, perf_results.data()));
  return find_best_heuristic_algorithm(perf_results, nondet_fwd_algos,
                                       deterministic, ws_size);
}

cudnnConvolutionBwdDataAlgo_t get_bwd_data_algo_heuristic(
  bool deterministic,
  const cudnnFilterDescriptor_t& kernel_desc,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& error_signal_desc,
  size_t ws_size) {
  int num_algos;
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                get_handle(), kernel_desc, prev_error_signal_desc, conv_desc,
                error_signal_desc, num_algos, &num_tested_algos,
                perf_results.data()));
  return find_best_heuristic_algorithm(perf_results, nondet_bwd_data_algos,
                                       deterministic, ws_size);
}

cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algo_heuristic(
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnFilterDescriptor_t& kernel_gradient_desc,
  size_t ws_size) {
  int num_algos;
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                get_handle(), input_desc, prev_error_signal_desc, conv_desc,
                kernel_gradient_desc, num_algos, &num_tested_algos,
                perf_results.data()));
  return find_best_heuristic_algorithm(perf_results, nondet_bwd_filter_algos,
                                       deterministic, ws_size);
}

cudnnConvolutionFwdAlgo_t get_fwd_algo_autotune(
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const void* input,
  const cudnnFilterDescriptor_t& kernel_desc,
  const void* kernel,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& output_desc,
  void* output,
  size_t ws_size,
  void* ws) {
  constexpr int num_trials = 3;
  constexpr int num_skip = 1;
  int num_algos;
  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithmEx(
                  get_handle(), input_desc, input, kernel_desc, kernel,
                  conv_desc, output_desc, output, num_algos, &num_tested_algos,
                  perf_results.data(), ws, ws_size));
    if (trial > num_skip) {
      for (const auto& p : perf_results) {
        perf_results_all.push_back(p);
      }
    }
  }
  return find_best_algorithm(perf_results_all, nondet_fwd_algos,
                             deterministic, ws_size);
}

cudnnConvolutionBwdDataAlgo_t get_bwd_data_algo_autotune(
  bool deterministic,
  const cudnnFilterDescriptor_t& kernel_desc,
  const void* kernel,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnTensorDescriptor_t& error_signal_desc,
  void* error_signal,
  size_t ws_size,
  void* ws) {
  constexpr int num_trials = 3;
  constexpr int num_skip = 1;
  int num_algos;
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithmEx(
                  get_handle(), kernel_desc, kernel,
                  prev_error_signal_desc, prev_error_signal,
                  conv_desc, error_signal_desc, error_signal, num_algos,
                  &num_tested_algos, perf_results.data(), ws, ws_size));
    if (trial > num_skip) {
      for (const auto& p : perf_results) {
        perf_results_all.push_back(p);
      }
    }
  }
  return find_best_algorithm(perf_results_all, nondet_bwd_data_algos,
                             deterministic, ws_size);
}

cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algo_autotune(
  bool deterministic,
  const cudnnTensorDescriptor_t& input_desc,
  const void* input,
  const cudnnTensorDescriptor_t& prev_error_signal_desc,
  const void* prev_error_signal,
  const cudnnConvolutionDescriptor_t& conv_desc,
  const cudnnFilterDescriptor_t& kernel_gradient_desc,
  void* kernel_gradient,
  size_t ws_size,
  void* ws) {
  constexpr int num_trials = 3;
  constexpr int num_skip = 1;
  int num_algos;
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                  get_handle(), input_desc, input,
                  prev_error_signal_desc, prev_error_signal,
                  conv_desc, kernel_gradient_desc, kernel_gradient, num_algos,
                  &num_tested_algos, perf_results.data(), ws, ws_size));
    if (trial > num_skip) {
      for (const auto& p : perf_results) {
        perf_results_all.push_back(p);
      }
    }
  }
  return find_best_algorithm(perf_results_all, nondet_bwd_filter_algos,
                             deterministic, ws_size);
}

}  // namespace

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
  void* ws) {
  if (autotune) {
    return get_fwd_algo_autotune(deterministic,
                                 input_desc, input,
                                 kernel_desc, kernel,
                                 conv_desc,
                                 output_desc, output,
                                 ws_size, ws);
  } else {
    return get_fwd_algo_heuristic(deterministic, input_desc, kernel_desc,
                                  conv_desc, output_desc, ws_size);
  }
}

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
  void* ws) {
  if (autotune) {
    return get_bwd_data_algo_autotune(deterministic,
                                      kernel_desc, kernel,
                                      prev_error_signal_desc, prev_error_signal,
                                      conv_desc,
                                      error_signal_desc, error_signal,
                                      ws_size, ws);
  } else {
    return get_bwd_data_algo_heuristic(deterministic, kernel_desc,
                                       prev_error_signal_desc, conv_desc,
                                       error_signal_desc, ws_size);
  }
}

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
  void* ws) {
  if (autotune) {
    return get_bwd_filter_algo_autotune(deterministic,
                                        input_desc, input,
                                        prev_error_signal_desc, prev_error_signal,
                                        conv_desc,
                                        kernel_gradient_desc, kernel_gradient,
                                        ws_size, ws);
  } else {
    return get_bwd_filter_algo_heuristic(deterministic, input_desc,
                                         prev_error_signal_desc, conv_desc,
                                         kernel_gradient_desc, ws_size);
  }
}

} // namespace cudnn
} // namespace lbann

#endif // LBANN_HAS_CUDNN
