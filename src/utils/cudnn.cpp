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

#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/number_theory.hpp"

#include "El.hpp"
#include <iostream>
#include <unordered_map>
#include <tuple>

#ifdef LBANN_HAS_CUDNN

namespace lbann {
namespace cudnn {

void print_version() {
  std::cout << "cudnnGetVersion() : " << (int)cudnnGetVersion() << " , "
            << "CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION
            << std::endl;
}

////////////////////////////////////////////////////////////
// Global cuDNN objects
////////////////////////////////////////////////////////////

namespace {

/** Wrapper for cuDNN handle. */
struct handle_wrapper {
  cudnnHandle_t handle;
  handle_wrapper() : handle(nullptr) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    if (handle == nullptr) { FORCE_CHECK_CUDNN(cudnnCreate(&handle)); }
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
  const auto& dims = m_layer->get_prev_neuron_dims(parent_index);
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
  const auto& dims = m_layer->get_neuron_dims(child_index);
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
  const auto& dims = m_layer->get_neuron_dims(child_index);
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
  const auto& dims = m_layer->get_prev_neuron_dims(parent_index);
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

} // namespace cudnn
} // namespace lbann

#endif // LBANN_HAS_CUDNN
