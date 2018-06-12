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

void print_version() {
  std::cout << "cudnnGetVersion() : " << (int)cudnnGetVersion() << " , "
            << "CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION
            << std::endl;
}

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
    dims.insert(dims.begin(), 1);
    strides.insert(strides.begin(), strides.front());
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
// cuDNN manager
////////////////////////////////////////////////////////////

cudnn_manager::cudnn_manager(size_t workspace_size)
  : m_handle(nullptr),
    m_workspace_size(workspace_size) {

  // Check that Hydrogen has detected GPUs
  if (El::GPUManager::NumDevices() < 1)  {
    LBANN_ERROR("no GPUs detected");
  }
  CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));

#ifdef HYDROGEN_HAVE_CUB
  // Expand CUB GPU memory pool to contain workspace
  if (m_workspace_size > 0) {
    GPUMat workspace;
    workspace.SetMemoryMode(1);
    workspace.Resize((m_workspace_size + sizeof(DataType) - 1) / sizeof(DataType), 1);
  }
#endif // HYDROGEN_HAVE_CUB

  // Initialize cuDNN handle
  FORCE_CHECK_CUDNN(cudnnCreate(&m_handle));
  if (m_handle == nullptr) {
    LBANN_ERROR("failed to create cuDNN handle");
  }
  CHECK_CUDNN(cudnnSetStream(m_handle, El::GPUManager::Stream()));

}

cudnn_manager::~cudnn_manager() {
  if (m_handle != nullptr) { cudnnDestroy(m_handle); }
}

cudnnHandle_t& cudnn_manager::get_handle() {
  CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
  CHECK_CUDNN(cudnnSetStream(m_handle, El::GPUManager::Stream()));
  return m_handle;
}

////////////////////////////////////////////////////////////
// Base cuDNN tensor manager
////////////////////////////////////////////////////////////

layer_tensor_manager::layer_tensor_manager(const Layer* l)
  : m_layer(nullptr) {
  set_layer(l);
}

layer_tensor_manager::layer_tensor_manager(const layer_tensor_manager& other)
  : m_layer(nullptr) {
  set_layer(other.m_layer);
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
  set_layer(other.m_layer);
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

  // Set layer pointer
  m_layer = new_layer;

  // Destroy excess descriptors
  const int num_parents = (m_layer == nullptr ?
                           0 : m_layer->get_num_parents());
  const int num_children = (m_layer == nullptr ?
                            0 : m_layer->get_num_children());
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

  // Resize vectors to get correct number of descriptors
  m_prev_activations.resize(num_parents, nullptr);
  m_activations.resize(num_children, nullptr);
  m_prev_error_signals.resize(num_children, nullptr);
  m_error_signals.resize(num_parents, nullptr);

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
    while (dims.size() < 3) {
      dims.insert(dims.begin(), 1);
      strides.insert(strides.begin(), strides.front());
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
  set_layer(m_layer);
  const auto& local_data = m_layer->get_local_prev_activations(parent_index);
  const auto& dims = m_layer->get_prev_neuron_dims(parent_index);
  auto& desc = m_prev_activations[parent_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager::get_activations(int child_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  set_layer(m_layer);
  const auto& local_data = m_layer->get_local_activations(child_index);
  const auto& dims = m_layer->get_neuron_dims(child_index);
  auto& desc = m_activations[child_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager::get_prev_error_signals(int child_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  set_layer(m_layer);
  const auto& local_data = m_layer->get_local_prev_error_signals(child_index);
  const auto& dims = m_layer->get_neuron_dims(child_index);
  auto& desc = m_prev_error_signals[child_index];
  set_data_parallel_tensor_desc(desc, dims, local_data);
  return desc;
}

cudnnTensorDescriptor_t& data_parallel_layer_tensor_manager::get_error_signals(int parent_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  set_layer(m_layer);
  const auto& local_data = m_layer->get_local_error_signals(parent_index);
  const auto& dims = m_layer->get_prev_neuron_dims(parent_index);
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

/** Set a cuDNN tensor descriptor for a data-parallel data layout.
 */
void set_entrywise_tensor_desc(cudnnTensorDescriptor_t& desc,
                               const AbsMat& local_data) {
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("attempted to setup cuDNN tensor with non-GPU data");
  }
#endif // LBANN_DEBUG
  if (local_data.Height() > 0 && local_data.Width() > 0) {
    std::vector<int> dims(2), strides(2);
    dims[0] = local_data.Width();
    dims[1] = local_data.Height();
    strides[0] = local_data.LDim();
    strides[1] = 1;
    set_tensor_desc(desc, dims, strides);
  }
}

} // namespace

cudnnTensorDescriptor_t& entrywise_layer_tensor_manager::get_prev_activations(int parent_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  set_layer(m_layer);
  const auto& local_data = m_layer->get_local_prev_activations(parent_index);
  auto& desc = m_prev_activations[parent_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

cudnnTensorDescriptor_t& entrywise_layer_tensor_manager::get_activations(int child_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  set_layer(m_layer);
  const auto& local_data = m_layer->get_local_activations(child_index);
  auto& desc = m_activations[child_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

cudnnTensorDescriptor_t& entrywise_layer_tensor_manager::get_prev_error_signals(int child_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  set_layer(m_layer);
  const auto& local_data = m_layer->get_local_prev_error_signals(child_index);
  auto& desc = m_prev_error_signals[child_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

cudnnTensorDescriptor_t& entrywise_layer_tensor_manager::get_error_signals(int parent_index) {
  if (m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  set_layer(m_layer);
  const auto& local_data = m_layer->get_local_error_signals(parent_index);
  auto& desc = m_error_signals[parent_index];
  set_entrywise_tensor_desc(desc, local_data);
  return desc;
}

} // namespace cudnn

} // namespace lbann
#endif // LBANN_HAS_CUDNN
