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
                     int num_samples,
                     const std::vector<int> sample_dims,
                     int sample_stride) {

    // Determine tensor dimensions
    // Note: cuDNN tensors should be non-empty and have at least 4
    // dimensions
    std::vector<int> dims = sample_dims;
    dims.insert(dims.begin(), num_samples);
    while (dims.size() < 4) { dims.insert(dims.begin() + 1, 1); }

    // Check that tensor dimensions are valid
    if (std::any_of(dims.begin(), dims.end(),
                    [] (int d) { return d <= 0; })) {
      std::stringstream err;
      err << "attempted to set cuDNN tensor descriptor "
          << "with invalid dimensions (";
      for (size_t i = 0; i < dims.size(); ++i) {
        err << (i == 0 ? "" : " x ") << dims[i];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }

    // Determine tensor strides
    std::vector<int> strides(dims.size());
    strides.back() = 1;
    for(int i = dims.size() - 1; i > 0; --i) {
        strides[i-1] = strides[i] * dims[i];
    }
    strides.front() = std::max(strides.front(), sample_stride);

    // Set cuDNN tensor descriptor
    if (desc == nullptr) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    }
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                           get_data_type(),
                                           dims.size(),
                                           dims.data(),
                                           strides.data()));

}

void set_tensor_desc(cudnnTensorDescriptor_t& desc,
                     const AbsMat& matrix) {
    if (matrix.GetDevice() != El::Device::GPU) {
        LBANN_ERROR("cuDNN tensors must reside on GPU");
    }

    // Determine tensor dimensions and strides
    // Note: cuDNN tensors should have at least 4 dimension
    const int height = matrix.Height();
    const int width = matrix.Width();
    const int leading_dim = matrix.LDim();
    const std::vector<int> dims = {1, 1, width, height};
    const std::vector<int> strides = {width * leading_dim,
                                      width * leading_dim,
                                      leading_dim, 1};

    // Check that tensor dimensions are valid
    if (std::any_of(dims.begin(), dims.end(),
                    [] (int d) { return d <= 0; })) {
      std::stringstream err;
      err << "attempted to set cuDNN tensor descriptor "
          << "with invalid dimensions (";
      for (size_t i = 0; i < dims.size(); ++i) {
        err << (i == 0 ? "" : " x ") << dims[i];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }

    // Set cuDNN tensor descriptor
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

} // namespace cudnn

} // namespace lbann
#endif // LBANN_HAS_CUDNN
