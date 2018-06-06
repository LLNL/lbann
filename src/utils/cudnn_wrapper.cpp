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

cudnnDataType_t get_cudnn_data_type() {
  switch (sizeof(DataType)) {
  case 2: return CUDNN_DATA_HALF;
  case 4: return CUDNN_DATA_FLOAT;
  case 8: return CUDNN_DATA_DOUBLE;
  default: LBANN_ERROR("invalid data type for cuDNN");
  }
  return CUDNN_DATA_FLOAT;
}

void set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                           int num_samples,
                           const std::vector<int>& sample_dims,
                           int sample_stride) {

    // Create tensor descriptor if needed
    if (desc == nullptr) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    }

    // Determine tensor dimensions
    // Note: cuDNN tensors should have at least 4 dimension
    std::vector<int> dims = sample_dims;
    while (dims.size() < 3) {
        dims.insert(dims.begin(), 1);
    }
    dims.insert(dims.begin(), num_samples);

    // Determine tensor strides
    std::vector<int> strides(dims.size());
    strides.back() = 1;
    for(int i = dims.size() - 1; i > 0; --i) {
        strides[i-1] = strides[i] * dims[i];
    }
    strides.front() = std::max(strides.front(), sample_stride);

    // Set cuDNN tensor descriptor
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                           get_cudnn_data_type(),
                                           dims.size(),
                                           dims.data(),
                                           strides.data()));

}

void set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                           int height,
                           int width,
                           int leading_dim) {

    // Create tensor descriptor if needed
    if (desc == nullptr) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    }

    // Determine tensor dimensions and strides
    // Note: cuDNN tensors should have at least 4 dimension
    leading_dim = std::max(height, leading_dim);
    const std::vector<int> dims = {1, 1, width, height};
    const std::vector<int> strides = {width * leading_dim,
                                      width * leading_dim,
                                      leading_dim,
                                      1};

    // Set cuDNN tensor descriptor
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                           get_cudnn_data_type(),
                                           dims.size(),
                                           dims.data(),
                                           strides.data()));

}

void copy_tensor_cudnn_desc(const cudnnTensorDescriptor_t& src,
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

void copy_kernel_cudnn_desc(const cudnnFilterDescriptor_t& src,
                            cudnnFilterDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyFilterDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnDataType_t data_type;
        cudnnTensorFormat_t format;
        int num_dims;
        CHECK_CUDNN(cudnnGetFilterNdDescriptor(src,
                                               0,
                                               &data_type,
                                               &format,
                                               &num_dims,
                                               nullptr));
        std::vector<int> dims(num_dims);
        CHECK_CUDNN(cudnnGetFilterNdDescriptor(src,
                                               num_dims,
                                               &data_type,
                                               &format,
                                               &num_dims,
                                               dims.data()));
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(dst,
                                               data_type,
                                               format,
                                               num_dims,
                                               dims.data()));
    }

}

void copy_convolution_cudnn_desc(const cudnnConvolutionDescriptor_t& src,
                                 cudnnConvolutionDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnConvolutionMode_t mode;
        cudnnDataType_t data_type;
        int num_dims;
        CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(src,
                                                    0,
                                                    &num_dims,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    &mode,
                                                    &data_type));
        std::vector<int> pads(num_dims), strides(num_dims), upscales(num_dims);
        CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(src,
                                                    num_dims,
                                                    &num_dims,
                                                    pads.data(),
                                                    strides.data(),
                                                    upscales.data(),
                                                    &mode,
                                                    &data_type));
        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(dst,
                                                    num_dims,
                                                    pads.data(),
                                                    strides.data(),
                                                    upscales.data(),
                                                    mode,
                                                    data_type));
    }

}

void copy_pooling_cudnn_desc(const cudnnPoolingDescriptor_t& src,
                             cudnnPoolingDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyPoolingDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnPoolingMode_t mode;
        cudnnNanPropagation_t nan_propagation;
        int num_dims;
        CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                                0,
                                                &mode,
                                                &nan_propagation,
                                                &num_dims,
                                                nullptr,
                                                nullptr,
                                                nullptr));
        std::vector<int> dims(num_dims), pads(num_dims), strides(num_dims);
        CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                                0,
                                                &mode,
                                                &nan_propagation,
                                                &num_dims,
                                                dims.data(),
                                                pads.data(),
                                                strides.data()));
        CHECK_CUDNN(cudnnSetPoolingNdDescriptor(dst,
                                                mode,
                                                nan_propagation,
                                                num_dims,
                                                dims.data(),
                                                pads.data(),
                                                strides.data()));
    }

}

void copy_activation_cudnn_desc(const cudnnActivationDescriptor_t& src,
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

void copy_lrn_cudnn_desc(const cudnnLRNDescriptor_t& src,
                         cudnnLRNDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreateLRNDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyLRNDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        unsigned n;
        double alpha, beta, k;
        CHECK_CUDNN(cudnnGetLRNDescriptor(src, &n, &alpha, &beta, &k));
        CHECK_CUDNN(cudnnSetLRNDescriptor(dst, n, alpha, beta, k));
    }

}

}// namespace cudnn

}// namespace lbann
#endif // #ifdef LBANN_HAS_CUDNN
