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
#define LBANN_LAYERS_TRANSFORM_DISTCONV_SCATTER_INSTANTIATE
#include "lbann/utils/distconv.hpp"
#include "lbann/base.hpp"
#include "lbann/layers/transform/distconv/distconv_scatter.hpp"

namespace distconv{

  template<typename Backend, typename DataType>
  template<typename Allocator>
  int 
  Scatter<Backend, DataType>
  ::forward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &values, 
          const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &indices,
          tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output){
     
    const auto& values_dims = values.get_local_shape(); // Should be {1, F, E, B}
    const auto& indices_dims = indices.get_local_shape(); // Should be {1, 1, E, B}
    const auto& output_dims = output.get_local_shape(); // Should be {1, F, C, B}

    // Debug prints -- delete before PR
    #if 1
       util::MPIRootPrintStreamInfo() << "Values Dims: " << values_dims
      << "Indices Dims: " << indices_dims << "Output Dims: "<< output_dims; 

    util::MPIRootPrintStreamInfo() << values; 
    util::MPIRootPrintStreamInfo() << indices;
    util::MPIRootPrintStreamInfo() << output;
    #endif

    const auto& output_size = output_dims[1];
    const auto& indices_size = indices_dims[2];
    const auto& values_size = values_dims[1];

    const auto local_mini_batch_size = output_dims[3];

    const auto input_length = output_dims[2];
    const auto output_length = indices_dims[2];

    if(output.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "output buffer is null";
    }
      
    if (values.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "values buffer is null";
    }

    if (indices.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "indices buffer is null";
    }
   
    // Cast the local matrices and convert to 2D EL matrices
    El::Matrix<DataType, El::Device::GPU> out_mat(output_size,
                                                  local_mini_batch_size,
                                                  output.get_buffer(),
                                                  output_size);

    El::Matrix<DataType, El::Device::GPU> val_mat(values_size,
                                                  local_mini_batch_size,
                                                  values.get_buffer(),
                                                  values_size);

    El::Matrix<DataType, El::Device::GPU> ind_mat(indices_size,
                                                  local_mini_batch_size,
                                                  indices.get_buffer(),
                                                  indices_size);
    
    //  Attach values matrix to the NVSHMEM buffer
    // The size of the NVSHMEM_values buffer is for the local values matrix

    // Retreive value vectors onto the NVSHMEM workspace buffer 
    // The NVSHMEM workspace buffer is the size of the local output matrix 

    // Copy the local workspace buffer onto the output matrix
    return 0;
  }

  template<typename Backend, typename DataType>
  template<typename Allocator>
  int 
  Scatter<Backend, DataType>
  ::backward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad, 
             const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &indices, 
             tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &values_grad, 
             tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &indices_grad){
    return 0;
  }

  template<typename Backend, typename DataType>
  void 
  Scatter<Backend, DataType>
  ::setup(){
    return ;
  }

// Explicit template instantiation

#define ETI(T, Backend)                                                                   \
  template class Scatter<Backend, T>;                                                     \
  template int Scatter<Backend, T>::forward<tensor::CUDAAllocator>(                       \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &values,            \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &indices,           \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &output);                 \
  template int Scatter<Backend, T>::backward<tensor::CUDAAllocator>(                      \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &output_grad,       \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &indices,           \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &values_grad,             \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &indices_grad);

ETI(float, cudnn::BackendCUDNN)
ETI(double, cudnn::BackendCUDNN)
#undef ETI
} // namespace distconv