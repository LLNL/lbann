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
  namespace
  {

    /** Copy between two device buffers, using all threads in a warp. */
    template <typename T> __device__ __forceinline__
    T* memcpy_warp(T* __restrict__ dest, const T* __restrict__ src, int n) {
      constexpr int warp_size = 32;
      for (int i = threadIdx.x; i < n; i += warp_size) {
        dest[i] = src[i];
      }
      __syncwarp();
      return dest;
    }

    /** Set device buffer, using all threads in a warp. */
    template <typename T> __device__ __forceinline__
    T* memset_warp(T* buf, T val, int n) {
      constexpr int warp_size = 32;
      for (int i = threadIdx.x; i < n; i += warp_size) {
        buf[i] = val;
      }
      __syncwarp();
      return buf;
    }

    template <typename T>
    __global__ void get_vector(){

      // Matrix dimensions
      const auto rows = input_dims[0];
      const auto columns = input_dims[1];


      // Indices
      const size_t bidx = blockIdx.x;
      const size_t bidy = blockIdx.y;
      const size_t nblocksx = gridDim.x;
      const size_t nblocksy = gridDim.y;

    }
  } // namespace <anon>

  template<typename Backend, typename DataType>
  template<typename Allocator>
  int 
  Scatter<Backend, DataType>
  ::forward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &values, 
          const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &indices,
          tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output){
    if(output.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "output buffer is null";
    }
    const auto& values_dims = values.get_local_shape();
    const auto& indices_dims = indices.get_local_shape();

    util::MPIRootPrintStreamInfo() << "Values Dims: " << values_dims
      << "Indices dims: " << indices_dims; 

    util::MPIRootPrintStreamInfo() << values; 
    util::MPIRootPrintStreamInfo() << indices;
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