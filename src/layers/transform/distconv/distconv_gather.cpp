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
#define LBANN_LAYERS_TRANSFORM_DISTCONV_GATHER_INSTANTIATE
#include "lbann/utils/distconv.hpp"
#include "lbann/base.hpp"
#include "lbann/layers/transform/distconv/distconv_gather.hpp"

namespace distconv{
  template<typename Backend, typename DataType>
  template<typename Allocator>
  int 
  Gather<Backend, DataType>
  ::forward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &values, 
          const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &indices,
          tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output){
    
    if(output.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "output buffer is null";
      return 0;
    }
      
    if(values.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "values buffer is null";
      return 0;
    }

    if(indices.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "indices buffer is null";
      return 0;
    }

    const auto& values_shape = values.get_local_shape();    // Should be {1, F, N, B}
    const auto& indices_shape = indices.get_local_shape();  // Should be {1, 1, E, B}
    const auto& output_shape = output.get_local_shape();    // Should be {1, F, E, B}

    // Debug prints -- delete before PR
    #if 1
       util::MPIRootPrintStreamInfo() << "Values Dims: " << values_shape
      << "\tIndices Dims: " << indices_shape << "\tOutput Dims: "<< output_shape; 
    // util::MPIRootPrintStreamInfo() << values; 
    // util::MPIRootPrintStreamInfo() << indices;
    // util::MPIRootPrintStreamInfo() << output;
    #endif

    const auto& num_columns = values_shape[1];
    const auto& num_values_rows = values_shape[2];
    const auto& local_mini_batch_size = values_shape[3];
    const auto& num_output_rows = output_shape[2];

    m_dist_gather->gather(values.get_buffer(),
                          indices.get_buffer(),
                          output.get_buffer(),
                          local_mini_batch_size,
                          num_values_rows,
                          num_columns,
                          num_output_rows);
    
    return 1;
  }

  template<typename Backend, typename DataType>
  template<typename Allocator>
  int 
  Gather<Backend, DataType>
  ::backward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad, 
             const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &indices, 
             tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &values_grad, 
             tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &indices_grad){
    
    const auto& output_grad_shape = output_grad.get_local_shape(); // Should be {1, F, E, B}
    const auto& indices_shape = indices.get_local_shape();  // Should be {1, 1, E, B}
    const auto& values_grad_shape = values_grad.get_local_shape();  // Should be {1, F, N, B}

    const auto num_columns = output_grad_shape[1];            // F
    const auto num_output_grad_rows = output_grad_shape[2];   // E
    const auto local_mini_batch_size = output_grad_shape[3];  // B
    const auto num_values_grad_rows = values_grad_shape[2];   // N

    if(output_grad.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "output grad buffer is null";
      return 0; 
    }

    if(indices.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "indices buffer is null";
      return 0;
    }

    if(values_grad.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "values grad buffer is null";
      return 0;
    }

    m_dist_scatter->scatter(output_grad.get_buffer(),
                            indices.get_buffer(),
                            values_grad.get_buffer(),
                            local_mini_batch_size,
                            num_output_grad_rows,
                            num_columns,
                            num_values_grad_rows);
    return 1;
  }

  template<typename Backend, typename DataType>
  void
  Gather<Backend, DataType>
  ::setup(){
    return ;
  }

// Explicit template instantiation

#define ETI(T, Backend)                                                                   \
  template class Gather<Backend, T>;                                                      \
  template int Gather<Backend, T>::forward<tensor::CUDAAllocator>(                        \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &values,            \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &indices,           \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &output);                 \
  template int Gather <Backend, T>::backward<tensor::CUDAAllocator>(                      \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &output_grad,       \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &indices,           \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &values_grad,             \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &indices_grad);

ETI(float, cudnn::BackendCUDNN)
ETI(double, cudnn::BackendCUDNN)
#undef ETI
} // namespace distconv