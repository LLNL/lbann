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

#ifndef LBANN_LAYERS_TRANSFORM_DISTCONV_GATHER
#define LBANN_LAYERS_TRANSFORM_DISTCONV_GATHER
#include "lbann/utils/distconv.hpp"
#include "distconv/base.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"


#ifdef LBANN_HAS_DISTCONV
namespace distconv{
  template <typename Backend, typename DataType>
  class Gather{
    using LocaleMPI = tensor::LocaleMPI;
    
    public:
      Gather(Backend &backend):m_backend(backend){};
    
    template<typename Allocator>
    int forward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input,
                const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &indices,
                tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output);

    template<typename Allocator>
    int backward(const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &output_grad,       
                 const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &indices,           
                 tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &values_grad,             
                 tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &indices_grad);

  protected:
    Backend &m_backend;
  };  // class definition Gather
} // namespace distconv
#endif // LBANN_HAS_DISTCONV
#endif // LBANN_LAYERS_TRANSFORM_DISTCONV_GATHER