////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_LAYERS_MATH_DISTCONV_MATMUL 
#define LBANN_LAYERS_MATH_DISTCONV_MATMUL
#include "lbann/utils/distconv.hpp"
#include "distconv/base.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"

#ifdef LBANN_HAS_DISTCONV
namespace distconv{
  template <typename Backend, typename DataType>
  class MatMul {
    using LocalMPI = tensor::LocaleMPI;

    public:
      MatMul(Backend &backend):m_be(backend){};
    
    template <typename Allocator>
    int forward(
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_0,
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_1,
      tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output,
      const bool transpose_0,
      const bool transpose_1);

    template <typename Allocator>
    int backward(
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_0,
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_1,
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad,
      tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_grad_0,
      tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_grad_1,
      const bool transpose_0,
      const bool transpose_1);
    
    protected:
      Backend &m_be;
  };

  template<typename DataType, typename locale, typename Allocator>
  tensor::Shape
  get_matmul_local_tensor_shape(const tensor::Tensor<DataType, locale, Allocator> &input_0,
                                const tensor::Tensor<DataType, locale, Allocator> &input_1,
                                bool transpose_1,
                                bool transpose_2){
    // Use input dims to fill channel and mini-batch dimensions
    auto output_local_shape = input_0.get_local_shape(); 

    auto inp_0_dims = input_0.get_local_shape();
    auto inp_1_dims = input_1.get_local_shape();
    
    // Update the matrix dimensions according to transpose and input matrix shapes
    output_local_shape[0] = transpose_2? inp_1_dims[1] : inp_1_dims[0];
    output_local_shape[1] = transpose_1? inp_0_dims[0] : inp_0_dims[1];

    return output_local_shape; 
  }

extern template class MatMul<::distconv::BackendDNNLib, float>;
extern template class MatMul<::distconv::BackendDNNLib, double>;
}  // namespace distconv

#endif // LBANN_HAS_DISTCONV
#endif  // LBANN_LAYERS_MATH_DISTCONV_MATMUL
