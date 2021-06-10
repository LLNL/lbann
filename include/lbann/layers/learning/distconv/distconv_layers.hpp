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

#ifndef LBANN_LAYERS_LEARNING_DISTCONV_LAYERS
#define LBANN_LAYERS_LEARNING_DISTCONV_LAYERS

#ifdef LBANN_HAS_DISTCONV

#include "distconv/base.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"

namespace distconv{
  namespace lbann{
  template <typename Backend, typename DataType>
  class Linear {
    public:
      Linear(Backend &backend);

    template <typename Tensor, typename Nondc_Tensor>
    int forward(bool transpose_A,
               bool transpose_B,
               typename Tensor::data_type alpha,
               const Nondc_Tensor &weights,
               const Tensor &input,
               typename Tensor::data_type beta, 
               Tensor &output 
               ){

      const auto& one = El::TypeTraits<DataType>:: One();
      const auto& zero = El::TypeTraits<DataType>:: Zero();

      if (input.get_local_size() == 0){
        return 0; // no op for empty input
      }
      El::Gemm(
        transpose_A ? El::TRANSPOSE : El::NORMAL,
        El::NORMAL,
        one,  weights, input,
        zero, output);

      return 0;
    }

    template <typename Tensor>
    int  apply_bias(const Tensor &bias,
                    Tensor &output){

      using LocalMat = El::Matrix<TensorDataType, Device>;
      LocalMat ones(local_mini_batch_size * num_channels, 1);

      return 0;
    }

    template <typename Tensor>
    int backward(){
      return 0;
    }

    template <typename Tensor>
    int backward_bias(){
      return 0;
    }

  }; // class definition Linear
} // namespace LBANN
}  // namespace distconv
#endif // LBANN_HAS_DISTCONV
#endif // LBANN_LAYERS_LEARNING_DISTCONV_LAYERS