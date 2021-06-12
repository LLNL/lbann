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
  template <typename Backend, typename DataType>
  class Linear {
    public:
      Linear(Backend &backend);

      template <typename Tensor>
      int forward(bool transpose_A,
                  bool transpose_B,
                  typename Tensor::data_type alpha,
                  const Tensor &input,
                  typename Tensor::data_type beta, 
                  Tensor &output,
                  int local_mini_batch_size, 
                 ){

        const auto& one = El::TypeTraits<DataType>:: One();
        const auto& zero = El::TypeTraits<DataType>:: Zero();

        if (input.get_local_size() == 0){
          return 0; // no op for empty input
        }
        ::El::Matrix(input_size, local_mini_batch_size*num_local_channels, input.get_buffer(), input_size);
        ::El::Matrix(output_size, local_mini_batch_size*num_local_channels, output.get_buffer(), output_size);
        ::El::Matrix();

        ::El::Gemm(
          transpose_A ? El::TRANSPOSE : El::NORMAL,
          El::NORMAL,
          one,  m_weights, input,
          zero, output);

        return 0;
      }

      template <typename Tensor>
      int  apply_bias(const Tensor &bias,
                      Tensor &output){

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
      template <typename Tensor>
      void setup(const Tensor weights,
                 int num_local_channels;
                 size_t ws_size){
        m_num_local_channels = num_local_channels;
        m_weights = weights;
      }

      template <typename Tensor>
      void setup_bias(const Tensor bias){
      }

  protected:
    Tensor<DataType> m_weights;
    Tensor<DataType> m_bias; 
    int m_num_local_channels;


  }; // class definition Linear
}  // namespace distconv
#endif // LBANN_HAS_DISTCONV
#endif // LBANN_LAYERS_LEARNING_DISTCONV_LAYERS