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
      using LocaleMPI = tensor::LocaleMPI;

    public:
      Linear(Backend &backend);

      template <typename Allocator>
      int forward(bool transpose_A,
                  tensor::Tensor<DataType, LocaleMPI, Allocator> &input,
                  const tensor::Tensor<DataType, LocaleMPI, Allocator> &linearity,
                  tensor::Tensor<DataType, LocaleMPI, Allocator> &output,
                  int local_mini_batch_size 
                 ){

        const auto& one = El::TypeTraits<DataType>:: One();
        const auto& zero = El::TypeTraits<DataType>:: Zero();

        if (input.get_local_size() == 0){
          return 0; // no op for empty input
        }
        const auto& input_dims = input.get_local_shape();
        const auto& output_dims = output.get_local_shape();

        const auto& input_size = std::accumulate(input_dims.begin()+1, input_dims.end(), 1, std::multiplies<size_t>());
        const auto& output_size = std::accumulate(output_dims.begin()+1, output_dims.end(), 1, std::multiplies<size_t>());

        ::El::Matrix<DataType> in_mat(input_size, local_mini_batch_size*m_num_local_channels, input.get_buffer(), input_size);
        ::El::Matrix<DataType> out_mat(output_size, local_mini_batch_size*m_num_local_channels, output.get_buffer(), output_size);
        ::El::Matrix<DataType> weights(input_size, output_size, linearity.get_buffer(), input_size);

        // ::El::Gemm(
        //   transpose_A ? El::TRANSPOSE : El::NORMAL,
        //   El::NORMAL,
        //   one,  weights, in_mat,
        //   zero, out_mat);

        return 0;
      }

      // template <typename Tensor>
      int apply_bias(){

        return 0;
      }

      // template <typename Tensor>
      int backward_wrt_input(){
        return 0;
      }

      // template<typename Tensor>
      int backward_wrt_weight(){
        return 0;
      }

      // template<typename Tensor>
      int backward_wrt_bias(){
        return 0;
      }


      template <typename Allocator>
      void setup(int num_local_channels,
                 size_t ws_size){
        m_num_local_channels = num_local_channels;
      }

      template <typename Tensor>
      void setup_bias(){ 
      }

  protected:
    int m_num_local_channels; // Set in setup() 
    int m_num_out_channels; // Set in setup()


  }; // class definition Linear
  // template <typename DataType, typename locale, typename Allocator>
  // tensor::Shape 
  // get_fc_output_local_tensor_shape(const tensor::Tensor<DataType, Locale, Allocator> &input,
  //                                  const int_vector
  //                                  int num_groups){

  // }
}  // namespace distconv
#endif // LBANN_HAS_DISTCONV
#endif // LBANN_LAYERS_LEARNING_DISTCONV_LAYERS