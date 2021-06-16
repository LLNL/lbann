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
      Linear(Backend &backend){};

      template <typename Allocator>
      int forward(bool transpose_A,
                  const tensor::Tensor<DataType, LocaleMPI, Allocator> &input,
                  const tensor::Tensor<DataType, LocaleMPI, Allocator> &linearity,
                  tensor::Tensor<DataType, LocaleMPI, Allocator> &output,
                  int local_mini_batch_size 
                 ){

        util::MPIPrintStreamDebug()
          << "input tensor. global_shape: "
          << input.get_shape()
          << ", local shape: " << input.get_local_shape()
          << ", local real shape: " << input.get_local_real_shape()
          << ", dist: " << input.get_distribution();

        const auto& one = El::TypeTraits<DataType>::One();
        const auto& zero = El::TypeTraits<DataType>::Zero();

        if (input.get_local_size() == 0){
          return 0; // no op for empty input
        }
        const auto& input_dims = input.get_local_shape();
        const auto& output_dims = output.get_local_shape();

        const auto& input_size = std::accumulate(input_dims.begin()+1, input_dims.end(), 1, std::multiplies<size_t>());
        const auto& output_size = std::accumulate(output_dims.begin()+1, output_dims.end(), 1, std::multiplies<size_t>());

        El::Matrix<DataType> in_mat(input_size, local_mini_batch_size*m_num_local_channels, input.get_buffer(), input_size);
        El::Matrix<DataType> out_mat(output_size, local_mini_batch_size*m_num_local_channels, output.get_buffer(), output_size);
        El::Matrix<DataType> weights(input_size, output_size, linearity.get_buffer(), input_size);

        El::Gemm(transpose_A ? El::TRANSPOSE: El::NORMAL,
                   El::NORMAL,
                   one, 
                   weights,
                   in_mat,
                   zero,
                   out_mat);

        return 0;
      }

      template <typename Allocator>
      int apply_bias(const tensor::Tensor<DataType, LocaleMPI, Allocator> &bias, 
                     tensor::Tensor<DataType, LocaleMPI, Allocator> &output,
                     int local_mini_batch_size){

        const auto& one = El::TypeTraits<DataType>::One();

        const auto& output_dims = output.get_local_shape();

        const auto& output_size = std::accumulate(output_dims.begin()+1, output_dims.end(), 1, std::multiplies<size_t>());


        El::Matrix<DataType> ones(local_mini_batch_size * m_num_local_channels, 1);

        El::Matrix<DataType> out_mat(output_size, local_mini_batch_size*m_num_local_channels, output.get_buffer(), output_size);
        El::Matrix<DataType> bias_vec(output_size, 1, bias.get_buffer(), output_size);

        El::Fill(ones, one);

        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 one,
                 bias_vec,
                 ones,
                 one,
                 out_mat);

        return 0;
      }

      template <typename Allocator>
      int backward_wrt_input(bool transpose_A,
                             const tensor::Tensor<DataType, LocaleMPI, Allocator> &output_grad,
                             const tensor::Tensor<DataType, LocaleMPI, Allocator> &linearity,
                             tensor::Tensor<DataType, LocaleMPI, Allocator> &input_grad,
                             int local_mini_batch_size )
      {
        const auto& one = El::TypeTraits<DataType>:: One();
        const auto& zero = El::TypeTraits<DataType>:: Zero();

        const auto& input_dims = input_grad.get_local_shape();
        const auto& output_dims = output_grad.get_local_shape();


        const auto& input_size = std::accumulate(input_dims.begin()+1, input_dims.end(), 1, std::multiplies<size_t>());
        const auto& output_size = std::accumulate(output_dims.begin()+1, output_dims.end(), 1, std::multiplies<size_t>());

        El::Matrix<DataType> output_grad_mat(output_size, local_mini_batch_size*m_num_local_channels, output_grad.get_buffer(),output_size);
        El::Matrix<DataType> weights(input_size, output_size, linearity.get_buffer(), input_size);
        El::Matrix<DataType> input_grad_mat(input_size, local_mini_batch_size*m_num_local_channels, input_grad.get_buffer(), input_size);

        El::Gemm(transpose_A ? El::NORMAL : El::TRANSPOSE,
                 El::NORMAL,
                 one,
                 weights,
                 output_grad_mat,
                 zero,
                 input_grad_mat);
        return 0;
      }

      template<typename Allocator>
      int backward_wrt_weight(bool transpose,
                              DataType dst_scale,
                              DataType gradient_scale,
                              const tensor::Tensor<DataType, LocaleMPI, Allocator> &input, 
                              const tensor::Tensor<DataType, LocaleMPI, Allocator> &output_grad,
                              tensor::Tensor<DataType, LocaleMPI, Allocator> &linearity_grad,
                              int local_mini_batch_size){

        const auto& input_dims = input.get_local_shape();
        const auto& output_dims = output_grad.get_local_shape();

        const auto& input_size = std::accumulate(input_dims.begin()+1, input_dims.end(), 1, std::multiplies<size_t>());
        const auto& output_size = std::accumulate(output_dims.begin()+1, output_dims.end(), 1, std::multiplies<size_t>());

        El::Matrix<DataType> input_mat(input_size, local_mini_batch_size*m_num_local_channels, input.get_buffer(), input_size);
        El::Matrix<DataType> output_grad_mat(output_size, local_mini_batch_size*m_num_local_channels, output_grad.get_buffer(), output_size);
        El::Matrix<DataType> linearity_grad_mat(input_size, output_size, linearity_grad.get_buffer(), input_size);


        if(transpose){
          El::Gemm(El::NORMAL, El::TRANSPOSE,
                   gradient_scale, input_mat, output_grad_mat,
                   dst_scale, linearity_grad_mat);
        }
        else {
          El::Gemm(El::NORMAL, El::TRANSPOSE,
                   gradient_scale, output_grad_mat, input_mat,
                   dst_scale, linearity_grad_mat);
        }


        return 0;
      }

      template<typename Allocator>
      int backward_wrt_bias(DataType gradient_scale,
                            DataType dst_scale,
                            const tensor::Tensor<DataType, LocaleMPI, Allocator> &output_grad,
                            tensor::Tensor<DataType, LocaleMPI, Allocator> &bias_grad,
                            int local_mini_batch_size){
      
        const auto& one = El::TypeTraits<DataType>::One();
        El::Matrix<DataType> ones(local_mini_batch_size * m_num_local_channels, 1);
        El::Fill(ones, one);



        const auto& output_dims = output_grad.get_local_shape();
        const auto& output_size = std::accumulate(output_dims.begin()+1, output_dims.end(), 1, std::multiplies<size_t>());

        El::Matrix<DataType> out_grad_mat(output_size, local_mini_batch_size*m_num_local_channels, output_grad.get_buffer(), output_size);
        El::Matrix<DataType> bias_grad_vec(output_size, 1, bias_grad.get_buffer(), output_size);

        El::Gemv(El::NORMAL,
                 gradient_scale, out_grad_mat, ones,
                 dst_scale, bias_grad_vec);

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
