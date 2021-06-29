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
#include "lbann/utils/distconv.hpp"
#include "distconv/base.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"


#ifdef LBANN_HAS_DISTCONV
namespace distconv{
  template <typename Backend, typename DataType>
  class Linear {
      using LocaleMPI = tensor::LocaleMPI;

    public:
      Linear(Backend &backend):m_be(backend){};

      template <typename Allocator>
      void setup(int num_local_channels,
                 size_t ws_size){
        m_num_local_channels = num_local_channels;
      }

    template <typename Allocator>
    int forward(bool transpose_A,
            const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input,
            const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &linearity,
            tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output,
            int local_mini_batch_size){
    const auto& one = El::TypeTraits<DataType>::One();
    const auto& zero = El::TypeTraits<DataType>::Zero();

    if (input.get_local_size() == 0){
      return 0; // no op for empty input
    }
    const auto& input_dims = input.get_local_shape();
    const auto& output_dims = output.get_local_shape();

    const auto& input_size = std::accumulate(input_dims.begin(), input_dims.begin()+1, 1, std::multiplies<size_t>());
    const auto& output_size = std::accumulate(output_dims.begin(), output_dims.begin()+1, 1, std::multiplies<size_t>());

    const auto num_local_channels = output_dims[2];

    // util::MPIRootPrintStreamInfo()
    //   << "input tensor. global_shape: "
    //   << input.get_shape()
    //   << ", local shape: " << input.get_local_shape()
    //   << ", local real shape: " << input.get_local_real_shape()
    //   << ", dist: " << input.get_distribution();

    // util::MPIRootPrintStreamInfo()
    //   << "linearity tensor. global_shape: "
    //   << linearity.get_shape()
    //   << ", local shape: " << linearity.get_local_shape()
    //   << ", local real shape: " << linearity.get_local_real_shape()
    //   << ", dist: " << linearity.get_distribution();

    // util::MPIRootPrintStreamInfo()
    //   << "output tensor. global_shape: "
    //   << output.get_shape()
    //   << ", local shape: " << output.get_local_shape()
    //   << ", local real shape: " << output.get_local_real_shape()
    //   << ", dist: " << output.get_distribution() << "\n local mini batch size: " << local_mini_batch_size
    //   << "\n num local channels: " << num_local_channels
    //   << "\n input_size: " << input_size
    //   << "\n output_size: " << output_size;
    // Check if buffer is not null possibly 

    if(!output.get_buffer() || output.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo()<< "output buffer is null";
    }

    if(!input.get_buffer() || input.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() << "input buffer is null";
    }

    if(!linearity.get_buffer() || linearity.get_buffer() == nullptr){
      util::MPIRootPrintStreamInfo() <<"linearity buffer is null";
    }

    El::Matrix<DataType, El::Device::GPU> in_mat(input_size, local_mini_batch_size*num_local_channels, input.get_buffer(), input_size);
    El::Matrix<DataType, El::Device::GPU> out_mat(output_size, local_mini_batch_size*num_local_channels, output.get_buffer(), output_size);
    El::Matrix<DataType, El::Device::GPU> weights(output_size, input_size, linearity.get_buffer(), output_size);

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
    int apply_bias(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &bias, 
                 tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output,
                 int local_mini_batch_size){
    
    if (output.get_local_size() == 0) return 0;

    const auto& one = El::TypeTraits<DataType>::One();

    const auto& output_dims = output.get_local_shape();

    const auto& output_size = std::accumulate(output_dims.begin(), output_dims.begin()+1, 1, std::multiplies<size_t>());

    const auto num_local_channels = output_dims[2];

    util::MPIRootPrintStreamInfo() 
      << " Bias Global Shape" << bias.get_shape()
      << " Bias local shape" << bias.get_local_shape()
      << " Bias real local shape"  << bias.get_local_real_shape()
      << " Bias distribution" << bias.get_distribution();
    El::Matrix<DataType, El::Device::GPU>  ones(local_mini_batch_size * num_local_channels, 1);

    El::Matrix<DataType, El::Device::GPU>  out_mat(output_size, local_mini_batch_size*num_local_channels, output.get_buffer(), output_size);
    El::Matrix<DataType, El::Device::GPU>  bias_vec(output_size, 1, bias.get_buffer(), output_size);

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
                         const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad,
                         const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &linearity,
                         tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_grad,
                         int local_mini_batch_size ){
    const auto& one = El::TypeTraits<DataType>:: One();
    const auto& zero = El::TypeTraits<DataType>:: Zero();

    const auto& input_dims = input_grad.get_local_shape();
    const auto& output_dims = output_grad.get_local_shape();


    const auto& input_size = std::accumulate(input_dims.begin(), input_dims.begin()+1, 1, std::multiplies<size_t>());
    const auto& output_size = std::accumulate(output_dims.begin(), output_dims.begin()+1, 1, std::multiplies<size_t>());

    const auto num_local_channels = output_dims[2];
    
    util::MPIRootPrintStreamInfo()
    << " Output Grad shapes :" << output_grad.get_shape() 
    << " " << output_grad.get_local_shape()
    << '\t' << output_grad.get_distribution()
    << "\t Sanity check: " << output_size;

    util::MPIRootPrintStreamInfo()
    << " Input Graph shape : " << input_grad.get_shape()
    << '\t' << input_grad.get_local_shape() 
    << '\t' << input_grad.get_distribution()
    << "\t Sanity check: " << input_size;

    El::Matrix<DataType, El::Device::GPU>  output_grad_mat(output_size, local_mini_batch_size*num_local_channels, output_grad.get_buffer(),output_size);
    El::Matrix<DataType, El::Device::GPU>  weights(output_size, input_size, linearity.get_buffer(), output_size);
    El::Matrix<DataType, El::Device::GPU>  input_grad_mat(input_size, local_mini_batch_size*num_local_channels, input_grad.get_buffer(), input_size);
    
    El::Matrix<DataType, El::Device::GPU> output_grad_mat_reshaped;

    if (output_grad_mat.Contiguous()) {
      output_grad_mat_reshaped.LockedAttach(
      output_size,
      local_mini_batch_size * num_local_channels,
      output_grad_mat.LockedBuffer(),
      output_size);
    }
    else {
      El::Copy(output_grad_mat, output_grad_mat_reshaped);
      output_grad_mat_reshaped.Resize(
      output_size,
      local_mini_batch_size * num_local_channels);
    }

    El::Gemm(transpose_A ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             one,
             weights,
             output_grad_mat,
             zero,
             input_grad_mat);
    return 0;

    }

    template <typename Allocator>
    int backward_wrt_weight(bool transpose,
                          DataType dst_scale,
                          DataType gradient_scale,
                          const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input, 
                          const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad,
                          tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &linearity_grad,
                          int local_mini_batch_size){

    const auto is_empty_input = input.get_local_size() == 0;
    const auto is_empty_grad = output_grad.get_local_size() == 0;
    const auto is_empty_weights = linearity_grad.get_local_size() == 0;
    if(is_empty_input ||
       is_empty_weights ||
       is_empty_grad){
      // No op 
      return 0;
    }

    const auto& input_dims = input.get_local_shape();
    const auto& output_dims = output_grad.get_local_shape();

    const auto& input_size = std::accumulate(input_dims.begin(), input_dims.begin()+1, 1, std::multiplies<size_t>());
    const auto& output_size = std::accumulate(output_dims.begin(), output_dims.begin()+1, 1, std::multiplies<size_t>());

    const auto num_local_channels = output_dims[2];

    util::MPIRootPrintStreamInfo()
    << " Output Grad shapes :" << output_grad.get_shape() 
    << " " << output_grad.get_local_shape()
    << '\t' << output_grad.get_distribution()
    << "\t Sanity check: " << output_size;

    util::MPIRootPrintStreamInfo() << "Num local channels \t" << num_local_channels; 
    util::MPIRootPrintStreamInfo() << "local_mini_batch_size\t" << local_mini_batch_size;

    util::MPIRootPrintStreamInfo()
    << " Linearity dims: " << linearity_grad.get_shape()
    << " (local shape)" << linearity_grad.get_local_shape()
    << '\t' << linearity_grad.get_distribution()
    << " \t Sanity check " << input_dims; 

    El::Matrix<DataType, El::Device::GPU>  input_mat_reshaped; 
    El::Matrix<DataType, El::Device::GPU>  output_grad_mat_reshaped;

    El::Matrix<DataType, El::Device::GPU>  input_mat(input_size, local_mini_batch_size*num_local_channels, input.get_buffer(), input_size);
    El::Matrix<DataType, El::Device::GPU>  output_grad_mat(output_size, local_mini_batch_size*num_local_channels, output_grad.get_buffer(), output_size);
    El::Matrix<DataType, El::Device::GPU>  linearity_grad_mat(output_size, input_size, linearity_grad.get_buffer(), output_size);

    
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

    template <typename Allocator>
    int backward_wrt_bias(DataType gradient_scale,
                        DataType dst_scale,
                        const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad,
                        tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &bias_grad,
                        int local_mini_batch_size){

    const auto is_output_grad_empty = output_grad.get_local_size() == 0;
    const auto is_bias_grad_empty = bias_grad.get_local_size() == 0;

    if(is_output_grad_empty ||
       is_bias_grad_empty){

      return 0;
    }

    const auto& one = El::TypeTraits<DataType>::One();
    const auto& output_dims = output_grad.get_local_shape();
    const auto& output_size = std::accumulate(output_dims.begin(), output_dims.begin()+1, 1, std::multiplies<size_t>());

    const auto num_local_channels = output_dims[2];
    
    El::Matrix<DataType, El::Device::GPU>  ones(local_mini_batch_size * num_local_channels, 1);
    
    El::Matrix<DataType, El::Device::GPU>  out_grad_mat(output_size, local_mini_batch_size*num_local_channels, output_grad.get_buffer(), output_size);
    El::Matrix<DataType, El::Device::GPU>  bias_grad_vec(output_size, 1, bias_grad.get_buffer(), output_size);
    
    El::Fill(ones, one);
    El::Gemv(El::NORMAL,
             gradient_scale, out_grad_mat, ones,
             dst_scale, bias_grad_vec);

    return 0;
    }

  template <typename Allocator>
  void allreduce_gradients(
    tensor::Tensor<DataType, LocaleMPI, Allocator> &gradients){

    Al::Allreduce<Al::NCCLBackend, DataType>(
      gradients.get_base_ptr(),
      gradients.get_size(),
      Al::ReductionOperator::sum,
      m_be.get_al_nccl_comm());
  } 

  protected:
    int m_num_local_channels; // Set in setup() 
    int m_num_out_channels; // Set in setup()
    Backend &m_be;


  }; // class definition Linear


  template <typename DataType, typename locale, typename Allocator>
  tensor::Shape 
  get_fc_output_local_tensor_shape(const tensor::Tensor<DataType, locale, Allocator> &input,
                                   const int_vector &linearity_dims){

    //https://github.com/LLNL/DiHydrogen/blob/7f86db1f9701ac3afb5e16aefdd57563d57a1698/legacy/include/distconv/distconv.hpp#L173

    //Get the input layer local tensor shape 

    auto output_local_shape = input.get_local_shape();

      
    output_local_shape[0] = linearity_dims[0];

    return output_local_shape;
  }
// extern template class Linear<::distconv::cudnn::BackendCUDNN, float>;
// extern template class Linear<::distconv::cudnn::BackendCUDNN, double>;


}  // namespace distconv

#endif // LBANN_HAS_DISTCONV
#endif // LBANN_LAYERS_LEARNING_DISTCONV_LAYERS
