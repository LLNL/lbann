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

#define LBANN_LAYERS_MISC_CHANNELWISE_SOFTMAX_INSTANTIATE
#include "lbann/utils/distconv.hpp"
#include "lbann/base.hpp"
#include "lbann/layers/misc/distconv/distconv_channelwise_softmax.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include "../channelwise_softmax_kernels.cuh"


#ifdef LBANN_HAS_DISTCONV
namespace distconv{
  template<typename Backend, typename DataType>
  template<typename Allocator>
  int
  ChannelwiseSoftmax<Backend, DataType>
  ::forward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_0,
            tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output){

    if (input_0.get_local_size() == 0 || output.get_local_size() == 0){
      util::MPIRootPrintStreamInfo() << "WARNING: EMPTY INPUT FOUND \n";
      return 1; // no op for empty inputs
    }

    const auto& input_0_dims = input_0.get_local_shape();
    
    const auto num_channels = input_0_dims[2];
    const auto local_mini_batch_size = input_0_dims[3];
    const auto mat_channel_size = input_0_dims[0] * input_0_dims[1];
    const auto mat_stride = num_channels * mat_channel_size;

    // Convert to Hydrogen matrices for kernel launch

    using LocalMat = El::Matrix<DataType, El::Device::GPU>;

    LocalMat local_input(mat_stride,
                        local_mini_batch_size,
                        input_0.get_buffer(),
                        mat_stride);

    LocalMat local_output(mat_stride,
                          local_mini_batch_size,
                          output.get_buffer(),
                          mat_stride);
    
    ::lbann::channelwise_softmax_fp_impl(num_channels,
                                         mat_channel_size,
                                         local_input,
                                         local_output);
    return 1;        
  }

  template<typename Backend, typename DataType>
  template<typename Allocator>
  int
  ChannelwiseSoftmax<Backend, DataType>
  ::backward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_0,
             const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad,
             tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_grad_0){
    if (input_0.get_local_size() == 0 ||
        output_grad.get_local_size() == 0 ||
        input_grad_0.get_local_size() == 0){
      return 1; // no op for empty inputs
    }

    const auto& input_0_dims = input_0.get_local_shape();
    const auto num_channels = input_0_dims[2];
    const auto local_mini_batch_size = input_0_dims[3];
    const auto mat_channel_size = input_0_dims[0] * input_0_dims[1];
    const auto mat_stride = num_channels * mat_channel_size;

    // Convert to Hydrogen matrices for kernel launch

    using LocalMat = El::Matrix<DataType, El::Device::GPU>;

    LocalMat local_input(mat_stride,
                        local_mini_batch_size,
                        input_0.get_buffer(),
                        mat_stride);

    LocalMat local_output_grad(mat_stride,
                               local_mini_batch_size,
                               output_grad.get_buffer(),
                               mat_stride);
    
    LocalMat local_input_grad(mat_stride,
                              local_mini_batch_size,
                              input_grad_0.get_buffer(),
                              mat_stride);

    ::lbann::channelwise_softmax_bp_impl(num_channels,
                                         mat_channel_size,
                                         local_input,
                                         local_output_grad,
                                         local_input_grad);
    return 1;        
  }

// =========================================================
// Explicit template instantiation
// =========================================================

#define ETI(T, Backend)                                                                 \
  template class ChannelwiseSoftmax<Backend, T>;                                        \
  template int ChannelwiseSoftmax<Backend, T>::forward<tensor::CUDAAllocator>(          \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &input_0,         \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &output_0);             \
  template int ChannelwiseSoftmax<Backend, T>::backward<tensor::CUDAAllocator>(         \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &input_0,         \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &input_1,         \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator> &output_grad);   

ETI(float, BackendDNNLib)
ETI(double, BackendDNNLib)
#undef ETI
}  // namespace distconv
#endif // LBANN_HAS_DISTCONV