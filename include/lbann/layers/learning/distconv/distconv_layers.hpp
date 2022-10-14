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

#ifndef LBANN_LAYERS_LEARNING_DISTCONV_LAYERS
#define LBANN_LAYERS_LEARNING_DISTCONV_LAYERS
#include "lbann/utils/distconv.hpp"
#include "distconv/base.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"


#ifdef LBANN_HAS_DISTCONV
namespace distconv{
  template <typename Backend, typename DataType>
  class ChannelwiseFullyConnected {
      using LocaleMPI = tensor::LocaleMPI;

    public:
      ChannelwiseFullyConnected(Backend &backend):m_be(backend){};

    template <typename Allocator>
    int forward(bool transpose_A,
            const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input,
            const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &linearity,
            tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output);

    template <typename Allocator>
    int apply_bias(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &bias,
                 tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output);

    template <typename Allocator>
    int backward_wrt_input(
      bool transpose_A,
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad,
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &linearity,
      tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_grad);

    template <typename Allocator>
    int backward_wrt_weight(
      bool transpose,
      DataType dst_scale,
      DataType gradient_scale,
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input,
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad,
      tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &linearity_grad);

    template <typename Allocator>
    int backward_wrt_bias(
      DataType gradient_scale,
      DataType dst_scale,
      const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad,
      tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &bias_grad);

  protected:
    Backend &m_be;
  }; // class definition ChannelwiseFullyConnected


  template <typename DataType, typename locale, typename Allocator>
  tensor::Shape
  get_fc_output_local_tensor_shape(const tensor::Tensor<DataType, locale, Allocator> &input,
                                   const int_vector &linearity_dims,
                                   bool transpose){

    //https://github.com/LLNL/DiHydrogen/blob/7f86db1f9701ac3afb5e16aefdd57563d57a1698/legacy/include/distconv/distconv.hpp#L173

    //Get the input layer local tensor shape

    auto output_local_shape = input.get_local_shape();
    output_local_shape[0] = transpose? linearity_dims[1] : linearity_dims[0];
    return output_local_shape;
  }
extern template class ChannelwiseFullyConnected<::distconv::BackendDNNLib, float>;
extern template class ChannelwiseFullyConnected<::distconv::BackendDNNLib, double>;
}  // namespace distconv

#endif // LBANN_HAS_DISTCONV
#endif // LBANN_LAYERS_LEARNING_DISTCONV_LAYERS
