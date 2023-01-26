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

#ifndef LBANN_LAYERS_MISC_DISTCONV_CHANNELWISE_SOFTMAX
#define LBANN_LAYERS_MISC_DISTCONV_CHANNELWISE_SOFTMAX

#ifdef LBANN_HAS_DISTCONV
namespace distconv{
  template <typename Backend, typename DataType>
  class ChannelwiseSoftmax{
    using LocaleMPI = tensor::LocaleMPI;

    public:
      ChannelwiseSoftmax(Backend &backend):m_be(backend){};
    
    template <typename Allocator>
    int forward(
      const tensor::Tensor<DataType, LocaleMPI, Allocator> &input_0,
      tensor::Tensor<DataType, LocaleMPI, Allocator> &output);

    template <typename Allocator>
    int backward(
      const tensor::Tensor<DataType, LocaleMPI, Allocator> &input_0,
      const tensor::Tensor<DataType, LocaleMPI, Allocator> &output_grad,
      tensor::Tensor<DataType, LocaleMPI, Allocator> &input_grad_0);

    protected:
      Backend &m_be;

  };
}

#endif // LBANN_HAS_DISTCONV
#endif // LBANN_LAYERS_MISC_DISTCONV_CHANNELWISE_SOFTMAX