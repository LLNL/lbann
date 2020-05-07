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

#define LBANN_CONVOLUTION_LAYER_INSTANTIATE
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/learning/convolution.hpp"

#include "lbann/proto/proto_common.hpp"

#include <layers.pb.h>

namespace lbann {
namespace {

#ifdef LBANN_HAS_CUDNN
using ProtoTensorOpEnumType = decltype(lbann_data::Layer::DEFAULT_TENSOR_OPS);
cudnnMathType_t convert_to_cudnn_math_type(ProtoTensorOpEnumType mt)
{
  switch (mt)
  {
  case lbann_data::Layer::DEFAULT_TENSOR_OPS:
    return cudnn::get_default_convolution_math_type();
  case lbann_data::Layer::NO_TENSOR_OPS:
    return CUDNN_DEFAULT_MATH;
  case lbann_data::Layer::USE_TENSOR_OPS:
    return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  default:
    LBANN_ERROR("Bad math type value.");
  }
  return CUDNN_DEFAULT_MATH;
}
#endif // LBANN_HAS_CUDNN

template <typename TensorDataType, data_layout Layout, El::Device Device>
struct ConvLayerBuilder
{
  static std::unique_ptr<Layer> Build(
    lbann_comm* comm, lbann_data::Layer const& proto_layer){

    const auto& params = proto_layer.convolution();
    const auto& num_output_channels = params.num_output_channels();
    const auto& bias = params.has_bias();
    int num_groups = params.num_groups();
    if (num_groups == 0) {
      num_groups = 1;
    }

    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      std::vector<int> dilations = parse_list<int>(params.conv_dilations());
      if (dilations.empty()) {
        dilations.resize(dims.size(), 1);
      }
#ifdef LBANN_HAS_CUDNN
      auto ret = lbann::make_unique<convolution_layer<TensorDataType, Layout, Device>>(
        comm, dims.size(), num_output_channels,
        dims, pads, strides, dilations, num_groups, bias);
      ret->set_cudnn_math_mode(
        convert_to_cudnn_math_type(params.conv_tensor_op_mode()));
      return ret;
#else
      return lbann::make_unique<convolution_layer<TensorDataType, Layout, Device>>(
        comm, dims.size(), num_output_channels,
        dims, pads, strides, dilations, num_groups, bias);
#endif // LBANN_HAS_CUDNN
    }
    else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      int dilation = params.conv_dilations_i();
      if (dilation == 0) {
        dilation = 1;
      }
#ifdef LBANN_HAS_CUDNN
      auto ret =lbann::make_unique<convolution_layer<TensorDataType, Layout, Device>>(
        comm, num_dims, num_output_channels,
        dim, pad, stride, dilation, num_groups, bias);
      ret->set_cudnn_math_mode(
        convert_to_cudnn_math_type(params.conv_tensor_op_mode()));
      return ret;
#else
      return lbann::make_unique<convolution_layer<TensorDataType, Layout, Device>>(
        comm, num_dims, num_output_channels,
        dim, pad, stride, dilation, num_groups, bias);
#endif // LBANN_HAS_CUDNN
    }
  }
};

template <typename TensorDataType, El::Device Device>
struct ConvLayerBuilder<TensorDataType, data_layout::MODEL_PARALLEL, Device>
{
  static std::unique_ptr<Layer> Build(
    lbann_comm* comm, lbann_data::Layer const& proto_layer){
    LBANN_ERROR("convolution layer is only supported with "
                "a data-parallel layout");
  }
};

}// namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_convolution_layer_from_pbuf(
  lbann_comm* comm,
  const lbann_data::Layer& proto_layer) {
  using Builder = ConvLayerBuilder<TensorDataType, Layout, Device>;
  return Builder::Build(comm, proto_layer);
}

// Note: This unit will also instantiate the base_convolution_layer class.

#define PROTO_DEVICE(T, Device)                                            \
  template class base_convolution_layer<T, Device>;                        \
  template class convolution_layer<T, data_layout::DATA_PARALLEL, Device>; \
    template std::unique_ptr<Layer>                                       \
  build_convolution_layer_from_pbuf<T, data_layout::DATA_PARALLEL, Device>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template std::unique_ptr<Layer>                                       \
  build_convolution_layer_from_pbuf<T, data_layout::MODEL_PARALLEL, Device>( \
    lbann_comm*, lbann_data::Layer const&)


#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
