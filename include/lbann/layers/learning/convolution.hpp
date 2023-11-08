////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED

#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
class convolution_distconv_adapter
  : public base_convolution_adapter<TensorDataType, Device>
{
public:
  using TensorDevType =
    typename base_convolution_adapter<TensorDataType, Device>::TensorDevType;

  convolution_distconv_adapter(Layer& layer)
    : base_convolution_adapter<TensorDataType, Device>(layer)
  {}
  virtual ~convolution_distconv_adapter() = default;

  void setup_distributions(tensor_overlap_constraints& constraints) override;
  void setup_layer(size_t workspace_capacity) override;
  dc::Shape get_activations_local_shape(int index = 0) const override;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Convolution
 *
 *  Applies convolution (more precisely, cross-correlation) to input
 *  tensor. This is primarily optimized for image data in CHW format.
 *
 *  Two weights are required if bias is applied: a kernel tensor (in
 *  KCHW format) and per-channel biases. Only the kernel weights are
 *  required if bias is not applied. If weights aren't provided, the
 *  kernel weights are initialized with He normal initialization and
 *  the bias weights are initialized to zero.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class convolution_layer : public base_convolution_layer<TensorDataType, Device>
{

  static_assert(Layout == data_layout::DATA_PARALLEL,
                "convolution layer only supports DATA_PARALLEL");

public:
  convolution_layer(int num_data_dims,
                    int num_output_channels,
                    int conv_dim,
                    int pad,
                    int stride,
                    int dilation,
                    int groups,
                    bool has_bias = true);

  convolution_layer(int num_data_dims,
                    int num_output_channels,
                    std::vector<int> conv_dims,
                    std::vector<int> pads,
                    std::vector<int> strides,
                    std::vector<int> dilations,
                    int groups,
                    bool has_bias = true);

  convolution_layer* copy() const override
  {
    return new convolution_layer(*this);
  }

  std::string get_type() const override { return "convolution"; }

  data_layout get_data_layout() const override { return Layout; }

  El::Device get_device_allocation() const override { return Device; }

  bool can_run_inplace() const override { return false; }

  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | WEIGHTS | PREV_ACTIVATIONS;
  }

#ifdef LBANN_HAS_ONNX
  std::string get_onnx_op_type() const override { return "Conv"; }
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  convolution_layer();

  void setup_dims() override;
  std::vector<int> get_kernel_dims() const override;
  void fp_compute() override;
  void bp_compute() override;

#ifdef LBANN_HAS_DISTCONV
  friend class convolution_distconv_adapter<TensorDataType, Layout, Device>;

protected:
  void setup_distconv_adapter() override;
  bool is_distconv_supported() const override;
#endif // LBANN_HAS_DISTCONV
};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(convolution);

#ifndef LBANN_CONVOLUTION_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class convolution_layer<T,                                   \
                                          data_layout::DATA_PARALLEL,          \
                                          Device>;

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CONVOLUTION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED
