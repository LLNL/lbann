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

#ifndef LBANN_LAYERS_LEARNING_DECONVOLUTION_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_DECONVOLUTION_HPP_INCLUDED

#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

// Forward declaration.
namespace callback {
class imcomm;
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
class deconvolution_distconv_adapter
  : public base_convolution_adapter<TensorDataType, Device>
{
public:
  using TensorDevType =
    typename base_convolution_adapter<TensorDataType, Device>::TensorDevType;

  deconvolution_distconv_adapter(Layer& layer)
    : base_convolution_adapter<TensorDataType, Device>(layer)
  {}
  virtual ~deconvolution_distconv_adapter() = default;

  void setup_distributions(tensor_overlap_constraints& constraints) override;
  void setup_layer(size_t workspace_capacity) override;
  dc::Shape get_activations_local_shape(int index = 0) const override;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Convolution transpose
 *
 *  This operation is the transpose of standard deep learning
 *  convolution.
 *
 *  Pedantic comments: this operation is commonly called
 *  "deconvolution" in the deep learning community, but it is not a
 *  true deconvolution. Also, the "convolution" operation commonly
 *  used in the deep learning is actually cross-correlation.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class deconvolution_layer
  : public base_convolution_layer<TensorDataType, Device>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "deconvolution layer only supports DATA_PARALLEL");

private:
  friend class callback::imcomm;

public:
  deconvolution_layer(int num_data_dims,
                      int num_output_channels,
                      std::vector<int> conv_dims,
                      std::vector<int> pads,
                      std::vector<int> strides,
                      std::vector<int> dilations,
                      std::vector<int> output_pads,
                      int groups,
                      bool has_bias);

  deconvolution_layer* copy() const override
  {
    return new deconvolution_layer(*this);
  }

  std::string get_type() const override { return "deconvolution"; }

  data_layout get_data_layout() const override { return Layout; }

  El::Device get_device_allocation() const override { return Device; }

  bool can_run_inplace() const override { return false; }

  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | WEIGHTS | PREV_ACTIVATIONS;
  }

  void setup_dims() override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  deconvolution_layer();

  std::vector<int> get_kernel_dims() const override;
  void fp_compute() override;
  void bp_compute() override;

#ifdef LBANN_HAS_DISTCONV
  friend class deconvolution_distconv_adapter<TensorDataType, Layout, Device>;

protected:
  void setup_distconv_adapter() override;
  bool is_distconv_supported() const override;
#endif // LBANN_HAS_DISTCONV

private:
  /** @brief Padding for output tensor
   *  @details The output tensor size is ambiguous when the
   *  convolution is strided. If this is not set, then we will
   *  output the smallest valid output tensor.
   */
  std::vector<int> m_output_pads;
};

LBANN_DEFINE_LAYER_BUILDER(deconvolution);

#ifndef LBANN_DECONVOLUTION_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class deconvolution_layer<T,                                 \
                                            data_layout::DATA_PARALLEL,        \
                                            Device>;

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_DECONVOLUTION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_DECONVOLUTION_HPP_INCLUDED
