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

#ifndef LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann/utils/distconv.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "distconv/dnn_backend/relu.hpp"
#include "lbann/utils/distconv.hpp"
#endif

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
namespace dc {
using Backend = ::distconv::BackendDNNLib;
using ReLU = ::distconv::ReLU<Backend>;
} // namespace dc

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class relu_distconv_adapter : public data_type_distconv_adapter<TensorDataType>
{
public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  relu_distconv_adapter(Layer& layer)
    : data_type_distconv_adapter<TensorDataType>(layer)
  {}
  virtual ~relu_distconv_adapter() = default;
  void setup_distributions(tensor_overlap_constraints& constraints) override;
  void setup_layer(size_t workspace_capacity) override;
  std::unique_ptr<dc::ReLU> m_relu;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Rectified linear unit
 *
 *  \f[ ReLU(x) = \text{max}(x, 0) \f]
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class relu_layer : public data_type_layer<TensorDataType>
{
public:
  relu_layer() : data_type_layer<TensorDataType>(nullptr) {}
  relu_layer(lbann_comm* comm) : data_type_layer<TensorDataType>(comm) {}
  relu_layer* copy() const override { return new relu_layer(*this); }
  std::string get_type() const override { return "ReLU"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | ACTIVATIONS;
  }

#ifdef LBANN_HAS_ONNX
  std::string get_onnx_op_type() const override { return "Relu"; }
#endif // LBANN_HAS_ONNX

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void fp_compute() override;
  void bp_compute() override;
#ifdef LBANN_HAS_DISTCONV
  bool is_distconv_supported() const override
  {
    return Dev == El::Device::GPU && T_layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override
  {
    this->get_distconv_adapter_ptr() =
      std::make_unique<relu_distconv_adapter<TensorDataType, T_layout, Dev>>(
        *this);
  }
  relu_distconv_adapter<TensorDataType, T_layout, Dev>&
  get_distconv_adapter() override;
  const relu_distconv_adapter<TensorDataType, T_layout, Dev>&
  get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_RELU_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class relu_layer<T, data_layout::DATA_PARALLEL, Device>;     \
  extern template class relu_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_RELU_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
