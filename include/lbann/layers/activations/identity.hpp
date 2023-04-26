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

#ifndef LBANN_LAYERS_ACTIVATIONS_IDENTITY_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_IDENTITY_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/distconv.hpp"
#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/transform_tensor.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_DNN_LIB

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
namespace dc_backend = ::distconv::backend;

template <typename TensorDataType, data_layout Layout, El::Device Device>
class identity_distconv_adapter
  : public data_type_distconv_adapter<TensorDataType>
{
public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  identity_distconv_adapter(Layer& layer)
    : data_type_distconv_adapter<TensorDataType>(layer)
  {}
  virtual ~identity_distconv_adapter() = default;
  std::unique_ptr<TensorDevType> setup_activations_i(int index) const override;
  std::unique_ptr<TensorDevType>
  setup_error_signals_i(int index) const override;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Output the input tensor
 *
 *  This layer is very cheap since it just involves setting up tensor
 *  views.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class identity_layer : public data_type_layer<TensorDataType>
{
public:
  identity_layer() : data_type_layer<TensorDataType>(nullptr) {}
  identity_layer* copy() const override { return new identity_layer(*this); }
  std::string get_type() const override { return "identity"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

#ifdef LBANN_HAS_ONNX
  std::string get_onnx_op_type() const override { return "Identity"; }
#endif // LBANN_HAS_ONNX

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void setup_dims() override
  {
    data_type_layer<TensorDataType>::setup_dims();
    this->set_output_dims(this->get_input_dims());
  }
  void fp_setup_outputs() override
  {
#ifdef LBANN_HAS_DISTCONV
    // Copy activations when distconv is enabled

    if (this->distconv_enabled()) {
      data_type_layer<TensorDataType>::fp_setup_outputs();

      return;
    }
#endif // LBANN_HAS_DISTCONV
    El::LockedView(this->get_activations(), this->get_prev_activations());
  }
  void bp_setup_gradient_wrt_inputs() override
  {
#ifdef LBANN_HAS_DISTCONV
    // Copy gradients wrt inputs when distconv is enabled

    if (this->distconv_enabled()) {
      data_type_layer<TensorDataType>::bp_setup_gradient_wrt_inputs();

      return;
    }
#endif // LBANN_HAS_DISTCONV
    El::LockedView(this->get_error_signals(), this->get_prev_error_signals());
  }
  void fp_compute() override {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      auto& adapter = this->get_distconv_adapter();

      if (!m_equal_overlap_set) {
        const auto& prev_activations_overlap = adapter.get_prev_activations_dist().get_overlap();
        const auto& activations_overlap = adapter.get_activations_dist().get_overlap();

        m_equal_overlap = true;
        assert_eq(prev_activations_overlap.length(), activations_overlap.length());
        for (int i = 0; i < prev_activations_overlap.length(); i++) {
          if (prev_activations_overlap[i] != activations_overlap[i]) {
            m_equal_overlap = false;
            m_xdesc.create();
            m_ydesc.create();
            m_dxdesc.create();
            m_dydesc.create();
            break;
          }
        }
        m_equal_overlap_set = true;
      }

      if (m_equal_overlap) return;

      auto& prev_activations = adapter.get_prev_activations();
      auto& activations = adapter.get_activations();

      auto xdesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_xdesc.get());
      auto ydesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_ydesc.get());
      dc_backend::setup_tensor_descriptor(xdesc, prev_activations,
                                          prev_activations.get_local_shape());
      dc_backend::setup_tensor_descriptor(ydesc, activations,
                                          activations.get_local_shape());

      using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
      LibScalingParamT alpha = 1;
      LibScalingParamT beta = 0;
      dnn_lib::transform_tensor(alpha,
                                m_xdesc,
                                prev_activations.get_const_base_ptr(),
                                beta,
                                m_ydesc,
                                activations.get_base_ptr(),
                                dc::get_backend().get_handle());
    }
#endif // LBANN_HAS_DISTCONV
  }
  void bp_compute() override {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      if (m_equal_overlap) return;

      auto& adapter = this->get_distconv_adapter();

      auto& prev_error_signals = adapter.get_prev_error_signals();
      auto& error_signals = adapter.get_error_signals();

      auto dxdesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_dxdesc.get());
      auto dydesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_dydesc.get());
      dc_backend::setup_tensor_descriptor(dxdesc, error_signals,
                                          error_signals.get_local_shape());
      dc_backend::setup_tensor_descriptor(dydesc, prev_error_signals,
                                          prev_error_signals.get_local_shape());

      using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
      LibScalingParamT alpha = 1;
      LibScalingParamT beta = 0;
      dnn_lib::transform_tensor(alpha,
                                m_dydesc,
                                prev_error_signals.get_const_base_ptr(),
                                beta,
                                m_dxdesc,
                                error_signals.get_base_ptr(),
                                dc::get_backend().get_handle());
    }
#endif // LBANN_HAS_DISTCONV
  }
#ifdef LBANN_HAS_DISTCONV
protected:
  bool is_distconv_supported() const override
  {
    return Device == El::Device::GPU && Layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter() override
  {
    this->get_distconv_adapter_ptr() = std::make_unique<
      identity_distconv_adapter<TensorDataType, Layout, Device>>(*this);
  }
  dnn_lib::TensorDescriptor m_xdesc;
  dnn_lib::TensorDescriptor m_ydesc;
  dnn_lib::TensorDescriptor m_dxdesc;
  dnn_lib::TensorDescriptor m_dydesc;
  bool m_equal_overlap;
  bool m_equal_overlap_set = false;
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_IDENTITY_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class identity_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class identity_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_IDENTITY_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_IDENTITY_HPP_INCLUDED
