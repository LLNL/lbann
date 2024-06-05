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

#define LBANN_IDENTITY_LAYER_INSTANTIATE
#include "lbann/layers/activations/identity.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

template <typename T, data_layout L, El::Device D>
void identity_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_identity();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void identity_layer<TensorDataType, Layout, Device>::fp_compute()
{
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    auto& adapter = this->get_distconv_adapter();

    if (!m_equal_overlap_set) {
      const auto& prev_activations_overlap =
        adapter.get_prev_activations_dist().get_overlap();
      const auto& activations_overlap =
        adapter.get_activations_dist().get_overlap();

      m_equal_overlap = true;
      assert_eq(prev_activations_overlap.length(),
                activations_overlap.length());
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

    if (m_equal_overlap)
      return;

    auto& prev_activations = adapter.get_prev_activations();
    auto& activations = adapter.get_activations();

    auto xdesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_xdesc.get());
    auto ydesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_ydesc.get());
    dc_backend::setup_tensor_descriptor(xdesc,
                                        prev_activations,
                                        prev_activations.get_local_shape());
    dc_backend::setup_tensor_descriptor(ydesc,
                                        activations,
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

template <typename TensorDataType, data_layout Layout, El::Device Device>
void identity_layer<TensorDataType, Layout, Device>::bp_compute()
{
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    if (m_equal_overlap)
      return;

    auto& adapter = this->get_distconv_adapter();

    auto& prev_error_signals = adapter.get_prev_error_signals();
    auto& error_signals = adapter.get_error_signals();

    auto dxdesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_dxdesc.get());
    auto dydesc = const_cast<dnn_lib::dnnTensorDescriptor_t>(m_dydesc.get());
    dc_backend::setup_tensor_descriptor(dxdesc,
                                        error_signals,
                                        error_signals.get_local_shape());
    dc_backend::setup_tensor_descriptor(dydesc,
                                        prev_error_signals,
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
template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<typename identity_distconv_adapter<TensorDataType,
                                                   Layout,
                                                   Device>::TensorDevType>
identity_distconv_adapter<TensorDataType, Layout, Device>::setup_activations_i(
  int index) const
{
  assert_eq(index, 0);

  const auto& prev_activations_overlap =
    this->get_prev_activations_dist().get_overlap();
  const auto& activations_overlap = this->get_activations_dist().get_overlap();

  assert_eq(prev_activations_overlap.length(), activations_overlap.length());
  for (int i = 0; i < prev_activations_overlap.length(); i++) {
    if (prev_activations_overlap[i] != activations_overlap[i]) {
      return data_type_distconv_adapter<TensorDataType>::setup_activations_i(
        index);
    }
  }

  const auto& prev_activations = this->get_prev_activations(0);
  return std::make_unique<TensorDevType>(prev_activations);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<typename identity_distconv_adapter<TensorDataType,
                                                   Layout,
                                                   Device>::TensorDevType>
identity_distconv_adapter<TensorDataType, Layout, Device>::
  setup_error_signals_i(int index) const
{
  assert_eq(index, 0);

  const auto& prev_error_signals_overlap =
    this->get_prev_error_signals_dist().get_overlap();
  const auto& error_signals_overlap =
    this->get_error_signals_dist().get_overlap();

  assert_eq(prev_error_signals_overlap.length(),
            error_signals_overlap.length());
  for (int i = 0; i < prev_error_signals_overlap.length(); i++) {
    if (prev_error_signals_overlap[i] != error_signals_overlap[i]) {
      return data_type_distconv_adapter<TensorDataType>::setup_error_signals_i(
        index);
    }
  }

  const auto& prev_error_signals = this->get_prev_error_signals(0);
  return std::make_unique<TensorDevType>(prev_error_signals);
}
#endif // LBANN_HAS_DISTCONV

#define PROTO_DEVICE(T, Device)                                                \
  template class identity_layer<T, data_layout::DATA_PARALLEL, Device>;        \
  template class identity_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
