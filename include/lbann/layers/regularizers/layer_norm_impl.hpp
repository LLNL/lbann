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

#ifndef LBANN_LAYER_REGULARIZER_LAYER_NORM_IMPL_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_LAYER_NORM_IMPL_HPP_INCLUDED

#include "lbann/layers/regularizers/layer_norm.hpp"

#ifdef LBANN_HAS_DISTONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif

namespace lbann{


// =========================================================
// Implementation
// =========================================================

template <typename T, data_layout L, El::Device D>
void layer_norm_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_layer_norm();
  msg->mutable_epsilon()->set_value(m_epsilon);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType, Layout, Device>::layer_norm_layer(
  TensorDataType epsilon)
  : data_type_layer<TensorDataType>(nullptr), m_epsilon(epsilon)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType, Layout, Device>::layer_norm_layer(
  const layer_norm_layer<TensorDataType, Layout, Device>& other)
  : data_type_layer<TensorDataType>(other),
    m_epsilon(other.m_epsilon),
    m_statistics(other.m_statistics ? other.m_statistics->Copy() : nullptr),
    m_statistics_gradient(other.m_statistics_gradient
                            ? other.m_statistics_gradient->Copy()
                            : nullptr)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType, Layout, Device>&
layer_norm_layer<TensorDataType, Layout, Device>::operator=(
  const layer_norm_layer<TensorDataType, Layout, Device>& other)
{
  data_type_layer<TensorDataType>::operator=(other);
  m_epsilon = other.m_epsilon;
  m_statistics.reset(other.m_statistics ? other.m_statistics->Copy() : nullptr);
  m_statistics_gradient.reset(other.m_statistics_gradient
                                ? other.m_statistics_gradient->Copy()
                                : nullptr);
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType, Layout, Device>*
layer_norm_layer<TensorDataType, Layout, Device>::copy() const
{
  return new layer_norm_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string layer_norm_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "layer norm";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
layer_norm_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
layer_norm_layer<TensorDataType, Layout, Device>::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
layer_norm_layer<TensorDataType, Layout, Device>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Epsilon", m_epsilon);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  this->set_output_dims(this->get_input_dims());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType, Layout, Device>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);
  auto dist = this->get_prev_activations().DistData();
  dist.colDist = El::STAR;
  m_statistics.reset(AbsDistMatrixType::Instantiate(dist));
  m_statistics_gradient.reset(AbsDistMatrixType::Instantiate(dist));
}


#ifdef LBANN_HAS_DISTCONV

// =============================================================
// DistConv-enabled Scatter member functions
// =============================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
bool layer_norm_layer<TensorDataType, Layout, Device>::is_distconv_supported()
  const
{
  return Device == El::Device::GPU && Layout == data_layout::DATA_PARALLEL;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType, Layout, Device>::setup_distconv_adapter(
  const DataReaderMetaData& dr_metadata)
{
  this->get_distconv_adapter_ptr() = std::make_unique<
    layer_norm_distconv_adapter<TensorDataType, Layout, Device>>(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
const layer_norm_distconv_adapter<TensorDataType, Layout, Device>&
layer_norm_layer<TensorDataType, Layout, Device>::get_distconv_adapter() const
{
  return dynamic_cast<
    const layer_norm_distconv_adapter<TensorDataType, Layout, Device>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_distconv_adapter<TensorDataType, Layout, Device>&
layer_norm_layer<TensorDataType, Layout, Device>::get_distconv_adapter()
{
  return const_cast<
    layer_norm_distconv_adapter<TensorDataType, Layout, Device>&>(
    static_cast<const layer_norm_layer<TensorDataType, Layout, Device>&>(*this)
      .get_distconv_adapter());

// =============================================================
// LayerNorm DistConv Adapter implementation
// =============================================================

  template <typename TensorDataType, data_layout Layout, El::Device Device>
  void layer_norm_distconv_adapter<TensorDataType, Layout, Device>::
    setup_distributions(tensor_overlap_constraints & constraints)
  {
    data_type_distconv_adapter<TensorDataType>::setup_distributions(
      constraints);
    // no overlap needed
    for (auto& d : this->m_prev_activations_dists) {
      d.clear_overlap();
      constraints.mark_updated(d);
      constraints.mark_invariant(d);
    }
    for (auto& d : this->m_activations_dists) {
      d.clear_overlap();
      constraints.mark_updated(d);
      constraints.mark_invariant(d);
    }
    for (auto& d : this->m_prev_error_signals_dists) {
      d.clear_overlap();
      constraints.mark_updated(d);
      constraints.mark_invariant(d);
    }
    for (auto& d : this->m_error_signals_dists) {
      d.clear_overlap();
      constraints.mark_updated(d);
      constraints.mark_invariant(d);
    }
  }

  template <typename TensorDataType, data_layout Layout, El::Device Device>
  void layer_norm_distconv_adapter<TensorDataType, Layout, Device>::setup_layer(
    size_t workspace_capacity)
  {
    data_type_distconv_adapter<TensorDataType>::setup_layer(workspace_capacity);
    auto& layer = dynamic_cast<
      channelwise_fully_connected_layer<TensorDataType, Layout, Device>&>(
      this->layer());
    const auto max_mini_batch_size =
      layer.get_model()->m_max_mini_batch_size_distconv;

    m_layer_norm_operator =
      make_unique<dc::LayerNormalization<TensorDataType>>(dc::get_backend(),
                                                          layer.m_epsilon,
                                                          max_mini_batch_size);
  }

#endif LBANN_HAS_DISTCONV
} // namespace lbann
#endif // LBANN_LAYER_REGULARIZER_LAYER_NORM_IMPL_HPP_INCLUDED