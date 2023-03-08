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

#ifndef LBANN_LAYERS_ACTIVATIONS_SOFTMAX_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_SOFTMAX_IMPL_HPP_INCLUDED

#include "lbann/layers/activations/softmax.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

template <typename T, data_layout L, El::Device D>
void softmax_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_softmax();
  switch (m_mode) {
  case softmax_mode::INSTANCE:
    msg->set_softmax_mode("instance");
    break;
  case softmax_mode::CHANNEL:
    msg->set_softmax_mode("channel");
    break;
  default:
    msg->set_softmax_mode("invalid");
  }
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
softmax_distconv_adapter<TensorDataType, T_layout, Dev>&
softmax_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<softmax_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<const softmax_layer<TensorDataType, T_layout, Dev>&>(*this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const softmax_distconv_adapter<TensorDataType, T_layout, Dev>&
softmax_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const
{
  return dynamic_cast<
    const softmax_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void softmax_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_distributions(tensor_overlap_constraints& constraints)
{
  data_type_distconv_adapter<TensorDataType>::setup_distributions(constraints);
  // No overlap supported yet
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

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void softmax_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
  size_t workspace_capacity)
{
  auto& l =
    dynamic_cast<softmax_layer<TensorDataType, T_layout, Dev>&>(this->layer());
  m_softmax = std::make_unique<dc::Softmax>(dc::get_backend());
  auto mode = l.m_mode == softmax_mode::INSTANCE
                ? ::distconv::SoftmaxMode::INSTANCE
                : ::distconv::SoftmaxMode::CHANNEL;
  m_softmax->setup(this->get_prev_activations(), mode);
}
#endif // LBANN_HAS_DISTCONV

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_SOFTMAX_IMPL_HPP_INCLUDED
