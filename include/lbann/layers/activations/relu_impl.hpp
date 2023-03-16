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

#ifndef LBANN_LAYER_ACTIVATION_RELU_IMPL_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_RELU_IMPL_HPP_INCLUDED

#include "lbann/layers/activations/relu.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

template <typename T, data_layout L, El::Device D>
void relu_layer<T, L, D>::write_specific_proto(lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_relu();
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
relu_distconv_adapter<TensorDataType, T_layout, Dev>&
relu_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<relu_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<const relu_layer<TensorDataType, T_layout, Dev>&>(*this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const relu_distconv_adapter<TensorDataType, T_layout, Dev>&
relu_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const
{
  return dynamic_cast<
    const relu_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void relu_distconv_adapter<TensorDataType, T_layout, Dev>::setup_distributions(
  tensor_overlap_constraints& constraints)
{
  data_type_distconv_adapter<TensorDataType>::setup_distributions(constraints);

  auto& x = this->get_prev_activations_dist();
  auto& y = this->get_activations_dist();
  auto& dx = this->get_error_signals_dist();
  auto& dy = this->get_prev_error_signals_dist();

  // x == dx
  constraints.mark_equivalent(x, dx);
  // y == dy
  constraints.mark_equivalent(y, dy);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void relu_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
  size_t workspace_capacity)
{
  m_relu = std::make_unique<dc::ReLU>(dc::get_backend());
  m_relu->setup(this->get_prev_activations(),
                this->get_activations(),
                this->get_error_signals(),
                this->get_prev_error_signals());
}
#endif // LBANN_HAS_DISTCONV

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_RELU_IMPL_HPP_INCLUDED
