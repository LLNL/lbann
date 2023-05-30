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

#ifndef LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_IMPL_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_IMPL_HPP_INCLUDED

#include "lbann/layers/regularizers/batch_normalization.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

template <typename T, data_layout L, El::Device D>
void batch_normalization_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_batch_normalization();
  msg->set_decay(m_decay);
  msg->set_epsilon(m_epsilon);
  msg->set_statistics_group_size(m_statistics_group_size);
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>&
batch_normalization_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
  const
{
  return dynamic_cast<
    const batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>&
batch_normalization_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<
    batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<
      const batch_normalization_layer<TensorDataType, T_layout, Dev>&>(*this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
  get_per_channel_stat_shape() const
{
  auto& l = dynamic_cast<
    const batch_normalization_layer<TensorDataType, T_layout, Dev>&>(
    this->layer());
  const int num_channels = this->get_activations_shape()[dc::get_channel_dim()];
  // Sanity check that the shared tensors have the correct shape
  assert_ne(num_channels, 0);
  assert_eq(l.m_mean_and_var->Matrix().Width() *
              l.m_mean_and_var->Matrix().Height(),
            num_channels * 2);
  dc::Shape per_channel_stat_shape(dc::get_num_dims(l), 1);
  per_channel_stat_shape[dc::get_channel_dim()] = num_channels;
  return per_channel_stat_shape;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Dist batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
  get_per_channel_stat_dist(const dc::Dist& input_dist) const
{
  auto shared_dist = dc::Dist::make_distribution(input_dist.get_locale_shape());
  auto split_shape = input_dist.get_split_shape();
  // set all dimensions to be 1 except for the channel dimension
  auto pc = split_shape[-2];
  // set all elements to 1
  split_shape = 1;
  split_shape[-2] = pc;
  shared_dist.set_split_shape(split_shape);

  return shared_dist;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_fp_tensors()
{
  data_type_distconv_adapter<TensorDataType>::setup_fp_tensors();

  auto& l =
    static_cast<batch_normalization_layer<TensorDataType, T_layout, Dev>&>(
      this->layer());
  const auto& input_dist = this->get_prev_activations_dist();

  const auto per_channel_stat_shape = get_per_channel_stat_shape();
  const auto shared_dist = get_per_channel_stat_dist(input_dist);

  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  // mean
  m_mean = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(m_mean, l.m_mean_v->Buffer()));
  // var
  m_var = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(m_var, l.m_var_v->Buffer()));
  // scale: view to weights[0]
  m_scale = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  // bias: view to weights[1]
  m_bias = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  // running_mean: view to weights[2]
  m_running_mean = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  // running_var: view to weights[3]
  m_running_var = TensorDevType(per_channel_stat_shape, loc, shared_dist);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_bp_tensors()
{
  data_type_distconv_adapter<TensorDataType>::setup_bp_tensors();

  const auto& prev_error_signal_dist = this->get_prev_error_signals_dist();
  auto& l =
    static_cast<batch_normalization_layer<TensorDataType, T_layout, Dev>&>(
      this->layer());

  const auto per_channel_stat_shape = get_per_channel_stat_shape();
  const auto shared_dist = get_per_channel_stat_dist(prev_error_signal_dist);

  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  // scale_gradient
  m_scale_gradient = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(m_scale_gradient, l.m_scale_gradient->Buffer()));
  // bias_gradient
  m_bias_gradient = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(m_bias_gradient, l.m_bias_gradient->Buffer()));
  // mean_gradient
  m_mean_gradient = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(m_mean_gradient, l.m_mean_gradient_v->Buffer()));
  // var_gradient
  m_var_gradient = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(m_var_gradient, l.m_var_gradient_v->Buffer()));
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_layer(size_t workspace_capacity)
{
  auto& l =
    dynamic_cast<batch_normalization_layer<TensorDataType, T_layout, Dev>&>(
      this->layer());
  bool global_stats;
  if (l.m_statistics_group_size == 0) {
    global_stats = true;
  }
  else if (l.m_statistics_group_size == 1) {
    global_stats = false;
  }
  else {
    LBANN_ERROR("statistics_group_size must be either 0 or 1 for now.");
  }

  m_bn = std::make_unique<dc::BatchNormalization<TensorDataType>>(
    dc::get_backend(),
    dc::get_num_dims(l),
    l.m_decay,
    l.m_epsilon,
    global_stats);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
std::unique_ptr<typename batch_normalization_distconv_adapter<TensorDataType,
                                                              T_layout,
                                                              Dev>::TensorDevType>
batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_error_signals_i(int index) const
{
  assert_eq(index, 0);
  if (this->layer().get_parent_layer().get_backprop_requirements() & ACTIVATIONS) {
    return data_type_distconv_adapter<TensorDataType>::setup_error_signals_i(0);
  }
  const auto& prev_activations = this->get_prev_activations(0);
  return std::make_unique<TensorDevType>(prev_activations);
}
#endif // LBANN_HAS_DISTCONV

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_IMPL_HPP_INCLUDED
