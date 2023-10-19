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
#ifndef LBANN_LAYERS_OPERATOR_LAYER_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_OPERATOR_LAYER_IMPL_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/operator_layer.hpp"

#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/proto/factories.hpp"
#include "lbann/proto/operator_factory.hpp"
#include "lbann/utils/exception.hpp"

#include "lbann/proto/layers.pb.h"
#include <cereal/types/base_class.hpp>
#include <memory>

namespace lbann {

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
OperatorLayer<InputT, OutputT, Layout, D>::OperatorLayer(lbann_comm& comm,
                                                         OperatorPtr op)
  : DataTypeLayer(&comm)
{
  LBANN_ASSERT(op);
  m_ops.reserve(1);
  m_ops.emplace_back(std::move(op));
  this->m_expected_num_parent_layers = -1; // No limit on parents
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
OperatorLayer<InputT, OutputT, Layout, D>::OperatorLayer(
  lbann_comm& comm,
  std::vector<OperatorPtr> operators)
  : DataTypeLayer(&comm), m_ops{std::move(operators)}
{
  LBANN_ASSERT(m_ops.size() == 1UL); // For starters.
  LBANN_ASSERT(m_ops[0]);
  this->m_expected_num_parent_layers = -1; // No limit on parents
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
OperatorLayer<InputT, OutputT, Layout, D>::OperatorLayer(
  OperatorLayer const& other)
  : DataTypeLayer(other), m_ops{clone_ops(other.m_ops)}
{}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
auto OperatorLayer<InputT, OutputT, Layout, D>::operator=(
  OperatorLayer const& other) -> OperatorLayer&
{
  // This is self-assignment safe
  data_type_layer<InputT, OutputT>::operator=(other);
  m_ops = clone_ops(other.m_ops);
  return *this;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
auto OperatorLayer<InputT, OutputT, Layout, D>::copy() const -> OperatorLayer*
{
  return new OperatorLayer<InputT, OutputT, Layout, D>(*this);
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
std::string OperatorLayer<InputT, OutputT, Layout, D>::get_type() const
{
  return "operator";
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
data_layout OperatorLayer<InputT, OutputT, Layout, D>::get_data_layout() const
{
  return Layout;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
El::Device
OperatorLayer<InputT, OutputT, Layout, D>::get_device_allocation() const
{
  return D;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
bool OperatorLayer<InputT, OutputT, Layout, D>::can_run_inplace() const
{
  return true;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
int OperatorLayer<InputT, OutputT, Layout, D>::get_backprop_requirements() const
{
  // Find the union of all internal operators
  int result = ERROR_SIGNALS;
  for (const auto& op : m_ops) {
    result |= op->get_backprop_requirements();
  }
  return result;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
void OperatorLayer<InputT, OutputT, Layout, D>::fp_compute()
{
  return m_ops[0]->fp_compute(this->get_inputs(), this->get_outputs());
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
void OperatorLayer<InputT, OutputT, Layout, D>::bp_compute()
{
  return m_ops[0]->bp_compute(this->get_inputs(),
                              this->get_grad_wrt_outputs(),
                              this->get_grad_wrt_inputs());
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
description OperatorLayer<InputT, OutputT, Layout, D>::get_description() const
{
  auto desc = DataTypeLayer::get_description();
  for (auto const& op : m_ops)
    desc.add(op->get_description());
  return desc;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
template <typename ArchiveT>
void OperatorLayer<InputT, OutputT, Layout, D>::serialize(ArchiveT& ar)
{
  ar(cereal::base_class<DataTypeLayer>(this), m_ops);
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
OperatorLayer<InputT, OutputT, Layout, D>::OperatorLayer()
  : DataTypeLayer(nullptr)
{
  m_ops.reserve(1);
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
auto OperatorLayer<InputT, OutputT, Layout, D>::clone_ops(
  std::vector<OperatorPtr> const& ops) -> std::vector<OperatorPtr>
{
  std::vector<OperatorPtr> out;
  out.reserve(ops.size());
  for (auto const& x : ops) {
    out.emplace_back(x->clone());
  }
  return out;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
std::vector<size_t>
OperatorLayer<InputT, OutputT, Layout, D>::fix_type(std::vector<int> const& in)
{
  return std::vector<size_t>{cbegin(in), cend(in)};
}

// WARNING: The next 4 functions all assume the minibatch dim is the
// width of the matrix.

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
std::vector<utils::ConstDistTensorView<InputT, D>>
OperatorLayer<InputT, OutputT, Layout, D>::get_inputs() const
{
  auto n_parents = this->get_num_parents();
  std::vector<utils::ConstDistTensorView<InputT, D>> out;
  out.reserve(n_parents);
  for (int p = 0; p < n_parents; ++p) {
    auto const& prev_acts = this->get_prev_activations(p);
    out.emplace_back(prev_acts,
                     splice_dims(prev_acts.Width(), this->get_input_dims(p)));
  }
  return out;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
std::vector<utils::DistTensorView<OutputT, D>>
OperatorLayer<InputT, OutputT, Layout, D>::get_outputs()
{
  auto n_children = this->get_num_children();
  std::vector<utils::DistTensorView<OutputT, D>> out;
  out.reserve(n_children);
  for (int c = 0; c < n_children; ++c) {
    auto& acts = this->get_activations(c);
    out.emplace_back(acts, splice_dims(acts.Width(), this->get_output_dims(c)));
  }
  return out;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
std::vector<utils::ConstDistTensorView<OutputT, D>>
OperatorLayer<InputT, OutputT, Layout, D>::get_grad_wrt_outputs() const
{
  auto n_children = this->get_num_children();
  std::vector<utils::ConstDistTensorView<OutputT, D>> out;
  out.reserve(n_children);
  for (int c = 0; c < n_children; ++c) {
    auto const& prev_sigs = this->get_prev_error_signals(c);
    out.emplace_back(prev_sigs,
                     splice_dims(prev_sigs.Width(), this->get_output_dims(c)));
  }
  return out;
}

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
std::vector<utils::DistTensorView<InputT, D>>
OperatorLayer<InputT, OutputT, Layout, D>::get_grad_wrt_inputs()
{
  auto n_parents = this->get_num_parents();
  std::vector<utils::DistTensorView<InputT, D>> out;
  out.reserve(n_parents);
  for (int p = 0; p < n_parents; ++p) {
    auto& error_sigs = this->get_error_signals(p);
    out.emplace_back(error_sigs,
                     splice_dims(error_sigs.Width(), this->get_input_dims(p)));
  }
  return out;
}

} // namespace lbann

template <typename InputT,
          typename OutputT,
          lbann::data_layout Layout,
          El::Device D>
auto lbann::build_operator_layer_from_pbuf(lbann_comm* comm,
                                           lbann_data::Layer const& msg)
  -> std::unique_ptr<Layer>
{
  using LayerType = OperatorLayer<InputT, OutputT, Layout, D>;
  using OperatorType = Operator<InputT, OutputT, D>;
  using OperatorPtr = std::unique_ptr<OperatorType>;

  LBANN_ASSERT(comm); // Sanity check

  // Build up the list of operators for this layer.
  auto const& params = msg.operator_layer();

  auto const num_ops = params.ops_size();
  std::vector<OperatorPtr> ops;
  ops.reserve(num_ops);
  for (int ii = 0; ii < num_ops; ++ii) {
#ifdef LBANN_DEBUG
    LBANN_ASSERT(proto::resolve_default_datatype(msg.datatype()) ==
                 proto::resolve_default_datatype(params.ops(ii).input_datatype()));
    LBANN_ASSERT(proto::resolve_default_datatype(msg.datatype()) ==
                 proto::resolve_default_datatype(params.ops(ii).output_datatype()));
#endif
    ops.emplace_back(
      proto::construct_operator<InputT, OutputT, D>(params.ops(ii)));
  }
  return std::make_unique<LayerType>(*comm, std::move(ops));
}

#ifndef LBANN_INSTANTIATE_OPERATOR_LAYER
namespace lbann {

#define PROTO_DEVICE(T, D)                                                     \
  extern template class OperatorLayer<T, T, data_layout::DATA_PARALLEL, D>;    \
  extern template class OperatorLayer<T, T, data_layout::MODEL_PARALLEL, D>;   \
  extern template std::unique_ptr<Layer>                                       \
  build_operator_layer_from_pbuf<T, T, data_layout::DATA_PARALLEL, D>(         \
    lbann_comm*,                                                               \
    lbann_data::Layer const&);                                                 \
  extern template std::unique_ptr<Layer>                                       \
  build_operator_layer_from_pbuf<T, T, data_layout::MODEL_PARALLEL, D>(        \
    lbann_comm*,                                                               \
    lbann_data::Layer const&)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
#endif // LBANN_INSTANTIATE_OPERATOR_LAYER
#endif // LBANN_LAYERS_OPERATOR_LAYER_IMPL_HPP_INCLUDED
