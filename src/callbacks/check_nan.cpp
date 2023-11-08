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

#include "lbann/callbacks/check_nan.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/exception.hpp"

#include "lbann/utils/serialize.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

namespace lbann {
namespace callback {

namespace {

/** Check whether a matrix contains a NaN.
 *  If a NaN entry is detected, return true and output the local entry
 *  position in row and col. mat is assumed to be a CPU matrix.
 */
template <typename TensorDataType>
bool has_nan(const El::AbstractDistMatrix<TensorDataType>& mat,
             El::Int& row,
             El::Int& col)
{
  row = -1;
  col = -1;
  const auto& local_mat = mat.LockedMatrix();
  for (El::Int j = 0; j < local_mat.Width(); ++j) {
    for (El::Int i = 0; i < local_mat.Height(); ++i) {
      if (std::isnan(local_mat(i, j))) {
        row = i;
        col = j;
        return true;
      }
    }
  }
  return false;
}

/** Check whether a matrix contains an inf.
 *  If an inf entry is detected, return true and output the entry
 *  position in row and col. mat is assumed to be a CPU matrix.
 */
template <typename TensorDataType>
bool has_inf(const El::AbstractDistMatrix<TensorDataType>& mat,
             El::Int& row,
             El::Int& col)
{
  row = -1;
  col = -1;
  const auto& local_mat = mat.LockedMatrix();
  for (El::Int j = 0; j < local_mat.Width(); ++j) {
    for (El::Int i = 0; i < local_mat.Height(); ++i) {
      if (std::isinf(local_mat(i, j))) {
        row = i;
        col = j;
        return true;
      }
    }
  }
  return false;
}

struct DefaultErrorReporter
{
  template <typename... Ts>
  void DispatchError(Ts&&...)
  {
    LBANN_ERROR("Unable to dispatch functor.");
  }

  template <typename... Ts>
  void DeductionError(Ts&&...)
  {
    LBANN_ERROR("Unable to deduce an argument type.");
  }
};

struct DumpLayerFunctor : DefaultErrorReporter
{
  model* m;
  SGDExecutionContext const& c;

  DumpLayerFunctor(model* arg_m, SGDExecutionContext const& arg_c)
    : m(arg_m), c(arg_c)
  {}

  template <typename TensorDataType>
  void operator()(data_type_layer<TensorDataType> const& dtl)
  {
    const std::string prefix =
      build_string("model",
                   m->get_comm()->get_trainer_rank(),
                   "-rank",
                   m->get_comm()->get_rank_in_trainer(),
                   "-epoch",
                   c.get_epoch(),
                   "-step",
                   c.get_step(),
                   "-",
                   dtl.get_name(),
                   "-");
    for (int i = 0; i < dtl.get_num_children(); ++i) {
      El::Write(dtl.get_local_activations(i),
                prefix + "Activations" + std::to_string(i),
                El::ASCII);
    }
    for (int i = 0; i < dtl.get_num_parents(); ++i) {
      El::Write(dtl.get_local_error_signals(i),
                prefix + "ErrorSignal" + std::to_string(i),
                El::ASCII);
    }
  }
}; // struct DumpLayerFunctor

struct DumpWeightsFunctor : DefaultErrorReporter
{
  model* m;
  SGDExecutionContext const& c;

  DumpWeightsFunctor(model* arg_m, SGDExecutionContext const& arg_c)
    : m(arg_m), c(arg_c)
  {}

  template <typename TensorDataType>
  void operator()(data_type_weights<TensorDataType>& dtw)
  {
    const std::string prefix =
      build_string("model",
                   m->get_comm()->get_trainer_rank(),
                   "-rank",
                   m->get_comm()->get_rank_in_trainer(),
                   "-epoch",
                   c.get_epoch(),
                   "-step",
                   c.get_step(),
                   "-",
                   dtw.get_name(),
                   "-");
    El::Write(dtw.get_values().LockedMatrix(), prefix + "Weights", El::ASCII);
    auto* opt = dtw.get_optimizer();
    if (opt != nullptr) {
      auto grad = opt->get_gradient();
      El::Write(grad->LockedMatrix(), prefix + "Gradient", El::ASCII);
    }
  }
}; // struct DumpWeightsFunctor

template <typename T>
using SingleTypeDataTypeLayer = data_type_layer<T, T>;

/** Dump the local network matrices for debugging.
 *  Dump only the local matrices because not every rank will
 *  necessarily have bad data, and the check is purely local.
 */
void dump_network(model* m)
{
  using ValidFPTypes = supported_layer_data_type;

  const auto& c =
    dynamic_cast<SGDExecutionContext&>(m->get_execution_context());
  for (auto* l : m->get_layers()) {
    using LayerTypes =
      h2::meta::tlist::ExpandTL<SingleTypeDataTypeLayer, ValidFPTypes>;
    using Dispatcher = h2::multimethods::
      SwitchDispatcher<DumpLayerFunctor, void, Layer, LayerTypes>;
    Dispatcher::Exec(DumpLayerFunctor(m, c), *l);
  }
  for (auto* w : m->get_weights()) {
    using WeightsTypes =
      h2::meta::tlist::ExpandTL<data_type_weights, ValidFPTypes>;
    using Dispatcher = h2::multimethods::
      SwitchDispatcher<DumpWeightsFunctor, void, weights, WeightsTypes>;
    Dispatcher::Exec(DumpWeightsFunctor(m, c), *w);
  }
}
} // namespace

template <class Archive>
void check_nan::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)));
}

void check_nan::write_specific_proto(lbann_data::Callback& proto) const
{
  proto.mutable_check_nan();
}

void check_nan::on_forward_prop_end(model* m, Layer* l)
{
  using proxy_type =
    El::AbstractDistMatrixReadDeviceProxy<DataType, El::Device::CPU>;

  if (!m || !l)
    LBANN_ERROR("Model or layer pointer is null.");

  const auto& num_outputs = l->get_num_children();
  for (int i = 0; i < num_outputs; ++i) {
    El::Int row, col;
    auto const& dtl = dynamic_cast<data_type_layer<DataType>&>(*l);
    proxy_type mat_proxy(dtl.get_activations(i));
    if (has_nan(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      std::string activation_id =
        (num_outputs > 1 ? std::to_string(i) + " " : "");
      LBANN_ERROR("rank ",
                  m->get_comm()->get_rank_in_world(),
                  ": "
                  "local entry (",
                  row,
                  ",",
                  col,
                  ") is NaN "
                  "in activations ",
                  activation_id,
                  "of layer \"",
                  l->get_name(),
                  "\"");
    }
    if (has_inf(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      std::string activation_id =
        (num_outputs > 1 ? std::to_string(i) + " " : "");
      LBANN_ERROR("rank ",
                  m->get_comm()->get_rank_in_world(),
                  ": "
                  "local entry (",
                  row,
                  ",",
                  col,
                  ") is inf "
                  "in activations ",
                  activation_id,
                  "of layer \"",
                  l->get_name(),
                  "\"");
    }
  }
}

void check_nan::on_backward_prop_end(model* m, Layer* l)
{
  using proxy_type =
    El::AbstractDistMatrixReadDeviceProxy<DataType, El::Device::CPU>;
  const auto& num_inputs = l->get_num_parents();
  for (int i = 0; i < num_inputs; ++i) {
    El::Int row, col;
    auto const& dtl = dynamic_cast<data_type_layer<DataType>&>(*l);
    proxy_type mat_proxy(dtl.get_error_signals(i));
    if (has_nan(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      std::string signal_id = (num_inputs > 1 ? std::to_string(i) + " " : "");
      LBANN_ERROR("rank ",
                  m->get_comm()->get_rank_in_world(),
                  ": "
                  "local entry (",
                  row,
                  ",",
                  col,
                  ") is NAN "
                  "in error signals ",
                  signal_id,
                  " of layer \"",
                  l->get_name(),
                  "\"");
    }
    if (has_inf(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      std::string signal_id = (num_inputs > 1 ? std::to_string(i) + " " : "");
      LBANN_ERROR("rank ",
                  m->get_comm()->get_rank_in_world(),
                  ": "
                  "local entry (",
                  row,
                  ",",
                  col,
                  ") is inf "
                  "in error signals ",
                  signal_id,
                  " of layer \"",
                  l->get_name(),
                  "\"");
    }
  }
}

void check_nan::on_backward_prop_end(model* m)
{
  using proxy_type =
    El::AbstractDistMatrixReadDeviceProxy<DataType, El::Device::CPU>;
  for (weights* w : m->get_weights()) {
    auto& dtw = dynamic_cast<data_type_weights<DataType>&>(*w);
    auto* opt = dtw.get_optimizer();
    if (opt != nullptr) {
      El::Int row, col;
      auto grad = opt->get_gradient();
      proxy_type mat_proxy(*grad);
      if (has_nan(mat_proxy.GetLocked(), row, col)) {
        dump_network(m);
        LBANN_ERROR("rank ",
                    m->get_comm()->get_rank_in_world(),
                    ": "
                    "local entry (",
                    row,
                    ",",
                    col,
                    ") is NaN "
                    "in gradient w.r.t. weights \"",
                    w->get_name(),
                    "\"");
      }
      if (has_inf(mat_proxy.GetLocked(), row, col)) {
        dump_network(m);
        LBANN_ERROR("rank ",
                    m->get_comm()->get_rank_in_world(),
                    ": "
                    "local entry (",
                    row,
                    ",",
                    col,
                    ") is inf "
                    "in gradient w.r.t. weights \"",
                    w->get_name(),
                    "\"");
      }
    }
  }
}

void check_nan::on_batch_end(model* m)
{
  using proxy_type =
    El::AbstractDistMatrixReadDeviceProxy<DataType, El::Device::CPU>;
  for (weights* w : m->get_weights()) {
    auto& dtw = dynamic_cast<data_type_weights<DataType>&>(*w);
    El::Int row, col;
    proxy_type mat_proxy(dtw.get_values());
    if (has_nan(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      LBANN_ERROR("rank ",
                  m->get_comm()->get_rank_in_world(),
                  ": "
                  "local entry (",
                  row,
                  ",",
                  col,
                  ") is NaN "
                  "in weights \"",
                  w->get_name(),
                  "\"");
    }
    if (has_inf(mat_proxy.GetLocked(), row, col)) {
      dump_network(m);
      LBANN_ERROR("rank ",
                  m->get_comm()->get_rank_in_world(),
                  ": "
                  "local entry (",
                  row,
                  ",",
                  col,
                  ") is inf "
                  "in weights \"",
                  w->get_name(),
                  "\"");
    }
  }
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::check_nan
#define LBANN_CLASS_LIBNAME callback_check_nan
#include <lbann/macros/register_class_with_cereal.hpp>
