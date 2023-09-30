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

#define LBANN_REDUCTION_LAYER_INSTANTIATE
#include "lbann/layers/transform/reduction.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/proto_common.hpp"

#include "lbann/proto/layers.pb.h"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
reduction_layer<TensorDataType, Layout, Device>::reduction_layer(
  reduction_mode mode)
  : data_type_layer<TensorDataType>(nullptr), m_mode(mode)
{
  if (mode == reduction_mode::INVALID) {
    LBANN_ERROR("invalid reduction mode");
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_reduction_layer_from_pbuf(lbann_comm* comm,
                                lbann_data::Layer const& proto_layer)
{
  using LayerType = reduction_layer<TensorDataType, Layout, Device>;
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, reduction);
  const auto& params = proto_layer.reduction();
  const std::string mode_str = params.mode();
  reduction_mode mode = reduction_mode::INVALID;
  if (mode_str == "sum" || mode_str.empty()) {
    mode = reduction_mode::SUM;
  }
  if (mode_str == "mean" || mode_str == "average") {
    mode = reduction_mode::AVERAGE;
  }
  return std::make_unique<LayerType>(mode);
}

template <typename T, data_layout L, El::Device D>
void reduction_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_reduction();
  switch (m_mode) {
  case reduction_mode::SUM:
    msg->set_mode("sum");
    break;
  case reduction_mode::AVERAGE:
    msg->set_mode("mean");
    break;
  default:
    msg->set_mode("invalid");
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void reduction_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();
  this->set_output_dims({1});
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void reduction_layer<TensorDataType, Layout, Device>::fp_compute()
{

  // Constants
  const auto one = El::TypeTraits<TensorDataType>::One();
  const auto zero = El::TypeTraits<TensorDataType>::Zero();

  // Data matrices
  using LocalMat = El::Matrix<TensorDataType, Device>;
  const auto& input = this->get_prev_activations();
  auto& output = this->get_activations();

  // Create workspace buffers
  LocalMat local_reduction, ones;
  const auto& col_comm = input.ColComm();
  const auto col_rank = El::mpi::Rank(col_comm);
  const auto owner_rank = output.RowOwner(0);
  if (col_rank == owner_rank) {
    El::View(local_reduction, output.Matrix());
  }
  else {
    local_reduction.Resize(1, input.LocalWidth());
  }
  El::Ones(ones, input.LocalHeight(), 1);

  // Compute local reductions
  switch (m_mode) {
  case reduction_mode::SUM:
    El::Gemv(El::TRANSPOSE,
             one,
             input.LockedMatrix(),
             ones,
             zero,
             local_reduction);
    break;
  case reduction_mode::AVERAGE:
    El::Gemv(El::TRANSPOSE,
             one / El::To<TensorDataType>(input.Height()),
             input.LockedMatrix(),
             ones,
             zero,
             local_reduction);
    break;
  default:
    LBANN_ERROR("invalid reduction mode");
  }

  // Accumulate local reductions in output matrix
  /// @todo Replace with Reduce when supported in Hydrogen.
  El::AllReduce(local_reduction, col_comm, El::mpi::SUM);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void reduction_layer<TensorDataType, Layout, Device>::bp_compute()
{

  // Constants
  const auto one = El::TypeTraits<TensorDataType>::One();
  const auto zero = El::TypeTraits<TensorDataType>::Zero();

  // Data matrices
  using LocalMat = El::Matrix<TensorDataType, Device>;
  const auto& output_grad = this->get_prev_error_signals();
  auto& input_grad = this->get_error_signals();

  // Create workspace buffers
  LocalMat local_output_grad, ones;
  const auto& col_comm = input_grad.ColComm();
  const auto col_rank = El::mpi::Rank(col_comm);
  const auto owner_rank = output_grad.RowOwner(0);
  if (col_rank == owner_rank) {
    El::LockedView(local_output_grad, output_grad.LockedMatrix());
  }
  else {
    local_output_grad.Resize(1, input_grad.LocalWidth());
  }
  /** @todo (tym1 3/12/21): We are working around a bug in Hydrogen.
   *  Broadcast with Matrix<T,D> is not instatiated. */
  El::Broadcast(
    static_cast<El::AbstractMatrix<TensorDataType>&>(local_output_grad),
    col_comm,
    owner_rank);
  El::Ones(ones, input_grad.LocalHeight(), 1);

  // Populate error signals
  switch (m_mode) {
  case reduction_mode::SUM:
    El::Gemm(El::NORMAL,
             El::NORMAL,
             one,
             ones,
             local_output_grad,
             zero,
             input_grad.Matrix());
    break;
  case reduction_mode::AVERAGE:
    El::Gemm(El::NORMAL,
             El::NORMAL,
             one / El::To<TensorDataType>(input_grad.Height()),
             ones,
             local_output_grad,
             zero,
             input_grad.Matrix());
    break;
  default:
    LBANN_ERROR("invalid reduction mode");
  }
}

#define PROTO_DEVICE(T, Device)                                                \
  template class reduction_layer<T, data_layout::DATA_PARALLEL, Device>;       \
  template class reduction_layer<T, data_layout::MODEL_PARALLEL, Device>;      \
  LBANN_LAYER_BUILDER_ETI(reduction, T, Device)
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
