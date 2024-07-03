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

#define LBANN_MULTIDIM_REDUCTION_LAYER_INSTANTIATE
#include "lbann/layers/transform/multidim_reduction.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/layers.pb.h"

#ifdef LBANN_HAS_CUTENSOR
#include "lbann/utils/cutensor_support.hpp"
#endif

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
multidim_reduction_layer<TensorDataType, Layout, Device>::
  multidim_reduction_layer(const std::vector<unsigned int>& axes,
                           multidim_reduction_mode mode)
  : data_type_layer<TensorDataType>(nullptr), m_axes(axes), m_mode(mode)
{
  if (mode == multidim_reduction_mode::INVALID) {
    LBANN_ERROR("invalid reduction mode");
  }
#ifndef LBANN_HAS_CUTENSOR
  LBANN_ERROR(
    "For MultiDimReduction to work, LBANN must be compiled with cuTENSOR.");
#endif

  // Sort axes (implementation precondition)
  std::sort(m_axes.begin(), m_axes.end());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_multidim_reduction_layer_from_pbuf(lbann_comm* comm,
                                         lbann_data::Layer const& proto_layer)
{
  using LayerType = multidim_reduction_layer<TensorDataType, Layout, Device>;
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, multidim_reduction);
  const auto& params = proto_layer.multidim_reduction();
  const std::string mode_str = params.mode();
  multidim_reduction_mode mode = multidim_reduction_mode::INVALID;
  if (mode_str == "sum" || mode_str == "add" || mode_str.empty()) {
    mode = multidim_reduction_mode::SUM;
  }
  else if (mode_str == "product" || mode_str == "mul") {
    mode = multidim_reduction_mode::PRODUCT;
  }
  else if (mode_str == "max" || mode_str == "maximum") {
    mode = multidim_reduction_mode::MAX;
  }
  else if (mode_str == "min" || mode_str == "minimum") {
    mode = multidim_reduction_mode::MIN;
  }
  else {
    LBANN_ERROR("Unrecognized reduction type \"", mode_str, "\"");
  }

  std::vector<unsigned int> axes =
    protobuf::to_vector<unsigned int>(params.axes());
  return std::make_unique<LayerType>(axes, mode);
}

template <typename T, data_layout L, El::Device D>
void multidim_reduction_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_multidim_reduction();
  switch (m_mode) {
  case multidim_reduction_mode::SUM:
    msg->set_mode("sum");
    break;
  case multidim_reduction_mode::PRODUCT:
    msg->set_mode("product");
    break;
  case multidim_reduction_mode::MAX:
    msg->set_mode("max");
    break;
  case multidim_reduction_mode::MIN:
    msg->set_mode("min");
    break;
  default:
    msg->set_mode("invalid");
  }
  protobuf::assign_to_repeated(*msg->mutable_axes(), this->m_axes);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void multidim_reduction_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();
#ifdef LBANN_HAS_CUTENSOR
  std::vector<int> dims = this->get_input_dims();
  int original_dims = static_cast<int>(dims.size());

  // Remove reduced dimensions and modes
  m_input_modes = make_modes(dims.size());
  m_output_modes = make_modes(dims.size());
  for (auto iter = m_axes.rbegin(); iter != m_axes.rend(); ++iter) {
    auto axis = *iter;
    if (static_cast<int>(axis) >= original_dims) {
      LBANN_ERROR("Axis ",
                  axis,
                  " is out of bounds for a tensor of rank ",
                  original_dims);
    }
    dims.erase(dims.begin() + axis);
  }
  for (auto const& axis : m_axes) {
    // The extra "- 1" is to avoid reducing the mini-batch dimension
    m_output_modes.erase(m_output_modes.begin() + (original_dims - axis - 1));
  }
  this->set_output_dims(dims);
#endif
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void multidim_reduction_layer<TensorDataType, Layout, Device>::fp_compute()
{
#ifdef LBANN_HAS_CUTENSOR
  // No model parallelism nor CPU
  LBANN_ASSERT(Layout == data_layout::DATA_PARALLEL);
  LBANN_ASSERT(Device == El::Device::GPU);
  if constexpr (Device == El::Device::GPU) {
    // Constants
    const auto one = El::TypeTraits<TensorDataType>::One();
    const auto zero = El::TypeTraits<TensorDataType>::Zero();

    // Data matrices
    using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
    LocalMat const& input =
      static_cast<LocalMat const&>(this->get_prev_activations().LockedMatrix());
    LocalMat& output = static_cast<LocalMat&>(this->get_activations().Matrix());

    // Determine reduction operator
    cutensorOperator_t cutensor_reduce_op;
    switch (m_mode) {
    case multidim_reduction_mode::SUM:
      cutensor_reduce_op = CUTENSOR_OP_ADD;
      break;
      // The following reduction modes are enabled only in forward prop
    case multidim_reduction_mode::PRODUCT:
      cutensor_reduce_op = CUTENSOR_OP_MUL;
      break;
    case multidim_reduction_mode::MAX:
      cutensor_reduce_op = CUTENSOR_OP_MAX;
      break;
    case multidim_reduction_mode::MIN:
      cutensor_reduce_op = CUTENSOR_OP_MIN;
      break;
    default:
      LBANN_ERROR("invalid reduction mode");
    }

    RowMajorDims<int64_t> input_dims(this->get_input_dims());
    RowMajorDims<int64_t> output_dims(this->get_output_dims());
    auto const& input_modes = m_input_modes;
    auto const& output_modes = m_output_modes;

    auto input_desc = get_descriptor(input, input_dims);
    auto output_desc = get_descriptor(output, output_dims);

    // Create workspace buffers
    LocalMat workspace;
    workspace.SetMemoryMode(1);
    auto handle = get_handle_ptr();
    uint64_t wspsize = 0;
    CHECK_CUTENSOR(
      cutensorReductionGetWorkspaceSize(handle,
                                        input.LockedBuffer(),
                                        &input_desc,
                                        input_modes.data(),
                                        output.LockedBuffer(),
                                        &output_desc,
                                        output_modes.data(),
                                        output.LockedBuffer(),
                                        &output_desc,
                                        output_modes.data(),
                                        cutensor_reduce_op,
                                        CUDATypeT<TensorDataType>::compute_type,
                                        &wspsize));
    workspace.Resize(wspsize, 1);

    auto multisync = El::MakeMultiSync(El::SyncInfoFromMatrix(output),
                                       El::SyncInfoFromMatrix(input));

    // Compute reduction locally
    CHECK_CUTENSOR(cutensorReduction(
      handle,
      &one,
      input.LockedBuffer(),
      &input_desc,
      input_modes.data(),
      &zero,
      output.LockedBuffer(),
      &output_desc,
      output_modes.data(),
      output.Buffer(),
      &output_desc,
      output_modes.data(),
      cutensor_reduce_op,
      CUDATypeT<TensorDataType>::compute_type,
      workspace.Buffer(),
      wspsize,
      static_cast<El::SyncInfo<El::Device::GPU>>(multisync).Stream()));
  }
#endif
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void multidim_reduction_layer<TensorDataType, Layout, Device>::bp_compute()
{
#ifdef LBANN_HAS_CUTENSOR
  if constexpr (Device == El::Device::GPU) {
    // Constants
    const auto one = El::TypeTraits<TensorDataType>::One();
    const auto zero = El::TypeTraits<TensorDataType>::Zero();

    // Data matrices
    using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
    LocalMat const& input = static_cast<LocalMat const&>(
      this->get_prev_error_signals().LockedMatrix());
    LocalMat& output =
      static_cast<LocalMat&>(this->get_error_signals().Matrix());

    RowMajorDims<int64_t> input_dims(this->get_output_dims());
    RowMajorDims<int64_t> output_dims(this->get_input_dims());
    auto const& input_modes = m_output_modes;
    auto const& output_modes = m_input_modes;

    auto multisync = El::MakeMultiSync(El::SyncInfoFromMatrix(output),
                                       El::SyncInfoFromMatrix(input));
    auto handle = get_handle_ptr();

    // Determine reduction operator
    switch (m_mode) {
    case multidim_reduction_mode::SUM: {
      // Broadcast dimensions
      auto input_desc = get_descriptor(input, input_dims);
      auto output_desc = get_descriptor(output, output_dims);
      CHECK_CUTENSOR(cutensorElementwiseBinary(
        handle,
        &one,
        input.LockedBuffer(),
        &input_desc,
        input_modes.data(),
        &zero,
        output.LockedBuffer(),
        &output_desc,
        output_modes.data(),
        output.Buffer(),
        &output_desc,
        output_modes.data(),
        CUTENSOR_OP_ADD,
        CUDATypeT<TensorDataType>::value,
        static_cast<El::SyncInfo<El::Device::GPU>>(multisync).Stream()));
      break;
    }
    default:
      LBANN_ERROR("invalid reduction mode, only sum is supported for training");
    }
  }
#endif
}

#define PROTO_DEVICE(T, Device)                                                \
  template class multidim_reduction_layer<T,                                   \
                                          data_layout::DATA_PARALLEL,          \
                                          Device>;                             \
  template class multidim_reduction_layer<T,                                   \
                                          data_layout::MODEL_PARALLEL,         \
                                          Device>;                             \
  LBANN_LAYER_BUILDER_ETI(multidim_reduction, T, Device)
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
