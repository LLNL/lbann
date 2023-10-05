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

#ifndef LBANN_LAYERS_TRANSFORM_SLICE_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_SLICE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/protobuf.hpp"

namespace lbann {

/** @brief Slice tensor along a specified dimension
 *
 *  The tensor is split along one dimension at user-specified points,
 *  and each child layer recieves one piece.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class slice_layer : public data_type_layer<TensorDataType>
{
public:
  slice_layer(lbann_comm* comm);
  slice_layer(const slice_layer& other) = default;
  slice_layer& operator=(const slice_layer& other) = default;

  slice_layer* copy() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

  description get_description() const override;

  void setup_slice_points(size_t slice_dim, std::vector<size_t> slice_points)
  {
    m_slice_dim = slice_dim;
    m_slice_points = std::move(slice_points);
  }

  void setup_slice_points(size_t slice_dim,
                          bool set_slice_points_from_data_reader,
                          const slice_points_mode var_category)
  {
    m_slice_dim = slice_dim;
    m_set_slice_points_from_data_reader = set_slice_points_from_data_reader;
    m_var_category = var_category;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  El::SyncInfo<Device> syncSubGridCommunication = El::SyncInfo<Device>();

  friend class cereal::access;
  slice_layer() : slice_layer(nullptr) {}

  void setup_dims() override;

  void fp_setup_outputs() override;
  void bp_setup_gradient_wrt_inputs() override;
  void fp_compute() override;
  void bp_compute() override;
  void fp_compute_subgrid();
  void bp_compute_subgrid();

private:
  /** Tensor dimension to slice. */
  size_t m_slice_dim;
  /** Slice points for each child layer. */
  std::vector<size_t> m_slice_points;
  /** Slice points are automatically defined by the data reader */
  bool m_set_slice_points_from_data_reader;
  /** Category for retrieving slice points from data reader */
  slice_points_mode m_var_category;

#ifdef LBANN_HAS_GPU
  /** @brief Workspace buffer.
   *
   *  Parameters for CUDA kernels are copied into this buffer and
   *  asynchronously transferred to GPU.
   */
  std::shared_ptr<hydrogen::simple_buffer<unsigned char, El::Device::CPU>>
    m_workspace;
  /** @brief CUDA event for workspace buffer.
   *
   *  Makes sure asynchronous GPU memory transfers are completed
   *  before modifying workspace buffer.
   */
  gpu_lib::event_wrapper m_workspace_event;
#endif // LBANN_HAS_GPU

  template <typename U, El::Device D>
  friend void fp_setup_outputs_impl(slice_layer<U, Layout, D>&);
  template <typename U>
  friend void fp_compute_impl(slice_layer<U, Layout, Device>&);
  template <typename U>
  friend void bp_compute_impl(slice_layer<U, Layout, Device>&);
};

// =========================================================
// Implementation
// =========================================================

template <typename T, data_layout L, El::Device D>
void slice_layer<T, L, D>::write_specific_proto(lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_slice();
  msg->set_axis(m_slice_dim);
  protobuf::assign_to_repeated(*msg->mutable_slice_points(), m_slice_points);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
slice_layer<TensorDataType, Layout, Device>::slice_layer(lbann_comm* comm)
  : data_type_layer<TensorDataType>(comm),
    m_set_slice_points_from_data_reader(false),
    m_var_category(slice_points_mode::NA)
#ifdef LBANN_HAS_GPU
    ,
    m_workspace{
      std::make_shared<hydrogen::simple_buffer<unsigned char, El::Device::CPU>>(
        0UL,
        hydrogen::SyncInfo<El::Device::CPU>{},
        1U /*=pinned*/)}
#endif /* LBANN_HAS_GPU */
{
  this->m_expected_num_child_layers = -1; // No limit on children
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
slice_layer<TensorDataType, Layout, Device>*
slice_layer<TensorDataType, Layout, Device>::copy() const
{
  return new slice_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string slice_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "slice";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout slice_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
slice_layer<TensorDataType, Layout, Device>::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description slice_layer<TensorDataType, Layout, Device>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Slice dimension", m_slice_dim);
  std::ostringstream ss;
  for (size_t i = 0; i < m_slice_points.size(); ++i) {
    ss << (i > 0 ? ", " : "") << m_slice_points[i];
  }
  desc.add("Slice points", ss.str());
  return desc;
}

template <typename TensorDataType, El::Device Device>
void fp_setup_outputs_impl(
  slice_layer<TensorDataType, data_layout::MODEL_PARALLEL, Device>& l)
{

  // Slice Elemental matrices
  // Note: Assume each mini-batch sample is flat.
  const size_t num_outputs = l.get_num_children();
  const auto& input = l.get_prev_activations();
  size_t offset = l.m_slice_points.front();
  for (size_t j = 0; j < num_outputs; ++j) {
    auto& output = l.get_activations(j);
    const auto& output_size = l.get_output_size(j);
    El::LockedView(output,
                   input,
                   El::IR(offset, offset + output_size),
                   El::ALL);
    offset += output_size;
  }
}

template <typename TensorDataType, El::Device Device>
void fp_setup_outputs_impl(
  slice_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>& l)
{

  const size_t num_outputs = l.get_num_children();
  const auto& input = l.get_prev_activations();
  for (size_t j = 0; j < num_outputs; ++j) {
    auto& output = l.get_activations(j);
    // output.AlignWith(input);
    output.Resize(l.get_output_size(j), input.Width());
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType, Layout, Device>::fp_setup_outputs()
{
  fp_setup_outputs_impl(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType, Layout, Device>::fp_compute_subgrid()
{
  const auto& input_dims = this->get_input_dims();
  const size_t num_dims = input_dims.size();
  if (num_dims > 3) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "is operating on ",
                num_dims,
                "-D tensors, ",
                "but only 3-D tensors are currently supported");
  }

  const int split_dim = input_dims[this->m_slice_dim];

  if (this->m_slice_dim != num_dims - 1) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has axis ",
                this->m_slice_dim,
                " However, ",
                "Subgrpah parallelism is supported when split axis is the last "
                "dimension");
  }
  const auto& input = this->get_prev_activations();

  auto const* ptr_input = dynamic_cast<
    El::
      DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, Device> const*>(
    &input);

  if (this->get_communication_flag() == COLL_OPT) {
    El::copy::TranslateBetweenGridsScatter<TensorDataType, Device, Device>(
      *ptr_input,
      this->get_all_activations(),
      split_dim,
      this->get_subgrid_comm(),
      syncSubGridCommunication,
      3);
  }
  else if (this->get_communication_flag() == COLL) {
    El::copy::TranslateBetweenGridsScatter<TensorDataType, Device, Device>(
      *ptr_input,
      this->get_all_activations(),
      split_dim,
      this->get_subgrid_comm(),
      syncSubGridCommunication,
      2);
  }
  else {
    El::copy::TranslateBetweenGridsScatter<TensorDataType, Device, Device>(
      *ptr_input,
      this->get_all_activations(),
      split_dim,
      this->get_subgrid_comm(),
      syncSubGridCommunication,
      1);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType, Layout, Device>::fp_compute()
{
  const auto& input_dims = this->get_input_dims();
  const size_t num_dims = input_dims.size();

  if (this->m_slice_dim == num_dims - 1 &&
      this->subgraph_parallelism_execution()) {
    fp_compute_subgrid();
  }
  else {
    fp_compute_impl(*this);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType, Layout, Device>::bp_compute_subgrid()
{
  const auto& input_dims = this->get_input_dims();

  const int split_dim =
    int(input_dims[this->m_slice_dim] / this->get_num_children());

  auto& input_grad = this->get_error_signals();

  auto* ptr_input_grad = dynamic_cast<
    El::DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
    &input_grad);

  El::copy::TranslateBetweenGridsGather<TensorDataType, Device, Device>(
    *ptr_input_grad,
    this->get_all_prev_error_signals(),
    split_dim,
    this->get_subgrid_comm(),
    syncSubGridCommunication);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType, Layout, Device>::bp_setup_gradient_wrt_inputs()
{
  const auto& output0_grad = this->get_prev_error_signals(0);
  auto& input_grad = this->get_error_signals();
  input_grad.Empty(false);
  input_grad.Resize(this->get_input_size(), output0_grad.Width());
  El::Zeros(input_grad, this->get_input_size(), output0_grad.Width());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType, Layout, Device>::bp_compute()
{

  const auto& input_dims = this->get_input_dims();
  const size_t num_dims = input_dims.size();

  if (this->m_slice_dim == num_dims - 1 &&
      this->subgraph_parallelism_execution()) {
    bp_compute_subgrid();
  }
  else {
    bp_compute_impl(*this);
  }
}

#ifndef LBANN_SLICE_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class slice_layer<T, data_layout::DATA_PARALLEL, Device>;    \
  extern template class slice_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SLICE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_SLICE_HPP_INCLUDED
