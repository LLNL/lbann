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

#ifndef LBANN_LAYERS_TRANSFORM_CONCATENATE_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_CONCATENATE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/exception.hpp"

#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include <lbann/proto/proto_common.hpp>

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
class concatenate_distconv_adapter
  : public data_type_distconv_adapter<TensorDataType>
{
public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  concatenate_distconv_adapter(Layer& layer)
    : data_type_distconv_adapter<TensorDataType>(layer)
  {}
  virtual ~concatenate_distconv_adapter() = default;
  dc::Shape get_activations_local_shape(int index = 0) const override;
  void fp_compute();
  void bp_compute();
};
#endif // LBANN_HAS_DISTCONV

/** @brief Concatenate tensors along specified dimension
 *
 *  All input tensors must have identical dimensions, except for the
 *  concatenation dimension.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class concatenate_layer : public data_type_layer<TensorDataType>
{
public:
  concatenate_layer(lbann_comm* comm, size_t concat_dim);
  concatenate_layer(const concatenate_layer& other) = default;
  concatenate_layer& operator=(const concatenate_layer& other) = default;

  concatenate_layer* copy() const override;

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

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  El::SyncInfo<Device> syncSubGridCommunication = El::SyncInfo<Device>();

  friend class cereal::access;
  concatenate_layer() : concatenate_layer(nullptr, 0) {}

  void setup_pointers() override;
  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void fp_setup_outputs(El::Int mini_batch_size) override;
  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override;
  void fp_compute() override;
  void bp_compute() override;

private:
  /** @brief Tensor dimension to concatenate along. */
  size_t m_concat_dim;

#ifdef LBANN_HAS_GPU
  /** @brief Workspace buffer.
   *
   *  Parameters for CUDA kernels are copied into this buffer and
   *  asynchronously transferred to GPU.
   */
  std::vector<unsigned char> m_workspace;
  /** @brief CUDA event for workspace buffer.
   *
   *  Makes sure asynchronous GPU memory transfers are completed
   *  before modifying workspace buffer.
   */
  gpu_lib::event_wrapper m_workspace_event;
#endif // LBANN_HAS_GPU

  template <typename U>
  friend void fp_compute_impl(concatenate_layer<U, Layout, Device>&, size_t);
  template <typename U, El::Device D>
  friend void
  bp_setup_gradient_wrt_inputs_impl(concatenate_layer<U, Layout, D>&);
  template <typename U>
  friend void bp_compute_impl(concatenate_layer<U, Layout, Device>&, size_t);

  void fp_compute_subgrid();

  void bp_compute_subgrid();

#ifdef LBANN_HAS_DISTCONV
  friend class concatenate_distconv_adapter<TensorDataType, Layout, Device>;

protected:
  bool is_distconv_supported() const override
  {
    // Only supported for the channel dimension
    return Device == El::Device::GPU && Layout == data_layout::DATA_PARALLEL &&
           m_concat_dim == 0;
  }
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override
  {
    this->get_distconv_adapter_ptr() = std::make_unique<
      concatenate_distconv_adapter<TensorDataType, Layout, Device>>(*this);
  }
  concatenate_distconv_adapter<TensorDataType, Layout, Device>&
  get_distconv_adapter() override;
  const concatenate_distconv_adapter<TensorDataType, Layout, Device>&
  get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

// =========================================================
// Implementation
// =========================================================

template <typename T, data_layout L, El::Device D>
void concatenate_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_concatenation();
  msg->set_axis(m_concat_dim);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
concatenate_layer<TensorDataType, Layout, Device>::concatenate_layer(
  lbann_comm* comm,
  size_t concat_dim)
  : data_type_layer<TensorDataType>(comm), m_concat_dim{concat_dim}
{
  this->m_expected_num_parent_layers = -1; // No limit on parents
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
concatenate_layer<TensorDataType, Layout, Device>*
concatenate_layer<TensorDataType, Layout, Device>::copy() const
{
  return new concatenate_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string concatenate_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "concatenate";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
concatenate_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
concatenate_layer<TensorDataType, Layout, Device>::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
concatenate_layer<TensorDataType, Layout, Device>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Concatenation dimension", m_concat_dim);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType, Layout, Device>::setup_pointers()
{
  data_type_layer<TensorDataType>::setup_pointers();
  if (this->get_num_parents() < 1) {
    LBANN_ERROR(get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has no parents");
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Dimensions of first input tensor
  auto output_dims = this->get_input_dims(0);
  if (m_concat_dim >= output_dims.size()) {
    std::ostringstream err;
    err << get_type() << " layer \"" << this->get_name() << "\" "
        << "is concatenating along dimension " << m_concat_dim << ", "
        << "but it has a " << output_dims.size() << "-D input tensor "
        << "(parent layer \"" << this->get_parent_layers()[0]->get_name()
        << "\" "
        << "outputs with dimensions ";
    for (size_t d = 0; d < output_dims.size(); ++d) {
      err << (d > 0 ? " x " : "") << output_dims[d];
    }
    err << ")";
    LBANN_ERROR(err.str());
  }

  // Dimensions of remaining input tensors
  for (int j = 1; j < this->get_num_parents(); ++j) {
    const auto& input_dims = this->get_input_dims(j);
    if (input_dims.size() != output_dims.size() ||
        !std::equal(input_dims.begin(),
                    input_dims.begin() + m_concat_dim,
                    output_dims.begin()) ||
        !std::equal(input_dims.begin() + m_concat_dim + 1,
                    input_dims.end(),
                    output_dims.begin() + m_concat_dim + 1)) {
      std::ostringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "expects input tensors with dimensions ";
      for (size_t d = 0; d < output_dims.size(); ++d) {
        err << (d > 0 ? " x " : "");
        if (d == m_concat_dim) {
          err << "X";
        }
        else {
          err << output_dims[d];
        }
      }
      err << ", but parent layer "
          << "\"" << this->get_parent_layers()[j]->get_name() << "\" "
          << "outputs with dimensions ";
      for (size_t d = 0; d < input_dims.size(); ++d) {
        err << (d > 0 ? " x " : "") << input_dims[d];
      }
      LBANN_ERROR(err.str());
    }
    output_dims[m_concat_dim] += input_dims[m_concat_dim];
  }

  // Model-parallel implementation only supports flat data
  if (Layout == data_layout::MODEL_PARALLEL &&
      get_linear_size(m_concat_dim, output_dims.data()) > 1) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "attempted to concatenate along dimension ",
                m_concat_dim,
                ", ",
                "but model-parallel concatenate layer "
                "only supports flat data");
  }

  // Update output dimensions
  this->set_output_dims(output_dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType, Layout, Device>::fp_setup_outputs(
  El::Int mini_batch_size)
{
  std::cout<<"Concat setup output\n";
#ifdef LBANN_HAS_DISTCONV
  if (!this->keep_original_outputs(0))
    return;
#endif // LBANN_HAS_DISTCONV
  const auto& input0 = this->get_prev_activations(0);
  auto& output = this->get_activations();
  output.Empty(false);
  if (this->get_num_parents() == 1) {
    El::LockedView(output, input0);
  }
  else {
    if (this->subgraph_parallelism_execution() == false) {
      output.AlignWith(input0);
    }

    output.Resize(this->get_output_size(), input0.Width());
  }
  std::cout<<"Concat setup output done\n";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType, Layout, Device>::fp_compute_subgrid()
{
  std::cout<<"Running Concat fp\n";

  const auto& input_dims = this->get_input_dims(0);

  int split_dim = int(input_dims[m_concat_dim]);

  auto& input = this->get_activations();

  auto* ptr_input = dynamic_cast<
    El::DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
    &input);

  El::copy::TranslateBetweenGridsGather<TensorDataType, Device, Device>(
    *ptr_input,
    this->get_all_prev_activations(),
    split_dim,
    this->get_subgrid_comm(),
    syncSubGridCommunication);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType, Layout, Device>::bp_compute_subgrid()
{
  std::cout<<"Running Concat bp\n";
  const auto& input_dims = this->get_input_dims(0);

  int split_dim = int(input_dims[m_concat_dim] * this->get_num_parents());

  const auto& input_grad = this->get_prev_error_signals();

  auto const* ptr_input_grad = dynamic_cast<
    El::
      DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, Device> const*>(
    &input_grad);

  if (this->get_communication_flag() == COLL_OPT) {
    El::copy::TranslateBetweenGridsScatter<TensorDataType, Device, Device>(
      *ptr_input_grad,
      this->get_all_error_signals(),
      split_dim,
      this->get_subgrid_comm(),
      syncSubGridCommunication,
      3);
  }
  else if (this->get_communication_flag() == COLL) {
    El::copy::TranslateBetweenGridsScatter<TensorDataType, Device, Device>(
      *ptr_input_grad,
      this->get_all_error_signals(),
      split_dim,
      this->get_subgrid_comm(),
      syncSubGridCommunication,
      2);
  }
  else {
    El::copy::TranslateBetweenGridsScatter<TensorDataType, Device, Device>(
      *ptr_input_grad,
      this->get_all_error_signals(),
      split_dim,
      this->get_subgrid_comm(),
      syncSubGridCommunication,
      1);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType, Layout, Device>::fp_compute()
{
  const auto& input_dims = this->get_input_dims();
  const size_t num_dims = input_dims.size();
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    get_distconv_adapter().fp_compute();
    return;
  }
#endif

  // Just make a view if there is one input
  if (this->get_num_parents() == 1) {
    El::LockedView(this->get_activations(), this->get_prev_activations(0));
    return;
  }

  // Perform concatenation
  std::cout<<"FP compute Concat\n";
  if (m_concat_dim == num_dims - 1 && this->subgraph_parallelism_execution()) {
    this->fp_compute_subgrid();
  }
  else {
    fp_compute_impl(*this, m_concat_dim);
  }
}

template <typename TensorDataType, El::Device Device>
void bp_setup_gradient_wrt_inputs_impl(
  concatenate_layer<TensorDataType, data_layout::MODEL_PARALLEL, Device>& l)
{
#ifdef LBANN_HAS_DISTCONV
  if (l.distconv_enabled()) {
    LBANN_ERROR("Model-parallel LBANN matrix not supported in distconv");
  }
#endif // LBANN_HAS_DISTCONV

  // Slice Elemental matrices
  // Note: Assume each mini-batch sample is flat.
  const size_t num_inputs = l.get_num_parents();
  const auto& output_grad = l.get_prev_error_signals();
  size_t offset = 0;
  for (size_t j = 0; j < num_inputs; ++j) {
    auto& input_grad = l.get_error_signals(j);
    const auto& input_size = l.get_input_size(j);
    El::LockedView(input_grad,
                   output_grad,
                   El::IR(offset, offset + input_size),
                   El::ALL);
    offset += input_size;
  }
}

template <typename TensorDataType, El::Device Device>
void bp_setup_gradient_wrt_inputs_impl(
  concatenate_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>& l)
{

  const size_t num_inputs = l.get_num_parents();
  const auto& output_grad = l.get_prev_error_signals();
  if (num_inputs == 1) {
#ifdef LBANN_HAS_DISTCONV
    if (!l.keep_original_gradient_wrt_inputs(0))
      return;
#endif
    El::LockedView(l.get_error_signals(0), output_grad);
  }
  else {
    for (size_t j = 0; j < num_inputs; ++j) {
#ifdef LBANN_HAS_DISTCONV
      if (!l.keep_original_gradient_wrt_inputs(j))
        continue;
#endif
      auto& input_grad = l.get_error_signals(j);
      if (l.subgraph_parallelism_execution() == false) {
        input_grad.AlignWith(output_grad);
      }
      input_grad.Resize(l.get_input_size(j), output_grad.Width());
    }
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType, Layout, Device>::
  bp_setup_gradient_wrt_inputs(El::Int mini_batch_size)
{
  bp_setup_gradient_wrt_inputs_impl(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType, Layout, Device>::bp_compute()
{

  const auto& input_dims = this->get_input_dims();
  const size_t num_dims = input_dims.size();

#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    get_distconv_adapter().bp_compute();
    return;
  }
#endif

  // Just make a view if there is one input
  if (this->get_num_parents() == 1) {
    El::LockedView(this->get_error_signals(0), this->get_prev_error_signals());
    return;
  }

  // Perform slice
  if (m_concat_dim == num_dims - 1 && this->subgraph_parallelism_execution()) {
    this->bp_compute_subgrid();
  }
  else {
    bp_compute_impl(*this, m_concat_dim);
  }
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
concatenate_distconv_adapter<TensorDataType, T_layout, Dev>&
concatenate_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<
    concatenate_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<const concatenate_layer<TensorDataType, T_layout, Dev>&>(*this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const concatenate_distconv_adapter<TensorDataType, T_layout, Dev>&
concatenate_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const
{
  return dynamic_cast<
    const concatenate_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape concatenate_distconv_adapter<TensorDataType, Layout, Device>::
  get_activations_local_shape(int index) const
{
  assert_eq(index, 0);
  auto shape = this->get_prev_activations().get_local_shape();
  shape[-2] = this->get_activations_shape()[-2];
  return shape;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_distconv_adapter<TensorDataType, Layout, Device>::fp_compute()
{
  assert_always(this->layer().get_num_parents() == 2);
  dc::tensor::Concatenate(this->get_activations(0),
                          this->get_prev_activations(0),
                          this->get_prev_activations(1),
                          default_hydrogen_stream());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_distconv_adapter<TensorDataType, Layout, Device>::bp_compute()
{
  dc::tensor::Slice(this->get_error_signals(0),
                    this->get_error_signals(1),
                    this->get_prev_error_signals(0),
                    default_hydrogen_stream());
}
#endif // LBANN_HAS_DISTCONV

#ifndef LBANN_CONCATENATE_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class concatenate_layer<T,                                   \
                                          data_layout::DATA_PARALLEL,          \
                                          Device>;                             \
  extern template class concatenate_layer<T,                                   \
                                          data_layout::MODEL_PARALLEL,         \
                                          Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CONCATENATE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_CONCATENATE_HPP_INCLUDED
