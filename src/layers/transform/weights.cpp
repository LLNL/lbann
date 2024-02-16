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

#define LBANN_WEIGHTS_LAYER_INSTANTIATE
#include "lbann/layers/transform/weights.hpp"

#include "lbann/models/model.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/layers.pb.h"
#include "lbann/proto/lbann.pb.h"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
weights_layer<TensorDataType, Layout, Device>::weights_layer(
  std::vector<El::Int> dims)
  : data_type_layer<TensorDataType>(nullptr)
{
  std::vector<int> dims_;
  for (const auto& d : dims) {
    dims_.push_back(d);
  }
  this->set_output_dims(dims_);
  this->m_expected_num_parent_layers = 0;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
weights_layer<TensorDataType, Layout, Device>*
weights_layer<TensorDataType, Layout, Device>::copy() const
{
  return new weights_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string weights_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "weights";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
weights_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
weights_layer<TensorDataType, Layout, Device>::get_device_allocation() const
{
  return Device;
}

template <typename T, data_layout L, El::Device D>
void weights_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_weights_layer();
  protobuf::assign_to_repeated(*msg->mutable_dims(), this->get_output_dims());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void weights_layer<TensorDataType, Layout, Device>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Initialize default weights if none are provided
  if (!this->has_weights()) {
    auto w = std::make_shared<WeightsType>(*this->get_comm());
    auto init = std::make_unique<constant_initializer<DataType>>(DataType(0));
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->add_weights(w);
    this->m_model->add_weights(std::move(w));
  }
  if (this->num_weights() != 1) {
    LBANN_ERROR("attempted to setup ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "with an invalid number of weights ",
                "(expected at most 1, ",
                "but found ",
                this->num_weights(),
                ")");
  }

  // Setup weights
  const auto& output_dims_ = this->get_output_dims();
  std::vector<size_t> output_dims(output_dims_.begin(), output_dims_.end());
  auto dist = this->get_activations().DistData();
  dist.rowDist = El::STAR;
  this->get_weights(0).set_dims(output_dims);
  this->get_weights(0).set_matrix_distribution(dist);

  // Initialize freeze state
  if (this->m_frozen) {
    this->get_weights(0).freeze();
  }
  else {
    this->get_weights(0).unfreeze();
  }
  if (this->get_weights(0).is_frozen() != this->m_frozen) {
    LBANN_ERROR((this->m_frozen ? "" : "un"),
                "frozen ",
                "layer \"",
                this->get_name(),
                "\" has ",
                (this->get_weights(0).is_frozen() ? "" : "un"),
                "frozen ",
                "weights \"",
                this->get_weights(0).get_name(),
                "\"");
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void weights_layer<TensorDataType, Layout, Device>::fp_compute()
{

  // Do nothing if there is no local data
  auto& local_output = this->get_local_activations();
  if (local_output.IsEmpty()) {
    return;
  }

  // Duplicate weights across columns of output matrix
  const auto& local_weights = this->weights_values(0).LockedMatrix();
  if (local_output.Width() <= 32) { // The number 32 is a heuristic
    // Use copies for broadcast
    for (int i = 0; i < local_output.Width(); ++i) {
      MatType v;
      El::View(v,
               local_output,
               El::IR(0, local_weights.Height()),
               El::IR(i, i + 1));
      El::Copy(local_weights, v);
    }
  }
  else {
    // Use GEMM with ones for broadcast
    MatType ones;
    El::Ones(ones, local_output.Width(), 1);
    El::Gemm(El::NORMAL,
             El::TRANSPOSE,
             El::TypeTraits<TensorDataType>::One(),
             local_weights,
             ones,
             El::TypeTraits<TensorDataType>::Zero(),
             local_output);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void weights_layer<TensorDataType, Layout, Device>::bp_compute()
{

  // Do nothing if there is no optimizer
  auto* opt = this->get_weights(0).get_optimizer();
  if (opt == nullptr) {
    return;
  }

  // Accumulate gradients over rows of output grad matrix
  TensorDataType dst_scale, gradient_scale;
  const auto& local_output_grad = this->get_local_prev_error_signals();
  auto& weights_grad =
    opt->get_gradient_buffer(dst_scale, gradient_scale, true);
  auto& local_weights_grad = weights_grad.Matrix();
  if (!local_weights_grad.IsEmpty()) {
    MatType ones;
    El::Ones(ones, local_output_grad.Width(), 1);
    El::Gemv(El::NORMAL,
             gradient_scale,
             local_output_grad,
             ones,
             dst_scale,
             local_weights_grad);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_weights_layer_from_pbuf(lbann_comm* comm,
                              lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, weights_layer);
  using LayerType = weights_layer<TensorDataType, Layout, Device>;

  const auto& params = proto_layer.weights_layer();
  return std::make_unique<LayerType>(
    protobuf::to_vector<El::Int>(params.dims()));
}

// Explicit template instantiation
#define PROTO_DEVICE(T, Device)                                                \
  template class weights_layer<T, data_layout::DATA_PARALLEL, Device>;         \
  template class weights_layer<T, data_layout::MODEL_PARALLEL, Device>;        \
  LBANN_LAYER_BUILDER_ETI(weights, T, Device)
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
