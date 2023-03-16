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

#define LBANN_BERNOULLI_LAYER_INSTANTIATE
#include "lbann/layers/transform/bernoulli.hpp"

#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/layers.pb.h"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void bernoulli_layer<TensorDataType, Layout, Device>::fp_compute()
{
  auto& output = this->get_activations();
  if (this->m_model->get_execution_context().get_execution_mode() ==
      execution_mode::training) {
    bernoulli_fill(output, output.Height(), output.Width(), m_prob);
  }
  else {
    El::Zero(output);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_bernoulli_layer_from_pbuf(lbann_comm* comm,
                                lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, bernoulli);
  const auto& params = proto_layer.bernoulli();
  return std::make_unique<bernoulli_layer<TensorDataType, Layout, Device>>(
    comm,
    protobuf::to_vector<int>(params.neuron_dims()),
    params.prob());
}

template <typename T, data_layout L, El::Device D>
void bernoulli_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_bernoulli();
  msg->set_prob(m_prob);
  protobuf::assign_to_repeated(*msg->mutable_neuron_dims(),
                               this->get_output_dims());
}

#define PROTO_DEVICE(T, Device)                                                \
  template class bernoulli_layer<T, data_layout::DATA_PARALLEL, Device>;       \
  template class bernoulli_layer<T, data_layout::MODEL_PARALLEL, Device>;      \
  LBANN_LAYER_BUILDER_ETI(bernoulli, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
