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

#define LBANN_GAUSSIAN_LAYER_INSTANTIATE
#include "lbann/layers/transform/gaussian.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann/utils/protobuf.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gaussian_layer<TensorDataType, Layout, Device>::fp_compute()
{
  auto& output = this->get_activations();
  const auto& mode =
    this->m_model->get_execution_context().get_execution_mode();
  if (m_training_only && (mode != execution_mode::training)) {
    El::Fill(output, m_mean);
  }
  else {
    gaussian_fill(output, output.Height(), output.Width(), m_mean, m_stdev);
  }
}

template <typename T, data_layout L, El::Device D>
void gaussian_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_gaussian();
  msg->set_mean(m_mean);
  msg->set_stdev(m_stdev);
  protobuf::assign_to_repeated(*msg->mutable_neuron_dims(),
                               this->get_output_dims());
  msg->set_training_only(m_training_only);
}

#define PROTO_DEVICE(T, Device)                                                \
  template class gaussian_layer<T, data_layout::DATA_PARALLEL, Device>;        \
  template class gaussian_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
