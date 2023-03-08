////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#define LBANN_INSTANTIATE_OPERATOR_LAYER
#include "lbann/layers/operator_layer_impl.hpp"
#include "lbann/proto/datatype_helpers.hpp"

namespace lbann {

#define PROTO_DEVICE(T, D)                                                     \
  template class OperatorLayer<T, T, data_layout::DATA_PARALLEL, D>;           \
  template class OperatorLayer<T, T, data_layout::MODEL_PARALLEL, D>;          \
  template std::unique_ptr<Layer>                                              \
  build_operator_layer_from_pbuf<T, T, data_layout::DATA_PARALLEL, D>(         \
    lbann_comm*,                                                               \
    lbann_data::Layer const&);                                                 \
  template std::unique_ptr<Layer>                                              \
  build_operator_layer_from_pbuf<T, T, data_layout::MODEL_PARALLEL, D>(        \
    lbann_comm*,                                                               \
    lbann_data::Layer const&)

#include "lbann/macros/instantiate_device.hpp"

template <typename T, typename O, data_layout L, El::Device D>
void OperatorLayer<T, O, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{

  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_operator_layer();
  auto* op = msg->add_ops();
  op->set_input_datatype(proto::ProtoDataType<T>);
  op->set_output_datatype(proto::ProtoDataType<O>);
  op->set_device_allocation(proto::ProtoDevice<D>);
}

} // namespace lbann
