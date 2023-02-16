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

#define LBANN_DISCRETE_RANDOM_LAYER_INSTANTIATE
#include "lbann/comm.hpp"
#include "lbann/layers/transform/discrete_random.hpp"

#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/layers.pb.h"

#include "lbann/proto/datatype_helpers.hpp"


namespace lbann {

template <typename T, data_layout L, El::Device D>
void discrete_random_layer<T,L,D>::write_specific_proto(lbann_data::Layer& proto) const {
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_discrete_random();
  protobuf::assign_to_repeated(*msg->mutable_values(), m_values);
  protobuf::assign_to_repeated(*msg->mutable_dims(), this->get_output_dims());
}

#define PROTO(T)                                                               \
  template class discrete_random_layer<T,                                      \
                                       data_layout::DATA_PARALLEL,             \
                                       El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
