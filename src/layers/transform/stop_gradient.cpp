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

#define LBANN_STOP_GRADIENT_LAYER_INSTANTIATE
#include "lbann/layers/transform/stop_gradient.hpp"

#include "lbann/proto/datatype_helpers.hpp"
#include <lbann/proto/proto_common.hpp>

#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/layers.pb.h"

namespace lbann {

LBANN_LAYER_DEFAULT_BUILDER(stop_gradient)

template <typename T, data_layout L, El::Device D>
void stop_gradient_layer<T,L,D>::write_specific_proto(lbann_data::Layer& proto) const {
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_stop_gradient();
}

#define PROTO_DEVICE(T, Device) \
  template class stop_gradient_layer<T, data_layout::DATA_PARALLEL, Device>; \
  template class stop_gradient_layer<T, data_layout::MODEL_PARALLEL, Device>; \
  LBANN_LAYER_BUILDER_ETI(stop_gradient, T, Device)

#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
