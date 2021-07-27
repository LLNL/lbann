////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/transform/scatter.hpp"
#include "lbann/proto/helpers.hpp"

#include <lbann/proto/proto_common.hpp>
#include <layers.pb.h>

namespace lbann {

namespace
{
template <typename T, data_layout L, El::Device D>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR("Attempted to instantiate layer \"scatter\""
                "with Layout=", to_string(L), ".\n"
                "This layer is only "
                "supported with DATA_PARALLEL data layout.");
    return nullptr;
  }
};

template <typename TensorDataType, El::Device Device>
struct Builder<TensorDataType,data_layout::DATA_PARALLEL,Device>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType = scatter_layer<TensorDataType,
                                    data_layout::DATA_PARALLEL,
                                    Device>;
    return make_unique<LayerType>(std::forward<Args>(args)...);
  }
};
} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_scatter_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, scatter);
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  auto dims = parse_list<int>(proto_layer.scatter().dims());
  const auto& params = proto_layer.scatter();
  int axis = -1;
  if (params.has_axis()){
    axis = params.axis().value();
  }
  return BuilderType::Build(dims, axis);
}

#define PROTO_DEVICE(T, Device) \
  LBANN_LAYER_BUILDER_ETI(scatter, T, Device)
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
