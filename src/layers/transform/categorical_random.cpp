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

#define LBANN_CATEGORICAL_RANDOM_LAYER_INSTANTIATE
#include "lbann/comm.hpp"
#include "lbann/layers/transform/categorical_random.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {
namespace {

template <typename T, data_layout L, El::Device D>
struct Builder
{
  static std::unique_ptr<Layer> Build(lbann_comm*)
  {
    LBANN_ERROR("Attempted to instantiate layer \"categorical_random\" with "
                "Layout=", to_string(L), " and Device=", to_string(D) ,"\n",
                "This layer is only supported with DATA_PARALLEL data layout"
                "on CPU.");
  }
};

template <typename T>
struct Builder<T, data_layout::DATA_PARALLEL, El::Device::CPU>
{
  static std::unique_ptr<Layer> Build(lbann_comm* comm)
  {
    using LayerType = categorical_random_layer<T,
                                               data_layout::DATA_PARALLEL,
                                               El::Device::CPU>;
    return std::make_unique<LayerType>(comm);
  }
};

#ifdef LBANN_HAS_GPU_FP16
template <>
struct Builder<El::gpu_half_type, data_layout::DATA_PARALLEL, El::Device::CPU>
{
  static std::unique_ptr<Layer> Build(lbann_comm*)
  {
    LBANN_ERROR("Attempted to instantiate layer \"categorical_random\" with "
                "TensorDataType=gpu_half_type. This layer is does not support "
                "this type.");
  }
};
#endif // LBANN_HAS_GPU_FP16
}// namespace

template <typename T, data_layout L, El::Device D>
void categorical_random_layer<T,L,D>::write_specific_proto(lbann_data::Layer& proto) const {
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_categorical_random();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_categorical_random_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const&)
{
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  return BuilderType::Build(comm);
}

#define PROTO(T)                                                        \
  template class                                                        \
  categorical_random_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO

#define PROTO_DEVICE(T, Device) \
  LBANN_LAYER_BUILDER_ETI(categorical_random, T, Device)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
