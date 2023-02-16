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

#define LBANN_LOCAL_RESPONSE_NORMALIZATION_LAYER_INSTANTIATE
#include "lbann/comm.hpp"
#include "lbann/layers/regularizers/local_response_normalization.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {
namespace {

template <typename TensorDataType, data_layout layout, El::Device device>
struct lrn_builder;

template <typename TensorDataType, El::Device device>
struct lrn_builder<TensorDataType, data_layout::DATA_PARALLEL, device> {
  using LayerType =
    local_response_normalization_layer<TensorDataType,
                                       data_layout::DATA_PARALLEL,
                                       device>;
  static std::unique_ptr<LayerType> Get(lbann_comm* comm,
                                        lbann_data::Layer const& layer_msg) {
    const auto& params = layer_msg.local_response_normalization();
    return std::make_unique<LayerType>(
      params.window_width(),
      El::To<TensorDataType>(params.lrn_alpha()),
      El::To<TensorDataType>(params.lrn_beta()),
      El::To<TensorDataType>(params.lrn_k()));
  }
};

template <typename TensorDataType, El::Device device>
struct lrn_builder<TensorDataType, data_layout::MODEL_PARALLEL, device> {
  static std::unique_ptr<Layer> Get(lbann_comm* comm,
                                    lbann_data::Layer const& layer_msg) {
    LBANN_ERROR("local response normalization layer is only supported "
                "with a data-parallel layout");
    return nullptr;
  }
};
}// namespace

template <typename T, data_layout L, El::Device D>
void local_response_normalization_layer<T,L,D>::write_specific_proto(lbann_data::Layer& proto) const {
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_local_response_normalization();
  msg->set_window_width(m_window_width);
  msg->set_lrn_alpha(m_alpha);
  msg->set_lrn_beta(m_beta);
  msg->set_lrn_k(m_k);
}

template <typename TensorDataType, data_layout layout, El::Device device>
std::unique_ptr<Layer> build_local_response_normalization_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const& layer_msg)
{
  using Builder = lrn_builder<TensorDataType,layout,device>;
  return Builder::Get(comm, layer_msg);
}

#define PROTO_DEVICE(T, Device)                                         \
  template std::unique_ptr<Layer>                                       \
  build_local_response_normalization_layer_from_pbuf<                   \
      T, data_layout::DATA_PARALLEL, Device>(                           \
    lbann_comm*, lbann_data::Layer const&);                             \
  template std::unique_ptr<Layer>                                       \
  build_local_response_normalization_layer_from_pbuf<                   \
      T, data_layout::MODEL_PARALLEL, Device>(                          \
    lbann_comm*, lbann_data::Layer const&);                             \
  template class local_response_normalization_layer<                    \
    T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
