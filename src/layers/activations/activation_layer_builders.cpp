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

#include "lbann/comm.hpp"
#include "lbann/layers/activations/activation_layer_builders.hpp"

#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/identity.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/log_softmax.hpp"
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/softmax.hpp"

#include "lbann/proto/layers.pb.h"

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_elu_layer_from_pbuf(lbann_comm* comm,
                                 lbann_data::Layer const& proto_layer)
{
  auto const& params = proto_layer.elu();
  auto const alpha = params.alpha();
  if (alpha != 0.0)
    return std::make_unique<elu_layer<T, L, D>>(comm, El::To<T>(alpha));
  else
    return std::make_unique<elu_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_identity_layer_from_pbuf(lbann_comm*, lbann_data::Layer const&)
{
  return std::make_unique<identity_layer<T, L, D>>();
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_leaky_relu_layer_from_pbuf(lbann_comm* comm,
                                        lbann_data::Layer const& proto_layer)
{
  auto const& params = proto_layer.leaky_relu();
  auto const negative_slope = params.negative_slope();
  if (negative_slope != 0.0)
    return std::make_unique<leaky_relu_layer<T, L, D>>(
      comm,
      El::To<T>(negative_slope));
  else
    return std::make_unique<leaky_relu_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_log_softmax_layer_from_pbuf(lbann_comm* comm,
                                         lbann_data::Layer const&)
{
  return std::make_unique<log_softmax_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_relu_layer_from_pbuf(lbann_comm* comm, lbann_data::Layer const&)
{
  return std::make_unique<relu_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_softmax_layer_from_pbuf(lbann_comm* comm,
                                     lbann_data::Layer const& proto_layer)
{
  using LayerType = softmax_layer<T, L, D>;
  const auto& sm_mode = proto_layer.softmax().softmax_mode();
  if (sm_mode == "instance" || sm_mode == "")
    return std::make_unique<LayerType>(comm, softmax_mode::INSTANCE);
  else if (sm_mode == "channel")
    return std::make_unique<LayerType>(comm, softmax_mode::CHANNEL);
  else
    return std::make_unique<LayerType>(comm, softmax_mode::INVALID);
}

namespace lbann {

#define PROTO_DEVICE(T, Device)                                                \
  LBANN_LAYER_BUILDER_ETI(elu, T, Device);                                     \
  LBANN_LAYER_BUILDER_ETI(identity, T, Device);                                \
  LBANN_LAYER_BUILDER_ETI(leaky_relu, T, Device);                              \
  LBANN_LAYER_BUILDER_ETI(log_softmax, T, Device);                             \
  LBANN_LAYER_BUILDER_ETI(relu, T, Device);                                    \
  LBANN_LAYER_BUILDER_ETI(softmax, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
