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

#include "lbann/layers/regularizers/layer_norm.hpp"
#include "lbann/utils/exception.hpp"

#include "lbann/proto/layers.pb.h"

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_layer_norm_layer_from_pbuf(lbann_comm* /*comm*/,
                                        lbann_data::Layer const& proto_layer)
{
  auto const& params = proto_layer.layer_norm();
  bool const scale = params.scale();
  bool const bias = params.bias();
  double const epsilon =
    (params.has_epsilon() ? params.epsilon().value() : 1e-5);
  if constexpr (std::is_same_v<T, float>)
    return std::make_unique<layer_norm_layer<float, L, D>>(epsilon,
                                                           scale,
                                                           bias);
  else if constexpr (std::is_same_v<T, double>)
    return std::make_unique<layer_norm_layer<double, L, D>>(epsilon,
                                                            scale,
                                                            bias);
  else
    LBANN_ERROR(
      "layer_norm_layer is only supported for \"float\" and \"double\".");
}

namespace lbann {

#define PROTO_DEVICE(T, Device) LBANN_LAYER_BUILDER_ETI(layer_norm, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
