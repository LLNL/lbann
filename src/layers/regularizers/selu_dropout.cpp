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

#include "lbann/layers/layer.hpp"
#define LBANN_SELU_DROPOUT_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/selu_dropout.hpp"

#include "lbann/utils/protobuf.hpp"

#include <layers.pb.h>

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_selu_dropout_layer_from_pbuf(lbann_comm* comm,
                                          lbann_data::Layer const& proto_layer)
{
  auto const& params = proto_layer.selu_dropout();
  auto const keep_prob = params.keep_prob();
  auto const alpha = params.alpha();
  auto const scale = params.scale();
  if (alpha != 0.0 && scale != 0.0) {
    return std::make_unique<selu_dropout<T, L, D>>(keep_prob, alpha, scale);
  }
  else {
    return std::make_unique<selu_dropout<T, L, D>>(keep_prob);
  }
}

namespace lbann {

#define PROTO_DEVICE(T, Device)                                                \
  template class selu_dropout<T, data_layout::DATA_PARALLEL, Device>;          \
  template class selu_dropout<T, data_layout::MODEL_PARALLEL, Device>;         \
  LBANN_LAYER_BUILDER_ETI(selu_dropout, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
