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

#include "lbann/layers/regularizers/entrywise_batch_normalization.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

#include "lbann/proto/layers.pb.h"

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_entrywise_batch_normalization_layer_from_pbuf(
  lbann_comm* comm,
  lbann_data::Layer const& proto_layer)
{
  auto const& params = proto_layer.entrywise_batch_normalization();
  if constexpr (std::is_same_v<T, float>)
    return std::make_unique<entrywise_batch_normalization_layer<float, L, D>>(
      params.decay(),
      params.epsilon());
#ifdef LBANN_HAS_DOUBLE
  else if constexpr (std::is_same_v<T, double>)
    return std::make_unique<entrywise_batch_normalization_layer<double, L, D>>(
      params.decay(),
      params.epsilon());
#endif // LBANN_HAS_DOUBLE
  else
    LBANN_ERROR("entrywise_batch_normalization_layer is only supported for "
                "\"float\" and \"double\".");
}

namespace lbann {

#define PROTO_DEVICE(T, Device)                                                \
  LBANN_LAYER_BUILDER_ETI(entrywise_batch_normalization, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
