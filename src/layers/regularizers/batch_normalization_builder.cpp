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

#include "lbann/layers/regularizers/batch_normalization.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV

#include "lbann/proto/layers.pb.h"
#include <type_traits>

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer> lbann::build_batch_normalization_layer_from_pbuf(
  lbann_comm* comm,
  lbann_data::Layer const& proto_layer)
{
  const auto& params = proto_layer.batch_normalization();
  if constexpr (L == data_layout::DATA_PARALLEL &&
                (std::is_same_v<T, float> || std::is_same_v<T, double>
#ifdef LBANN_HAS_GPU_FP16
                 || (std::is_same_v<T, fp16> && (D == El::Device::GPU))
#endif
                  )) {
    int statistics_group_size = params.statistics_group_size();
    if (statistics_group_size < 0) {
      statistics_group_size = 0; // Global statistics.
    }
    else if (statistics_group_size == 0) {
      statistics_group_size = 1; // Default to local.
    }
    const auto& aggr_str = params.stats_aggregation();
    if (!aggr_str.empty()) {
      LBANN_WARNING(
        "stats_aggregation field for BatchNormalization is deprecated");
      if (aggr_str == "local") {
        statistics_group_size = 1;
      }
      else if (aggr_str == "node_local") {
        statistics_group_size = comm->get_procs_per_node();
      }
      else if (aggr_str == "global") {
        statistics_group_size = 0;
      }
      else {
        LBANN_ERROR("Invalid batch normalization stats aggregation ", aggr_str);
        return nullptr;
      }
    }
    // Set defaults if not given.
    auto const decay = params.decay() == 0.0 ? 0.9 : params.decay();
    auto const epsilon = params.epsilon() == 0.0 ? 1e-5 : params.epsilon();
    auto const bessel = params.no_bessel_correction() ? false : true;
    if constexpr (std::is_same_v<T, float>) {
      return std::make_unique<
        batch_normalization_layer<float, data_layout::DATA_PARALLEL, D>>(
        decay,
        epsilon,
        statistics_group_size,
        bessel);
    }
    else if constexpr (std::is_same_v<T, double>) {
      return std::make_unique<
        batch_normalization_layer<double, data_layout::DATA_PARALLEL, D>>(
        decay,
        epsilon,
        statistics_group_size,
        bessel);
    }
#ifdef LBANN_HAS_GPU_FP16
    else if constexpr (std::is_same_v<T, fp16> && D == El::Device::GPU) {
      return std::make_unique<
        batch_normalization_layer<half, data_layout::DATA_PARALLEL, D>>(
        decay,
        epsilon,
        statistics_group_size,
        bessel);
    }
#endif
  }
  else {
    LBANN_ERROR("batch normalization layer is only supported for \"float\" and "
                "\"double\" (and \"half\" on GPU) with a data-parallel layout");
    return nullptr;
  }
}

namespace lbann {

#define PROTO_DEVICE(T, Device)                                                \
  LBANN_LAYER_BUILDER_ETI(batch_normalization, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
