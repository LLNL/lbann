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

#define LBANN_CROSS_GRID_SUM_SLICE_LAYER_INSTANTIATE
#include "lbann/layers/transform/cross_grid_sum_slice.hpp"

#include <lbann.pb.h>
#include <lbann/proto/proto_common.hpp>

namespace lbann {

template <typename TensorDataType, data_layout Layout, El ::Device Device>
std ::unique_ptr<Layer>
build_cross_grid_sum_slice_layer_from_pbuf(lbann_comm* comm,
                                           lbann_data ::Layer const&)
{
  if constexpr (Layout != data_layout::DATA_PARALLEL) {
    LBANN_ERROR(
      "cross_grid_sum_slice_layer only supports DATA_PARALLEL layout");
    return nullptr;
  }

  using LayerType = cross_grid_sum_slice_layer<TensorDataType, Device>;
  return make_unique<LayerType>(comm);
}

#define PROTO_DEVICE(T, D)                                                     \
  template class cross_grid_sum_slice_layer<T, D>;                             \
  LBANN_LAYER_BUILDER_ETI(cross_grid_sum_slice, T, D)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO

} // namespace lbann
