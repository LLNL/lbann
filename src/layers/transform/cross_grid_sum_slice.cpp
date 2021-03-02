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

#include <lbann/proto/proto_common.hpp>
#include <lbann.pb.h>

namespace lbann {

LBANN_LAYER_DEFAULT_BUILDER(cross_grid_sum_slice)

#define PROTO(T)                                    \
  template class cross_grid_sum_slice_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>; \
  template class cross_grid_sum_slice_layer<T, data_layout::MODEL_PARALLEL, El::Device::CPU>; \
  LBANN_LAYER_BUILDER_ETI(cross_grid_sum_slice, T, El::Device::CPU)

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO


}// namespace lbann
