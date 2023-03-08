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

#define LBANN_SPLIT_LAYER_INSTANTIATE
#include "lbann/layers/transform/split.hpp"

#include <lbann/proto/proto_common.hpp>

namespace lbann {

LBANN_LAYER_DEFAULT_BUILDER(split)

#define PROTO(T)                                                               \
  template class split_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>;  \
  template class split_layer<T, data_layout::MODEL_PARALLEL, El::Device::CPU>; \
  LBANN_LAYER_BUILDER_ETI(split, T, El::Device::CPU)

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Dev>
void split_distconv_adapter<TensorDataType, Layout, Dev>::bp_compute()
{
  LBANN_ERROR(this->get_name(), ": Distconv not supported");
}

#define PROTO(T)                                                               \
  template class split_distconv_adapter<T,                                     \
                                        data_layout::DATA_PARALLEL,            \
                                        El::Device::CPU>;                      \
  template class split_distconv_adapter<T,                                     \
                                        data_layout::MODEL_PARALLEL,           \
                                        El::Device::CPU>

#include "lbann/macros/instantiate.hpp"
#endif // LBANN_HAS_DISTCONV

} // namespace lbann
