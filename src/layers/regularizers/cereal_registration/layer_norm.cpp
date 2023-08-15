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
#include "lbann/utils/serialize.hpp"
#include <lbann/layers/regularizers/layer_norm.hpp>

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
template <typename ArchiveT>
void layer_norm_layer<TensorDataType, Layout, Device>::serialize(ArchiveT& ar)
{
  using DataTypeLayer = data_type_layer<TensorDataType>;
  ar(::cereal::make_nvp("DataTypeLayer",
                        ::cereal::base_class<DataTypeLayer>(this)),
     CEREAL_NVP(m_epsilon),
     CEREAL_NVP(m_scale),
     CEREAL_NVP(m_bias));
}

} // namespace lbann

// In this case, we want to exclude FP16 types, so we must handle
// registration manually.
#include "lbann/macros/common_cereal_registration.hpp"
#define LBANN_COMMA ,
#define PROTO_DEVICE(T, D)                                                     \
  LBANN_ADD_ALL_SERIALIZE_ETI(                                                 \
    ::lbann::layer_norm_layer<T, ::lbann::data_layout::DATA_PARALLEL, D>);     \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                              \
    ::lbann::layer_norm_layer<                                                 \
      T LBANN_COMMA ::lbann::data_layout::DATA_PARALLEL LBANN_COMMA D>,        \
    "layer_norm_layer(" #T ",DATA_PARALLEL," #D ")")

#define PROTO_CPU(T) PROTO_DEVICE(T, El::Device::CPU)
#ifdef LBANN_HAS_GPU
#define PROTO_GPU(T) PROTO_DEVICE(T, El::Device::GPU)
#else
#define PROTO_GPU(T)
#endif

#define PROTO(T)                                                               \
  PROTO_CPU(T);                                                                \
  PROTO_GPU(T)

#include "lbann/macros/instantiate.hpp"

CEREAL_REGISTER_DYNAMIC_INIT(layer_norm_layer);
