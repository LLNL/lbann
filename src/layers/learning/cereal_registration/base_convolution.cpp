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
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/macros/common_cereal_registration.hpp"

namespace lbann {

template <typename TensorDataType, El::Device Device>
template <typename ArchiveT>
void base_convolution_layer<TensorDataType,Device>::serialize(ArchiveT& ar)
{
  using DataTypeLayer = data_type_layer<TensorDataType>;
  ar(::cereal::make_nvp("DataTypeLayer",
                        ::cereal::base_class<DataTypeLayer>(this)),
     CEREAL_NVP(m_output_channels),
     CEREAL_NVP(m_conv_dims),
     CEREAL_NVP(m_pads),
     CEREAL_NVP(m_strides),
     CEREAL_NVP(m_dilations),
     CEREAL_NVP(m_groups),
     CEREAL_NVP(m_bias_scaling_factor));
  /// @todo Consider serializing m_convolution_math_type
}
} // namespace lbann

#define LBANN_COMMA ,
#define PROTO_DEVICE(T, D)                                              \
  LBANN_ADD_ALL_SERIALIZE_ETI(::lbann::base_convolution_layer<T,D>);    \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                       \
    ::lbann::base_convolution_layer<T LBANN_COMMA D>,                   \
    "base_convolution_layer(" #T "," #D ")")
#include "lbann/macros/instantiate_device.hpp"

CEREAL_REGISTER_DYNAMIC_INIT(base_convolution_layer);
