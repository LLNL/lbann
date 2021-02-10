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
#include <lbann/layers/data_type_layer.hpp>

namespace lbann {

template <typename TensorDataType>
template <typename ArchiveT>
void data_type_layer<TensorDataType>::serialize(ArchiveT& ar)
{
  ar(::cereal::make_nvp("BaseLayer",
                        ::cereal::base_class<Layer>(this)),
     CEREAL_NVP(m_persistent_error_signals));
  // Members not serialized:
  //   m_weights_proxy
  //   m_inputs
  //   m_outputs
  //   m_gradient_wrt_outputs
  //   m_gradient_wrt_inputs;
}


#define PROTO(T)                                                             \
  template void data_type_layer<T>::serialize(cereal::XMLOutputArchive&);    \
  template void data_type_layer<T>::serialize(cereal::XMLInputArchive&);     \
  template void data_type_layer<T>::serialize(cereal::BinaryOutputArchive&); \
  template void data_type_layer<T>::serialize(cereal::BinaryInputArchive&);  \
  template void data_type_layer<T>::serialize(RootedXMLOutputArchive&);      \
  template void data_type_layer<T>::serialize(RootedXMLInputArchive&);       \
  template void data_type_layer<T>::serialize(RootedBinaryOutputArchive&);   \
  template void data_type_layer<T>::serialize(RootedBinaryInputArchive&)
#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann

#define PROTO(T)                                                        \
  extern template class ::cereal::detail::PolymorphicVirtualCaster<     \
    lbann::Layer, lbann::data_type_layer<T>>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#undef PROTO
#define PROTO(T)                                                \
  CEREAL_REGISTER_TYPE_WITH_NAME(                               \
    ::lbann::data_type_layer<T>,                                \
    "dtl(" #T ")");                                             \
  template class ::cereal::detail::PolymorphicVirtualCaster<    \
    lbann::Layer, lbann::data_type_layer<T>>

#include "lbann/macros/instantiate.hpp"
