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
#include "lbann/comm.hpp"
#include "lbann/layers/transform/permute.hpp"

#include "../permute/permuteimpl.hpp"

#include "lbann/utils/serialize.hpp"

#include <utility>
#include <vector>

template <typename T>
using PimplT = typename ::lbann::PermuteLayer<T>::PermuteImpl;

namespace lbann {

template <typename T>
template <typename ArchiveT>
void PermuteLayer<T>::PermuteImpl::save(ArchiveT& ar) const
{
  auto const perm = get_perm();
  ar(perm);
}

template <typename T>
template <typename ArchiveT>
void PermuteLayer<T>::PermuteImpl::load(ArchiveT& ar)
{
  using PimplT = typename PermuteLayer<T>::PermuteImpl;
  using DevImplT = typename PimplT::DeviceImplType;
  std::vector<int> perm;
  ar(perm);
  DevImplT{RowMajor(std::move(perm))}.swap(m_device_impl);
}

template <typename T>
template <typename ArchiveT>
void PermuteLayer<T>::PermuteImpl::load_and_construct(
  ArchiveT& ar,
  cereal::construct<PermuteLayer<T>::PermuteImpl>& construct)
{
  std::vector<int> perm;
  ar(perm);
  construct(std::move(perm));
}

template <typename T>
template <typename ArchiveT>
void PermuteLayer<T>::serialize(ArchiveT& ar)
{
  using DataTypeLayer = data_type_layer<T>;
  ar(::cereal::make_nvp("DataTypeLayer",
                        ::cereal::base_class<DataTypeLayer>(this)),
     CEREAL_NVP(m_impl));
}

} // namespace lbann

// ETI for PIMPL (https://uscilab.github.io/cereal/pimpl.html)
#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
#define LBANN_ADD_BINARY_SAVE_LOAD_ETI(T)                                      \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::save(                   \
    cereal::BinaryOutputArchive&) const;                                       \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::load(                   \
    cereal::BinaryInputArchive&);                                              \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::save(                   \
    RootedBinaryOutputArchive&) const;                                         \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::load(                   \
    RootedBinaryInputArchive&)
#else
#define LBANN_ADD_BINARY_SERIALIZE_ETI(...)
#endif

#ifdef LBANN_HAS_CEREAL_JSON_ARCHIVES
#define LBANN_ADD_JSON_SERIALIZE_ETI(T)                                        \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::save(                   \
    cereal::JSONOutputArchive&) const;                                         \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::load(                   \
    cereal::JSONInputArchive&);                                                \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::save(                   \
    RootedJSONOutputArchive&) const;                                           \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::load(                   \
    RootedJSONInputArchive&)
#else
#define LBANN_ADD_JSON_SERIALIZE_ETI(...)
#endif

#ifdef LBANN_HAS_CEREAL_PORTABLE_BINARY_ARCHIVES
#define LBANN_ADD_PORTABLE_BINARY_SERIALIZE_ETI(T)                             \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::save(                   \
    cereal::PortableBinaryOutputArchive&) const;                               \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::load(                   \
    cereal::PortableBinaryInputArchive&);                                      \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::save(                   \
    RootedPortableBinaryOutputArchive&) const;                                 \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::load(                   \
    RootedPortableBinaryInputArchive&)
#else
#define LBANN_ADD_PORTABLE_BINARY_SERIALIZE_ETI(...)
#endif

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
#define LBANN_ADD_XML_SERIALIZE_ETI(T)                                         \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::save(                   \
    cereal::XMLOutputArchive&) const;                                          \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::load(                   \
    cereal::XMLInputArchive&);                                                 \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::save(                   \
    RootedXMLOutputArchive&) const;                                            \
  template void ::lbann::PermuteLayer<T>::PermuteImpl::load(                   \
    RootedXMLInputArchive&)
#else
#define LBANN_ADD_XML_SERIALIZE_ETI(...)
#endif

#include "lbann/macros/common_cereal_registration.hpp"
#define PROTO(T)                                                               \
  LBANN_ADD_BINARY_SAVE_LOAD_ETI(T);                                           \
  LBANN_ADD_JSON_SERIALIZE_ETI(T);                                             \
  LBANN_ADD_PORTABLE_BINARY_SERIALIZE_ETI(T);                                  \
  LBANN_ADD_XML_SERIALIZE_ETI(T);                                              \
                                                                               \
  LBANN_ADD_ALL_SERIALIZE_ETI(::lbann::PermuteLayer<T>);                       \
  CEREAL_REGISTER_TYPE_WITH_NAME(::lbann::PermuteLayer<T>,                     \
                                 "PermuteLayer(" #T ")")
#define LBANN_INSTANTIATE_GPU_HALF
#include <lbann/macros/instantiate.hpp>
