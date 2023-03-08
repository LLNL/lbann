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

#include "lbann_config.hpp"

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
#define LBANN_ADD_BINARY_SERIALIZE_ETI(...)                                    \
  template void __VA_ARGS__::serialize(cereal::BinaryOutputArchive&);          \
  template void __VA_ARGS__::serialize(cereal::BinaryInputArchive&);           \
  template void __VA_ARGS__::serialize(RootedBinaryOutputArchive&);            \
  template void __VA_ARGS__::serialize(RootedBinaryInputArchive&)
#else
#define LBANN_ADD_BINARY_SERIALIZE_ETI(...)
#endif

#ifdef LBANN_HAS_CEREAL_JSON_ARCHIVES
#define LBANN_ADD_JSON_SERIALIZE_ETI(...)                                      \
  template void __VA_ARGS__::serialize(cereal::JSONOutputArchive&);            \
  template void __VA_ARGS__::serialize(cereal::JSONInputArchive&);             \
  template void __VA_ARGS__::serialize(RootedJSONOutputArchive&);              \
  template void __VA_ARGS__::serialize(RootedJSONInputArchive&)
#else
#define LBANN_ADD_JSON_SERIALIZE_ETI(...)
#endif

#ifdef LBANN_HAS_CEREAL_PORTABLE_BINARY_ARCHIVES
#define LBANN_ADD_PORTABLE_BINARY_SERIALIZE_ETI(...)                           \
  template void __VA_ARGS__::serialize(cereal::PortableBinaryOutputArchive&);  \
  template void __VA_ARGS__::serialize(cereal::PortableBinaryInputArchive&);   \
  template void __VA_ARGS__::serialize(RootedPortableBinaryOutputArchive&);    \
  template void __VA_ARGS__::serialize(RootedPortableBinaryInputArchive&)
#else
#define LBANN_ADD_PORTABLE_BINARY_SERIALIZE_ETI(...)
#endif

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
#define LBANN_ADD_XML_SERIALIZE_ETI(...)                                       \
  template void __VA_ARGS__::serialize(cereal::XMLOutputArchive&);             \
  template void __VA_ARGS__::serialize(cereal::XMLInputArchive&);              \
  template void __VA_ARGS__::serialize(RootedXMLOutputArchive&);               \
  template void __VA_ARGS__::serialize(RootedXMLInputArchive&)
#else
#define LBANN_ADD_XML_SERIALIZE_ETI(...)
#endif

#define LBANN_ADD_ALL_SERIALIZE_ETI(...)                                       \
  LBANN_ADD_BINARY_SERIALIZE_ETI(__VA_ARGS__);                                 \
  LBANN_ADD_JSON_SERIALIZE_ETI(__VA_ARGS__);                                   \
  LBANN_ADD_PORTABLE_BINARY_SERIALIZE_ETI(__VA_ARGS__);                        \
  LBANN_ADD_XML_SERIALIZE_ETI(__VA_ARGS__)

#define LBANN_REGISTER_DYNAMIC_INIT(NAME) CEREAL_REGISTER_DYNAMIC_INIT(NAME)
