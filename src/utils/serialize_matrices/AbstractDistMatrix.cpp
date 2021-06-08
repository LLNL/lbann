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

#include "common.hpp"
#include <lbann/utils/serialization/serialize_matrices.hpp>
#include <lbann/utils/serialization/serialize_matrices_impl.hpp>

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
#define ETI_BINARY_SAVELOAD(TYPE)                                      \
  template void El::load(cereal::BinaryInputArchive&,                          \
                         El::AbstractDistMatrix<TYPE>&);                       \
  template void El::save(cereal::BinaryOutputArchive&,                         \
                         El::AbstractDistMatrix<TYPE> const&);
#else
#define ETI_BINARY_SAVELOAD(TYPE)
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_JSON_ARCHIVES // Not yet supported
#define ETI_JSON_SAVELOAD(TYPE)                                        \
  template void El::load(cereal::JSONInputArchive&,                            \
                         El::AbstractDistMatrix<TYPE>&);                       \
  template void El::save(cereal::JSONOutputArchive&,                           \
                         El::AbstractDistMatrix<TYPE> const&);
#else
#define ETI_JSON_SAVELOAD(TYPE)
#endif // LBANN_HAS_CEREAL_JSON_ARCHIVES

#ifdef LBANN_HAS_CEREAL_PORTABLE_BINARY_ARCHIVES // Not yet supported
#define ETI_PORTABLEBINARY_SAVELOAD(TYPE)                              \
  template void El::load(cereal::PortableBinaryInputArchive&,                  \
                         El::AbstractDistMatrix<TYPE>&);                       \
  template void El::save(cereal::PortableBinaryOutputArchive&,                 \
                         El::AbstractDistMatrix<TYPE> const&);
#else
#define ETI_PORTABLEBINARY_SAVELOAD(TYPE)
#endif // LBANN_HAS_CEREAL_PORTABLE_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
#define ETI_XML_SAVELOAD(TYPE)                                         \
  template void El::load(cereal::XMLInputArchive&,                             \
                         El::AbstractDistMatrix<TYPE>&);                       \
  template void El::save(cereal::XMLOutputArchive&,                            \
                         El::AbstractDistMatrix<TYPE> const&);
#else
#define ETI_XML_SAVELOAD(TYPE)
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES

// Enumerate all the valid data types.
#define PROTO(T)                                                               \
  ETI_BINARY_SAVELOAD(T)                                                       \
  ETI_JSON_SAVELOAD(T)                                                         \
  ETI_PORTABLEBINARY_SAVELOAD(T)                                               \
  ETI_XML_SAVELOAD(T)

#include <lbann/macros/instantiate.hpp>
