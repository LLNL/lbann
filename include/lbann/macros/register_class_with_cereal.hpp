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

#include <cereal/types/polymorphic.hpp>

/** @file
 *
 *  Define LBANN_CLASS_NAME to be the full class name before
 *  including this file. Don't include this file inside the lbann
 *  namespace.
 */


#define CEREAL_REGISTER_SIMPLE_CLASS_BASE(CLS_NAME, ARCHIVE_TYPE)           \
  template void ::lbann::CLS_NAME::serialize(ARCHIVE_TYPE&)

#define CEREAL_REGISTER_SIMPLE_CLASS(CLS_NAME)                              \
  CEREAL_REGISTER_SIMPLE_CLASS_BASE(CLS_NAME, cereal::XMLOutputArchive);    \
  CEREAL_REGISTER_SIMPLE_CLASS_BASE(CLS_NAME, cereal::XMLInputArchive);     \
  CEREAL_REGISTER_SIMPLE_CLASS_BASE(CLS_NAME, cereal::BinaryOutputArchive); \
  CEREAL_REGISTER_SIMPLE_CLASS_BASE(CLS_NAME, cereal::BinaryInputArchive);  \
  CEREAL_REGISTER_SIMPLE_CLASS_BASE(CLS_NAME, RootedXMLOutputArchive);      \
  CEREAL_REGISTER_SIMPLE_CLASS_BASE(CLS_NAME, RootedXMLInputArchive);       \
  CEREAL_REGISTER_SIMPLE_CLASS_BASE(CLS_NAME, RootedBinaryOutputArchive);   \
  CEREAL_REGISTER_SIMPLE_CLASS_BASE(CLS_NAME, RootedBinaryInputArchive)

CEREAL_REGISTER_SIMPLE_CLASS(LBANN_CLASS_NAME);

#undef CEREAL_REGISTER_SIMPLE_CLASS
#undef CEREAL_REGISTER_SIMPLE_CLASS_BASE
