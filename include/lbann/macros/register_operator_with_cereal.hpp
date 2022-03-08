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

#include <lbann/macros/common_cereal_registration.hpp>

#include <cereal/types/polymorphic.hpp>

/** @file
 *
 *  Define LBANN_OPERATOR_NAME to be the full operator class name before
 *  including this file. Don't include this file inside the lbann
 *  namespace.
 */

#undef LBANN_COMMA
#undef LBANN_REGISTER_OPERATOR_WITH_CEREAL_BASE
#undef LBANN_REGISTER_OPERATOR_WITH_CEREAL
#undef PROTO_DEVICE
#undef PROTO

#define LBANN_COMMA ,

#define LBANN_REGISTER_OPERATOR_WITH_CEREAL_BASE(NAME, TYPE, DEVICE)           \
  LBANN_ADD_ALL_SERIALIZE_ETI(::lbann::NAME<TYPE LBANN_COMMA DEVICE>);         \
  CEREAL_REGISTER_TYPE_WITH_NAME(::lbann::NAME<TYPE LBANN_COMMA DEVICE>,       \
                                 #NAME "(" #TYPE "," #DEVICE ")")

#define LBANN_REGISTER_OPERATOR_WITH_CEREAL(NAME, TYPE, DEVICE)                \
  LBANN_REGISTER_OPERATOR_WITH_CEREAL_BASE(NAME, TYPE, DEVICE)

#define PROTO_DEVICE(T, D)                                                     \
  LBANN_REGISTER_OPERATOR_WITH_CEREAL(LBANN_OPERATOR_NAME, T, D)

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO
#undef LBANN_REGISTER_OPERATOR_WITH_CEREAL
#undef LBANN_REGISTER_OPERATOR_WITH_CEREAL_BASE
#undef LBANN_COMMA

LBANN_REGISTER_DYNAMIC_INIT(LBANN_OPERATOR_NAME);
