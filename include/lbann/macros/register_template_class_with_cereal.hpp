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

/** @file
 *
 *  Define LBANN_CLASS_NAME to be the full class name before
 *  including this file. Don't include this file inside the lbann
 *  namespace.
 */

#define LBANN_STRINGIFY(STR) LBANN_STRINGIFY_IMPL(STR)
#define LBANN_STRINGIFY_IMPL(STR) #STR

#undef PROTO
#define PROTO(T)                                                        \
  LBANN_ADD_ALL_SERIALIZE_ETI(::lbann::LBANN_CLASS_NAME<T>);            \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                       \
    lbann::LBANN_CLASS_NAME<T>,                                         \
    LBANN_STRINGIFY(LBANN_CLASS_NAME) "(" #T ")")

#include "lbann/macros/instantiate.hpp"
#undef PROTO
