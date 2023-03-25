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

#include <lbann/macros/common_cereal_registration.hpp>

#include <cereal/types/polymorphic.hpp>

/** @file
 *
 *  Define LBANN_LAYER_NAME to be the full layer class name before
 *  including this file. Don't include this file inside the lbann
 *  namespace.
 */

#undef LBANN_COMMA
#undef LBANN_REGISTER_LAYER_WITH_CEREAL_BASE
#undef LBANN_REGISTER_LAYER_WITH_CEREAL
#undef PROTO_DEVICE
#undef PROTO

#define LBANN_COMMA ,

#define LBANN_REGISTER_LAYER_WITH_CEREAL_BASE(NAME, TYPE, LAYOUT, DEVICE)      \
  LBANN_ADD_ALL_SERIALIZE_ETI(                                                 \
    ::lbann::NAME<TYPE, ::lbann::data_layout::LAYOUT, DEVICE>);                \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                              \
    ::lbann::NAME<                                                             \
      TYPE LBANN_COMMA ::lbann::data_layout::LAYOUT LBANN_COMMA DEVICE>,       \
    #NAME "(" #TYPE "," #LAYOUT "," #DEVICE ")")

#define LBANN_REGISTER_LAYER_WITH_CEREAL(NAME, TYPE, DEVICE)                   \
  LBANN_REGISTER_LAYER_WITH_CEREAL_BASE(NAME, TYPE, DATA_PARALLEL, DEVICE);    \
  LBANN_REGISTER_LAYER_WITH_CEREAL_BASE(NAME, TYPE, MODEL_PARALLEL, DEVICE)

#define PROTO_DEVICE(T, D)                                                     \
  LBANN_REGISTER_LAYER_WITH_CEREAL(LBANN_LAYER_NAME, T, D)
#include "instantiate_device.hpp"
#undef PROTO
#undef PROTO_DEVICE
#undef LBANN_REGISTER_LAYER_WITH_CEREAL
#undef LBANN_REGISTER_LAYER_WITH_CEREAL_BASE
#undef LBANN_COMMA

// Finally, add the dynamic init registration
LBANN_REGISTER_DYNAMIC_INIT(LBANN_LAYER_NAME)
