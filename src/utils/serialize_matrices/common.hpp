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

#include <cereal/types/polymorphic.hpp>
#include <lbann/utils/serialization/serialize_matrices.hpp>

#define LBANN_INSTANTIATE_DOUBLE

// Register types outside Cereal's namespace.
#define LBANN_COMMA ,

// Handle general DistMatrix registration.
#define REGISTER_DISTMATRIX(TYPE, CDIST, RDIST, DEVICE)                        \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                              \
    El::DistMatrix<TYPE LBANN_COMMA El::CDIST LBANN_COMMA El::RDIST            \
                     LBANN_COMMA El::ELEMENT LBANN_COMMA El::Device::DEVICE>,  \
    "DistMatrix(" #TYPE "," #CDIST "," #RDIST "," #DEVICE ")")                 \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(                                        \
    El::AbstractDistMatrix<TYPE>,                                              \
    El::DistMatrix<TYPE LBANN_COMMA El::CDIST LBANN_COMMA El::RDIST            \
                     LBANN_COMMA El::ELEMENT LBANN_COMMA El::Device::DEVICE>)
