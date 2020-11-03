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

#include <lbann/utils/serialization/serialize_matrices.hpp>

// Register types outside Cereal's namespace.
#define LBANN_COMMA ,

#define REGISTER_DISTMATRIX(TYPE, CDIST, RDIST, DEVICE)                 \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                       \
    El::DistMatrix<TYPE LBANN_COMMA El::CDIST LBANN_COMMA El::RDIST LBANN_COMMA El::ELEMENT LBANN_COMMA DEVICE>, \
    "DistMatrix<" #TYPE "," #CDIST "," #RDIST "," #DEVICE ">")          \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(                                 \
    El::AbstractDistMatrix<TYPE>,                                       \
    El::DistMatrix<TYPE LBANN_COMMA El::CDIST LBANN_COMMA El::RDIST LBANN_COMMA El::ELEMENT LBANN_COMMA DEVICE>)

#define REGISTER_ALL_DISTMATRIX(TYPE, DEVICE)     \
  REGISTER_DISTMATRIX(TYPE, CIRC, CIRC, DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, MC  , MR  , DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, MC  , STAR, DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, MD  , STAR, DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, MR  , MC  , DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, MR  , STAR, DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, STAR, MC  , DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, STAR, MD  , DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, STAR, MR  , DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, STAR, STAR, DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, STAR, VC  , DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, STAR, VR  , DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, VC  , STAR, DEVICE)   \
  REGISTER_DISTMATRIX(TYPE, VR  , STAR, DEVICE)

#define PROTO_DEVICE(T, D)                      \
  REGISTER_ALL_DISTMATRIX(T, D)                 \
  CEREAL_REGISTER_TYPE_WITH_NAME(               \
    El::Matrix<T LBANN_COMMA D>,                \
    "Matrix<" #T "," #D ">")
#include <lbann/macros/instantiate_device.hpp>

//CEREAL_REGISTER_DYNAMIC_INIT(LBANNMatrixTypeRegistration)
