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

#include <lbann/utils/serialization/serialize_matrices.hpp>
#include <lbann/utils/serialization/serialize_matrices_impl.hpp>

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

// Enumerate valid DistMatrix permutations and add Matrix.
#define REGISTER_ALL_DISTMATRIX(TYPE, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, CIRC, CIRC, DEVICE)                                \
  REGISTER_DISTMATRIX(TYPE, MC, MR, DEVICE)                                    \
  REGISTER_DISTMATRIX(TYPE, MC, STAR, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, MD, STAR, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, MR, MC, DEVICE)                                    \
  REGISTER_DISTMATRIX(TYPE, MR, STAR, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, STAR, MC, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, STAR, MD, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, STAR, MR, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, STAR, STAR, DEVICE)                                \
  REGISTER_DISTMATRIX(TYPE, STAR, VC, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, STAR, VR, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, VC, STAR, DEVICE)                                  \
  REGISTER_DISTMATRIX(TYPE, VR, STAR, DEVICE)                                  \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                              \
    El::Matrix<TYPE LBANN_COMMA El::Device::DEVICE>,                           \
    "Matrix(" #TYPE "," #DEVICE ")")                                           \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(                                        \
    El::AbstractMatrix<TYPE>,                                                  \
    El::Matrix<TYPE LBANN_COMMA El::Device::DEVICE>)

// Enumerate the valid device types. This is done here rather than
// "instantiate_device.hpp" because it's a miniscule overhead and the
// slightly prettier printing is worth it to me -- the namespace and
// enum class are irrelevant to users!
#if defined LBANN_HAS_GPU
#define REGISTER_ALL_DISTMATRIX_DEVICES(TYPE)                                  \
  REGISTER_ALL_DISTMATRIX(TYPE, CPU)                                           \
  REGISTER_ALL_DISTMATRIX(TYPE, GPU)
#else
#define REGISTER_ALL_DISTMATRIX_DEVICES(TYPE) REGISTER_ALL_DISTMATRIX(TYPE, CPU)
#endif // defined LBANN_HAS_GPU

// Enumerate all the valid data types.
#define PROTO(T) REGISTER_ALL_DISTMATRIX_DEVICES(T)
#define LBANN_INSTANTIATE_DOUBLE
#include <lbann/macros/instantiate.hpp>

CEREAL_REGISTER_DYNAMIC_INIT(DistMatrixTypes);
