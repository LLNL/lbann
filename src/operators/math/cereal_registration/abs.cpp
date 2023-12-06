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

#include "lbann/operators/math/abs.hpp"

#include "lbann/utils/serialize.hpp"

#define LBANN_OPERATOR_NAME AbsOperator
#include <lbann/macros/register_operator_with_cereal.hpp>

// This is just sort of a hack for now.
#include <cereal/types/polymorphic.hpp>
#include <lbann/macros/common_cereal_registration.hpp>

#undef LBANN_COMMA
#undef LBANN_REGISTER_OPERATOR_WITH_CEREAL_BASE
#undef LBANN_REGISTER_OPERATOR_WITH_CEREAL
#undef PROTO_DEVICE
#undef PROTO

#define LBANN_COMMA ,
#define LBANN_REGISTER_OPERATOR_WITH_CEREAL(TYPE, DEVICE)                      \
  LBANN_ADD_ALL_SERIALIZE_ETI(::lbann::AbsOperator<TYPE LBANN_COMMA DEVICE>);  \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                              \
    ::lbann::AbsOperator<TYPE LBANN_COMMA DEVICE>,                             \
    "AbsOperator(" #TYPE "," #DEVICE ")")

#define PROTO_DEVICE(T, D) LBANN_REGISTER_OPERATOR_WITH_CEREAL(T, D)

PROTO_DEVICE(El::Complex<float>, El::Device::CPU);
#ifdef LBANN_HAS_DOUBLE
PROTO_DEVICE(El::Complex<double>, El::Device::CPU);
#endif // LBANN_HAS_DOUBLE

#ifdef LBANN_HAS_GPU
PROTO_DEVICE(El::Complex<float>, El::Device::GPU);
#ifdef LBANN_HAS_DOUBLE
PROTO_DEVICE(El::Complex<double>, El::Device::GPU);
#endif // LBANN_HAS_DOUBLE
#endif // LBANN_HAS_GPU
