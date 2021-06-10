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

#include <lbann/operators/math/clamp.hpp>
#include <lbann/operators/math/math_builders.hpp>

#include <lbann/proto/proto_common.hpp>
#include <operators.pb.h>

namespace lbann
{

template <typename TensorDataType>
std::unique_ptr<Operator> build_clamp_operator_from_pbuf(
  lbann_data::Operator const& proto_operator)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_operator, clamp);
  using OperatorType = ClampOperator<TensorDataType>;
  auto const& params = proto_operator.clamp();
  return lbann::make_unique<OperatorType>(
    El::To<TensorDataType>(params.min()),
    El::To<TensorDataType>(params.max()));
}


#define PROTO(T)                               \
  LBANN_OPERATOR_BUILDER_ETI(clamp, T);
#include <lbann/macros/instantiate.hpp>
} // namespace lbann
