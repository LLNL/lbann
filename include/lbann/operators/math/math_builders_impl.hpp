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
#ifndef LBANN_INCLUDE_LBANN_OPERATORS_MATH_MATH_BUILDERS_IMPL_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_MATH_MATH_BUILDERS_IMPL_HPP_INCLUDED

#include "lbann/operators/math/math_builders.hpp"

#include "lbann/operators/math/abs.hpp"
#include "lbann/operators/math/binary.hpp"
#include "lbann/operators/math/binary_with_constant.hpp"
#include "lbann/operators/math/clamp.hpp"
#include "lbann/operators/math/select.hpp"
#include "lbann/operators/math/unary.hpp"

#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/operators.pb.h"

template <typename DataT, El::Device D>
std::unique_ptr<lbann::Operator<DataT, DataT, D>>
lbann::build_clamp_operator(lbann_data::Operator const& op)
{
  details::AssertConsistentTypeParameters<DataT, DataT, D>(op);
  lbann_data::ClampOperator params;
  LBANN_ASSERT(op.parameters().UnpackTo(&params));
  return std::make_unique<ClampOperator<DataT, D>>(params.min(), params.max());
}

template <typename DataT, El::Device D>
std::unique_ptr<lbann::Operator<DataT, El::Base<DataT>, D>>
lbann::build_abs_operator(lbann_data::Operator const& op)
{
  details::AssertConsistentTypeParameters<DataT, El::Base<DataT>, D>(op);
  return std::make_unique<AbsOperator<DataT, D>>();
}

template <typename DataT, El::Device D>
std::unique_ptr<lbann::Operator<DataT, DataT, D>>
lbann::build_select_operator(lbann_data::Operator const& op)
{
  details::AssertConsistentTypeParameters<DataT, DataT, D>(op);
  lbann_data::SelectOperator params;
  LBANN_ASSERT(op.parameters().UnpackTo(&params));
  return std::make_unique<SelectOperator<DataT, D>>(params.value(),
                                                    params.constant_if_true(),
                                                    params.constant_if_false(),
                                                    params.value_if_true(),
                                                    params.value_if_false(),
                                                    params.epsilon());
}

#define LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(OP_NAME, OP_LOWER_NAME)         \
  template <typename DataT, El::Device D>                                      \
  std::unique_ptr<lbann::Operator<DataT, DataT, D>>                            \
    lbann::build_##OP_LOWER_NAME##_operator(lbann_data::Operator const& op)    \
  {                                                                            \
    details::AssertConsistentTypeParameters<DataT, DataT, D>(op);              \
    lbann_data::OP_NAME##Operator params;                                      \
    LBANN_ASSERT(op.parameters().UnpackTo(&params));                           \
    return std::make_unique<OP_NAME##Operator<DataT, D>>(params.constant());   \
  }

LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(AddConstant, add_constant)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(ConstantSubtract, constant_subtract)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(EqualConstant, equal_constant)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(GreaterConstant, greater_constant)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(GreaterEqualConstant,
                                       greater_equal_constant)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(LessConstant, less_constant)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(LessEqualConstant, less_equal_constant)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(MaxConstant, max_constant)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(MinConstant, min_constant)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(NotEqualConstant, not_equal_constant)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(Scale, scale)
LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER(SubtractConstant, subtract_constant)

#undef LBANN_DEFINE_BIN_WITH_CONSTANT_BUILDER

LBANN_DEFINE_OPERATOR_BUILDER(acos, Acos)
LBANN_DEFINE_OPERATOR_BUILDER(acosh, Acosh)
LBANN_DEFINE_OPERATOR_BUILDER(add, Add)
LBANN_DEFINE_OPERATOR_BUILDER(asin, Asin)
LBANN_DEFINE_OPERATOR_BUILDER(asinh, Asinh)
LBANN_DEFINE_OPERATOR_BUILDER(atan, Atan)
LBANN_DEFINE_OPERATOR_BUILDER(atanh, Atanh)
LBANN_DEFINE_OPERATOR_BUILDER(ceil, Ceil)
LBANN_DEFINE_OPERATOR_BUILDER(cos, Cos)
LBANN_DEFINE_OPERATOR_BUILDER(cosh, Cosh)
LBANN_DEFINE_OPERATOR_BUILDER(divide, Divide)
LBANN_DEFINE_OPERATOR_BUILDER(equal, Equal)
LBANN_DEFINE_OPERATOR_BUILDER(erf, Erf)
LBANN_DEFINE_OPERATOR_BUILDER(erfinv, ErfInv)
LBANN_DEFINE_OPERATOR_BUILDER(exp, Exp)
LBANN_DEFINE_OPERATOR_BUILDER(expm1, Expm1)
LBANN_DEFINE_OPERATOR_BUILDER(floor, Floor)
LBANN_DEFINE_OPERATOR_BUILDER(gelu, Gelu)
LBANN_DEFINE_OPERATOR_BUILDER(gelunew, GeluNew)
LBANN_DEFINE_OPERATOR_BUILDER(greater, Greater)
LBANN_DEFINE_OPERATOR_BUILDER(greater_equal, GreaterEqual)
LBANN_DEFINE_OPERATOR_BUILDER(less, Less)
LBANN_DEFINE_OPERATOR_BUILDER(less_equal, LessEqual)
LBANN_DEFINE_OPERATOR_BUILDER(log, Log)
LBANN_DEFINE_OPERATOR_BUILDER(log1p, Log1p)
LBANN_DEFINE_OPERATOR_BUILDER(logical_and, LogicalAnd)
LBANN_DEFINE_OPERATOR_BUILDER(logical_not, LogicalNot)
LBANN_DEFINE_OPERATOR_BUILDER(logical_or, LogicalOr)
LBANN_DEFINE_OPERATOR_BUILDER(logical_xor, LogicalXor)
LBANN_DEFINE_OPERATOR_BUILDER(max, Max)
LBANN_DEFINE_OPERATOR_BUILDER(min, Min)
LBANN_DEFINE_OPERATOR_BUILDER(mod, Mod)
LBANN_DEFINE_OPERATOR_BUILDER(multiply, Multiply)
LBANN_DEFINE_OPERATOR_BUILDER(negative, Negative)
LBANN_DEFINE_OPERATOR_BUILDER(not_equal, NotEqual)
LBANN_DEFINE_OPERATOR_BUILDER(pow, Pow)
LBANN_DEFINE_OPERATOR_BUILDER(reciprocal, Reciprocal)
LBANN_DEFINE_OPERATOR_BUILDER(round, Round)
LBANN_DEFINE_OPERATOR_BUILDER(rsqrt, Rsqrt)
LBANN_DEFINE_OPERATOR_BUILDER(safe_divide, SafeDivide)
LBANN_DEFINE_OPERATOR_BUILDER(safe_reciprocal, SafeReciprocal)
LBANN_DEFINE_OPERATOR_BUILDER(sign, Sign)
LBANN_DEFINE_OPERATOR_BUILDER(sin, Sin)
LBANN_DEFINE_OPERATOR_BUILDER(sinh, Sinh)
LBANN_DEFINE_OPERATOR_BUILDER(sqrt, Sqrt)
LBANN_DEFINE_OPERATOR_BUILDER(square, Square)
LBANN_DEFINE_OPERATOR_BUILDER(squared_difference, SquaredDifference)
LBANN_DEFINE_OPERATOR_BUILDER(subtract, Subtract)
LBANN_DEFINE_OPERATOR_BUILDER(tan, Tan)
LBANN_DEFINE_OPERATOR_BUILDER(tanh, Tanh)
#endif // LBANN_INCLUDE_LBANN_OPERATORS_MATH_MATH_BUILDERS_IMPL_HPP_INCLUDED
