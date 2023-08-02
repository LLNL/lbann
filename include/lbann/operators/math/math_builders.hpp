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
#ifndef LBANN_INCLUDE_LBANN_OPERATORS_MATH_MATH_BUILDERS_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_MATH_MATH_BUILDERS_HPP_INCLUDED

#include "lbann/operators/builder_macros.hpp"
#include "lbann/operators/operator.hpp"

namespace lbann {

template <typename DataT, El::Device D>
std::unique_ptr<Operator<DataT, El::Base<DataT>, D>>
build_abs_operator(lbann_data::Operator const& op);

LBANN_DECLARE_OPERATOR_BUILDER(acos);
LBANN_DECLARE_OPERATOR_BUILDER(acosh);
LBANN_DECLARE_OPERATOR_BUILDER(add);
LBANN_DECLARE_OPERATOR_BUILDER(add_constant);
LBANN_DECLARE_OPERATOR_BUILDER(asin);
LBANN_DECLARE_OPERATOR_BUILDER(asinh);
LBANN_DECLARE_OPERATOR_BUILDER(atan);
LBANN_DECLARE_OPERATOR_BUILDER(atanh);
LBANN_DECLARE_OPERATOR_BUILDER(ceil);
LBANN_DECLARE_OPERATOR_BUILDER(clamp);
LBANN_DECLARE_OPERATOR_BUILDER(constant_subtract);
LBANN_DECLARE_OPERATOR_BUILDER(cos);
LBANN_DECLARE_OPERATOR_BUILDER(cosh);
LBANN_DECLARE_OPERATOR_BUILDER(divide);
LBANN_DECLARE_OPERATOR_BUILDER(equal);
LBANN_DECLARE_OPERATOR_BUILDER(equal_constant);
LBANN_DECLARE_OPERATOR_BUILDER(erf);
LBANN_DECLARE_OPERATOR_BUILDER(erfinv);
LBANN_DECLARE_OPERATOR_BUILDER(exp);
LBANN_DECLARE_OPERATOR_BUILDER(expm1);
LBANN_DECLARE_OPERATOR_BUILDER(floor);
LBANN_DECLARE_OPERATOR_BUILDER(greater);
LBANN_DECLARE_OPERATOR_BUILDER(greater_constant);
LBANN_DECLARE_OPERATOR_BUILDER(greater_equal);
LBANN_DECLARE_OPERATOR_BUILDER(greater_equal_constant);
LBANN_DECLARE_OPERATOR_BUILDER(less);
LBANN_DECLARE_OPERATOR_BUILDER(less_constant);
LBANN_DECLARE_OPERATOR_BUILDER(less_equal);
LBANN_DECLARE_OPERATOR_BUILDER(less_equal_constant);
LBANN_DECLARE_OPERATOR_BUILDER(log);
LBANN_DECLARE_OPERATOR_BUILDER(log1p);
LBANN_DECLARE_OPERATOR_BUILDER(logical_and);
LBANN_DECLARE_OPERATOR_BUILDER(logical_not);
LBANN_DECLARE_OPERATOR_BUILDER(logical_or);
LBANN_DECLARE_OPERATOR_BUILDER(logical_xor);
LBANN_DECLARE_OPERATOR_BUILDER(max);
LBANN_DECLARE_OPERATOR_BUILDER(max_constant);
LBANN_DECLARE_OPERATOR_BUILDER(min);
LBANN_DECLARE_OPERATOR_BUILDER(min_constant);
LBANN_DECLARE_OPERATOR_BUILDER(mod);
LBANN_DECLARE_OPERATOR_BUILDER(multiply);
LBANN_DECLARE_OPERATOR_BUILDER(negative);
LBANN_DECLARE_OPERATOR_BUILDER(not_equal);
LBANN_DECLARE_OPERATOR_BUILDER(not_equal_constant);
LBANN_DECLARE_OPERATOR_BUILDER(pow);
LBANN_DECLARE_OPERATOR_BUILDER(reciprocal);
LBANN_DECLARE_OPERATOR_BUILDER(round);
LBANN_DECLARE_OPERATOR_BUILDER(rsqrt);
LBANN_DECLARE_OPERATOR_BUILDER(safe_divide);
LBANN_DECLARE_OPERATOR_BUILDER(safe_reciprocal);
LBANN_DECLARE_OPERATOR_BUILDER(scale);
LBANN_DECLARE_OPERATOR_BUILDER(select);
LBANN_DECLARE_OPERATOR_BUILDER(sign);
LBANN_DECLARE_OPERATOR_BUILDER(sin);
LBANN_DECLARE_OPERATOR_BUILDER(sinh);
LBANN_DECLARE_OPERATOR_BUILDER(sqrt);
LBANN_DECLARE_OPERATOR_BUILDER(square);
LBANN_DECLARE_OPERATOR_BUILDER(squared_difference);
LBANN_DECLARE_OPERATOR_BUILDER(subtract);
LBANN_DECLARE_OPERATOR_BUILDER(subtract_constant);
LBANN_DECLARE_OPERATOR_BUILDER(tan);
LBANN_DECLARE_OPERATOR_BUILDER(tanh);

} // namespace lbann

#endif // LBANN_INCLUDE_LBANN_OPERATORS_MATH_MATH_BUILDERS_HPP_INCLUDED
