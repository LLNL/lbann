////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_INCLUDE_LBANN_OPERATORS_MATH_BINARY_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_MATH_BINARY_HPP_INCLUDED

#include "lbann/operators/declare_stateless_op.hpp"

namespace lbann {

// Arithmetic operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Add, "add");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Subtract, "subtract");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Multiply, "multiply");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Divide, "divide");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Mod, "modulo");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Pow, "power");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(SafeDivide, "safe divide");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(SquaredDifference,
                                             "squared difference");

// Comparison operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Max, "maximum");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Min, "minimum");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Equal, "equal");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(NotEqual, "not equal");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Less, "less than");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LessEqual, "less than or equal");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Greater, "greater than");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(GreaterEqual,
                                             "greater than or equal");

// Logical operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LogicalAnd, "logical and");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LogicalOr, "logical or");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LogicalXor, "logical xor");

} // namespace lbann

#endif // LBANN_INCLUDE_LBANN_OPERATORS_MATH_BINARY_HPP_INCLUDED
