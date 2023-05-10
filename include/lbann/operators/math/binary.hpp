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

#ifndef LBANN_INCLUDE_LBANN_OPERATORS_MATH_BINARY_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_MATH_BINARY_HPP_INCLUDED

#include "lbann/operators/declare_stateless_op.hpp"

namespace lbann {

// Arithmetic operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Add, "add", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Subtract, "subtract", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Multiply, "multiply", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Divide, "divide", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Mod, "modulo", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Pow, "power", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(SafeDivide, "safe divide", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(SquaredDifference,
                                             "squared difference",
                                             true);

// Comparison operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Max, "maximum", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Min, "minimum", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Equal, "equal", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(NotEqual, "not equal", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Less, "less than", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LessEqual,
                                             "less than or equal",
                                             false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Greater, "greater than", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(GreaterEqual,
                                             "greater than or equal",
                                             false);

// Logical operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LogicalAnd, "logical and", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LogicalOr, "logical or", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LogicalXor, "logical xor", false);

} // namespace lbann

#endif // LBANN_INCLUDE_LBANN_OPERATORS_MATH_BINARY_HPP_INCLUDED
