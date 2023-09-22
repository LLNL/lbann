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

#ifndef LBANN_INCLUDE_LBANN_OPERATORS_MATH_UNARY_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_MATH_UNARY_HPP_INCLUDED

#include "lbann/operators/declare_stateless_op.hpp"

namespace lbann {

// These are all single-type operators.

// Logical operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LogicalNot, "logical not", false);

// Sign operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Negative, "negative", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Sign, "sign", false);

// Rounding operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Round, "round", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Ceil, "ceil", false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Floor, "floor", false);

// Power operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Reciprocal, "reciprocal", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Square, "square", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Sqrt, "square root", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Rsqrt,
                                             "reciprocal square root",
                                             true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(SafeReciprocal,
                                             "safe reciprocal",
                                             true);

// Exponential and logarithmic operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Exp, "exponential", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Expm1, "expm1", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Log, "natural logarithm", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Log1p, "log1p", true);

// Trigonometric operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Cos, "cosine", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Sin, "sine", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Tan, "tangent", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Acos, "arccosine", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Asin, "arcsine", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Atan, "arctangent", true);

// Hyperbolic operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Cosh, "hyperbolic cosine", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Sinh, "hyperbolic sine", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Tanh, "hyperbolic tangent", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Acosh,
                                             "hyperbolic arccosine",
                                             true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Asinh, "hyperbolic arcsine", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Atanh,
                                             "hyperbolic arctangent",
                                             true);

// Error function
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Erf, "error function", true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(ErfInv,
                                             "inverse error function",
                                             true);

// Probabilistic operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Gelu,
                                             "gaussian error linear unit",
                                             true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(GeluNew,
                                             "gaussian error linear unit new",
                                             true);

} // namespace lbann

#endif // LBANN_INCLUDE_LBANN_OPERATORS_MATH_UNARY_HPP_INCLUDED
