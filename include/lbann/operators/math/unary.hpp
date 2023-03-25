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
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LogicalNot, "logical not");

// Sign operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Negative, "negative");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Sign, "sign");

// Rounding operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Round, "round");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Ceil, "ceil");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Floor, "floor");

// Power operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Reciprocal, "reciprocal");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Square, "square");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Sqrt, "square root");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Rsqrt, "reciprocal square root");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(SafeReciprocal, "safe reciprocal");

// Exponential and logarithmic operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Exp, "exponential");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Expm1, "expm1");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Log, "natural logarithm");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Log1p, "log1p");

// Trigonometric operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Cos, "cosine");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Sin, "sine");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Tan, "tangent");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Acos, "arccosine");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Asin, "arcsine");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Atan, "arctangent");

// Hyperbolic operations
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Cosh, "hyperbolic cosine");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Sinh, "hyperbolic sine");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Tanh, "hyperbolic tangent");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Acosh, "hyperbolic arccosine");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Asinh, "hyperbolic arcsine");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Atanh, "hyperbolic arctangent");

// Error function
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Erf, "error function");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(ErfInv, "inverse error function");

} // namespace lbann

#endif // LBANN_INCLUDE_LBANN_OPERATORS_MATH_UNARY_HPP_INCLUDED
