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

#ifdef LBANN_HAS_ONNX
#define LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(OP_NAME,                         \
                                              OP_STRING,                       \
                                              OP_ONNX_NAME)                    \
  LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(OP_NAME, OP_STRING);            \
  template <typename T, El::Device D>                                          \
  std::vector<onnx::NodeProto> get_onnx_nodes_impl(                            \
    OP_NAME##Operator<T, D> const& op)                                         \
  {                                                                            \
    std::vector<onnx::NodeProto> nodes(1UL);                                   \
    nodes.front().set_op_type(OP_ONNX_NAME);                                   \
    return nodes;                                                              \
  }
#else
#define LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(OP_NAME,                         \
                                              OP_STRING,                       \
                                              OP_ONNX_NAME)                    \
  LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(OP_NAME, OP_STRING)
#endif // LBANN_HAS_ONNX

namespace lbann {

// Arithmetic operations
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Add, "add", "Add")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Subtract, "subtract", "Sub")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Multiply, "multiply", "Mul")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Divide, "divide", "Div")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Mod, "modulo", "Mod")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Pow, "power", "Pow")
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(SafeDivide, "safe divide");
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(SquaredDifference,
                                             "squared difference");

// Comparison operations
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Max, "maximum", "Max")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Min, "minimum", "Min")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Equal, "equal", "Equal")
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(NotEqual, "not equal");
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Less, "less than", "Less")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(LessEqual,
                                      "less than or equal",
                                      "LessOrEqual")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(Greater, "greater than", "Greater")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(GreaterEqual,
                                      "greater than or equal",
                                      "GreaterOrEqual")

// Logical operations
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(LogicalAnd, "logical and", "And")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(LogicalOr, "logical or", "Or")
LBANN_DECLARE_STATELESS_EWISE_ONNX_OP(LogicalXor, "logical xor", "Xor")

} // namespace lbann

#endif // LBANN_INCLUDE_LBANN_OPERATORS_MATH_BINARY_HPP_INCLUDED
