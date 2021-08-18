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

#ifndef LBANN_INCLUDE_LBANN_OPERATORS_MATH_UNARY_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_MATH_UNARY_HPP_INCLUDED

#include "lbann_config.hpp"

#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/utils/cloneable.hpp"

#include <operators.pb.h>

namespace lbann {

// These are all single-type operators.

#define LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(OP_NAME, OP_STRING)             \
  template <typename DataT, El::Device D>                                      \
  class OP_NAME##Operator final                                                \
    : public Cloneable<OP_NAME##Operator<DataT, D>,                            \
                       ElementwiseOperator<DataT, DataT, D>>                   \
  {                                                                            \
    using BaseType = Cloneable<OP_NAME##Operator<DataT, D>,                    \
                               ElementwiseOperator<DataT, DataT, D>>;          \
    using LocalInputTensorType = typename BaseType::LocalInputTensorType;      \
    using LocalOutputTensorType = typename BaseType::LocalOutputTensorType;    \
    using ConstLocalInputTensorType =                                          \
      typename BaseType::ConstLocalInputTensorType;                            \
    using ConstLocalOutputTensorType =                                         \
      typename BaseType::ConstLocalOutputTensorType;                           \
                                                                               \
  public:                                                                      \
    OP_NAME##Operator() = default;                                             \
    OP_NAME##Operator(OP_NAME##Operator&&) = default;                          \
    OP_NAME##Operator(OP_NAME##Operator const&) = default;                     \
    OP_NAME##Operator& operator=(OP_NAME##Operator&&) = default;               \
    OP_NAME##Operator& operator=(OP_NAME##Operator const&) = default;          \
    ~OP_NAME##Operator() = default;                                            \
    std::string get_type() const final { return OP_STRING; }                   \
    template <typename ArchiveT>                                               \
    void serialize(ArchiveT& ar)                                               \
    {                                                                          \
      using OperatorType = ElementwiseOperator<DataT, DataT, D>;               \
      ar(::cereal::make_nvp("ElementwiseOperator",                             \
                            ::cereal::base_class<OperatorType>(this)));        \
    }                                                                          \
                                                                               \
  private:                                                                     \
    void                                                                       \
    fp_compute_local(std::vector<ConstLocalInputTensorType> inputs,            \
                     std::vector<LocalOutputTensorType> outputs) const final;  \
    void bp_compute_local(                                                     \
      std::vector<ConstLocalInputTensorType> inputs,                           \
      std::vector<ConstLocalOutputTensorType> grads_wrt_outputs,               \
      std::vector<LocalInputTensorType> grads_wrt_inputs) const final;         \
    void set_proto_params(lbann_data::Operator& msg) const final               \
    {                                                                          \
      msg.mutable_parameters()->PackFrom(lbann_data:: OP_NAME##Operator{}); \
    }                                                                          \
    void do_fill_description(description&) const final {}                      \
  }

// Logical operations
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(LogicalNot, "logical not");

// Sign operations
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Negative, "negative");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Sign, "sign");

// Rounding operations
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Round, "round");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Ceil, "ceil");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Floor, "floor");

// Power operations
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Reciprocal, "reciprocal");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Square, "square");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Sqrt, "square root");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Rsqrt, "reciprocal square root");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(SafeReciprocal, "safe reciprocal");

// Exponential and logarithmic operations
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Exp, "exponential");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Expm1, "expm1");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Log, "natural logarithm");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Log1p, "log1p");

// Trigonometric operations
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Cos, "cosine");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Sin, "sine");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Tan, "tangent");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Acos, "arccosine");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Asin, "arcsine");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Atan, "arctangent");

// Hyperbolic operations
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Cosh, "hyperbolic cosine");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Sinh, "hyperbolic sine");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Tanh, "hyperbolic tangent");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Acosh, "hyperbolic arccosine");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Asinh, "hyperbolic arcsine");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Atanh, "hyperbolic arctangent");

// Error function
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(Erf, "error function");
LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR(ErfInv, "inverse error function");

} // namespace lbann

#undef LBANN_DECLARE_ENTRYWISE_UNARY_OPERATOR

#endif // LBANN_INCLUDE_LBANN_OPERATORS_MATH_UNARY_HPP_INCLUDED
