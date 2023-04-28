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

#ifndef LBANN_INCLUDE_LBANN_OPERATORS_BINARY_WITH_CONSTANT_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_BINARY_WITH_CONSTANT_HPP_INCLUDED

#include "lbann_config.hpp"

#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/utils/cloneable.hpp"

#include "lbann/proto/operators.pb.h"

/** @file
 *
 *  These operators are idiomatic replacements for patterns like:
 *
 *    Op(layer, ConstantLayer(1))
 *
 *  where it's a pessimization to actually allocate a persistent array
 *  for such an ephemeral operation.
 */

#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/operators/operator.hpp"
#include "lbann/utils/cloneable.hpp"

#include "lbann/proto/operators.pb.h"

// These are all single-type operators.

#define LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(OP_NAME,                   \
                                                    OP_STRING,                 \
                                                    NEEDS_PREVACTS)            \
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
    OP_NAME##Operator(double constant = 0.)                                    \
      : m_constant{El::To<DataT>(constant)}                                    \
    {}                                                                         \
    OP_NAME##Operator(OP_NAME##Operator&&) = default;                          \
    OP_NAME##Operator(OP_NAME##Operator const&) = default;                     \
    OP_NAME##Operator& operator=(OP_NAME##Operator&&) = default;               \
    OP_NAME##Operator& operator=(OP_NAME##Operator const&) = default;          \
    ~OP_NAME##Operator() = default;                                            \
    std::string get_type() const final { return OP_STRING; }                   \
    int get_backprop_requirements() const final                                \
    {                                                                          \
      return ((NEEDS_PREVACTS) ? (ERROR_SIGNALS | PREV_ACTIVATIONS)            \
                               : ERROR_SIGNALS);                               \
    }                                                                          \
    template <typename ArchiveT>                                               \
    void serialize(ArchiveT& ar)                                               \
    {                                                                          \
      using OperatorType = ElementwiseOperator<DataT, DataT, D>;               \
      ar(::cereal::make_nvp("ElementwiseOperator",                             \
                            ::cereal::base_class<OperatorType>(this)),         \
         CEREAL_NVP(m_constant));                                              \
    }                                                                          \
    DataT get_constant() const noexcept { return m_constant; }                 \
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
      lbann_data::OP_NAME##Operator op_msg;                                    \
      op_msg.set_constant(m_constant);                                         \
      msg.mutable_parameters()->PackFrom(op_msg);                              \
    }                                                                          \
    void do_fill_description(description& desc) const final                    \
    {                                                                          \
      std::ostringstream oss;                                                  \
      oss << m_constant;                                                       \
      desc.add("Constant", oss.str());                                         \
    }                                                                          \
                                                                               \
  private:                                                                     \
    DataT m_constant;                                                          \
  }

namespace lbann {

// x + c -- treated as commutative.
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(AddConstant, "add constant", false);

// x + c -- treated as commutative.
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(Scale, "scale", false);

// x - C -- yes, could be "plus -C", but so could 7-4 be 7+-4, but
// nobody writes that.
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(SubtractConstant,
                                            "subtract constant",
                                            false);
// C - x -- yes, could be "negative-x plus C", but again, why write
// -4+7 when you could just write 7-4...
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(ConstantSubtract,
                                            "subtract from constant",
                                            false);

LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(MaxConstant, "max constant", true);
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(MinConstant, "min constant", true);

LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(EqualConstant,
                                            "equals constant",
                                            false);
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(NotEqualConstant,
                                            "not equals constant",
                                            false);
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(LessEqualConstant,
                                            "less-equals constant",
                                            false);
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(LessConstant,
                                            "less than constant",
                                            false);
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(GreaterEqualConstant,
                                            "greater-equals constant",
                                            false);
LBANN_DECLARE_BINARY_WITH_CONSTANT_OPERATOR(GreaterConstant,
                                            "greater than constant",
                                            false);

} // namespace lbann
#endif // LBANN_INCLUDE_LBANN_OPERATORS_BINARY_WITH_CONSTANT_HPP_INCLUDED
