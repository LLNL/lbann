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
#ifndef LBANN_INCLUDE_LBANN_OPERATORS_DECLARE_STATELESS_OP_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_DECLARE_STATELESS_OP_HPP_INCLUDED

#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/operators/operator.hpp"
#include "lbann/utils/cloneable.hpp"

#include "lbann/proto/operators.pb.h"

// These are all single-type operators.

#define LBANN_DECLARE_STATELESS_OPERATOR(OP_NAME, OP_STRING, NEEDS_PREVACTS)   \
  template <typename DataT, El::Device D>                                      \
  class OP_NAME##Operator final                                                \
    : public Cloneable<OP_NAME##Operator<DataT, D>, Operator<DataT, DataT, D>> \
  {                                                                            \
    using BaseType =                                                           \
      Cloneable<OP_NAME##Operator<DataT, D>, Operator<DataT, DataT, D>>;       \
    using InputTensorType = typename BaseType::InputTensorType;                \
    using OutputTensorType = typename BaseType::OutputTensorType;              \
    using ConstInputTensorType = typename BaseType::ConstInputTensorType;      \
    using ConstOutputTensorType = typename BaseType::ConstOutputTensorType;    \
                                                                               \
  public:                                                                      \
    OP_NAME##Operator() = default;                                             \
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
      using OperatorType = Operator<DataT, DataT, D>;                          \
      ar(::cereal::make_nvp("Operator",                                        \
                            ::cereal::base_class<OperatorType>(this)));        \
    }                                                                          \
    void fp_compute(std::vector<ConstInputTensorType> const& inputs,           \
                    std::vector<OutputTensorType> const& outputs) const final; \
    void bp_compute(                                                           \
      std::vector<ConstInputTensorType> const& inputs,                         \
      std::vector<ConstOutputTensorType> const& gradient_wrt_outputs,          \
      std::vector<InputTensorType> const& gradient_wrt_inputs) const final;    \
                                                                               \
  private:                                                                     \
    void set_proto_params(lbann_data::Operator& msg) const final               \
    {                                                                          \
      msg.mutable_parameters()->PackFrom(lbann_data::OP_NAME##Operator{});     \
    }                                                                          \
    void do_fill_description(description&) const final {}                      \
  }

#define LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(OP_NAME,                  \
                                                     OP_STRING,                \
                                                     NEEDS_PREVACTS)           \
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
      msg.mutable_parameters()->PackFrom(lbann_data::OP_NAME##Operator{});     \
    }                                                                          \
    void do_fill_description(description&) const final {}                      \
  }

#endif // LBANN_INCLUDE_LBANN_OPERATORS_DECLARE_STATELESS_OP_HPP_INCLUDED
