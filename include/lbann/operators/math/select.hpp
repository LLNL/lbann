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

#ifndef LBANN_INCLUDE_LBANN_OPERATORS_SELLECT_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_SELLECT_HPP_INCLUDED

#include "lbann_config.hpp"

#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/utils/cloneable.hpp"

#include "lbann/proto/operators.pb.h"

#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/operators/operator.hpp"
#include "lbann/utils/cloneable.hpp"

#include "lbann/proto/operators.pb.h"

namespace lbann {

template <typename DataT, El::Device D>
class SelectOperator final
  : public Cloneable<SelectOperator<DataT, D>,
                     ElementwiseOperator<DataT, DataT, D>>
{
  using BaseType =
    Cloneable<SelectOperator<DataT, D>, ElementwiseOperator<DataT, DataT, D>>;
  using LocalInputTensorType = typename BaseType::LocalInputTensorType;
  using LocalOutputTensorType = typename BaseType::LocalOutputTensorType;
  using ConstLocalInputTensorType =
    typename BaseType::ConstLocalInputTensorType;
  using ConstLocalOutputTensorType =
    typename BaseType::ConstLocalOutputTensorType;

public:
  SelectOperator(double value = 0.,
                 bool constant_if_true = false,
                 bool constant_if_false = false,
                 double value_if_true = 0.,
                 double value_if_false = 0.,
                 double epsilon = 1e-5)
    : m_value{El::To<DataT>(value)},
      m_constant_if_true{constant_if_true},
      m_constant_if_false{constant_if_false},
      m_value_if_true{El::To<DataT>(value_if_true)},
      m_value_if_false{El::To<DataT>(value_if_false)},
      m_epsilon{El::To<DataT>(epsilon)}
  {}
  SelectOperator(SelectOperator&&) = default;
  SelectOperator(SelectOperator const&) = default;
  SelectOperator& operator=(SelectOperator&&) = default;
  SelectOperator& operator=(SelectOperator const&) = default;
  ~SelectOperator() = default;
  std::string get_type() const final { return "select"; }
  int get_backprop_requirements() const final
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }
  template <typename ArchiveT>
  void serialize(ArchiveT& ar)
  {
    using OperatorType = ElementwiseOperator<DataT, DataT, D>;
    ar(::cereal::make_nvp("ElementwiseOperator",
                          ::cereal::base_class<OperatorType>(this)),
       CEREAL_NVP(m_value),
       CEREAL_NVP(m_constant_if_true),
       CEREAL_NVP(m_constant_if_false),
       CEREAL_NVP(m_value_if_true),
       CEREAL_NVP(m_value_if_false),
       CEREAL_NVP(m_epsilon));
  }

  DataT get_value() { return m_value; }
  DataT get_epsilon() { return m_epsilon; }
  bool is_true_case_constant() { return m_constant_if_true; }
  bool is_false_case_constant() { return m_constant_if_false; }
  DataT get_constant_true_case() { return m_value_if_true; }
  DataT get_constant_false_case() { return m_value_if_false; }

private:
  void fp_compute_local(std::vector<ConstLocalInputTensorType> inputs,
                        std::vector<LocalOutputTensorType> outputs) const final;
  void bp_compute_local(
    std::vector<ConstLocalInputTensorType> inputs,
    std::vector<ConstLocalOutputTensorType> grads_wrt_outputs,
    std::vector<LocalInputTensorType> grads_wrt_inputs) const final;
  void set_proto_params(lbann_data::Operator& msg) const final
  {
    lbann_data::SelectOperator op_msg;
    op_msg.set_value(m_value);
    op_msg.set_constant_if_true(m_constant_if_true);
    op_msg.set_constant_if_false(m_constant_if_false);
    op_msg.set_value_if_true(m_value_if_true);
    op_msg.set_value_if_false(m_value_if_false);
    op_msg.set_epsilon(m_epsilon);
    msg.mutable_parameters()->PackFrom(op_msg);
  }
  void do_fill_description(description& desc) const final
  {
    {
      std::ostringstream oss;
      oss << m_value;
      desc.add("Value", oss.str());
    }
    if (m_constant_if_true) {
      std::ostringstream oss;
      oss << m_value_if_true;
      desc.add("If equal (constant)", oss.str());
    }
    if (m_constant_if_false) {
      std::ostringstream oss;
      oss << m_value_if_false;
      desc.add("If unequal (constant)", oss.str());
    }
    {
      std::ostringstream oss;
      oss << m_epsilon;
      desc.add("Equality epsilon", oss.str());
    }
  }

private:
  DataT m_value;
  bool m_constant_if_true, m_constant_if_false;
  DataT m_value_if_true, m_value_if_false, m_epsilon;
};

} // namespace lbann
#endif // LBANN_INCLUDE_LBANN_OPERATORS_SELLECT_HPP_INCLUDED
