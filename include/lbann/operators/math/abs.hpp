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

#ifndef LBANN_OPERATORS_MATH_ABS_HPP_INCLUDED
#define LBANN_OPERATORS_MATH_ABS_HPP_INCLUDED

#include "lbann_config.hpp"

#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/utils/cloneable.hpp"

#include "lbann/proto/operators.pb.h"

#include <h2/meta/Core.hpp>

#include <google/protobuf/message.h>

namespace lbann {

/** @brief Entrywise absolute value.
 *
 *  @f[
 *    \text{abs}(x) = |x|
 *  @f]
 */
template <typename DataT, El::Device D>
class AbsOperator final
  : public Cloneable<AbsOperator<DataT, D>,
                     ElementwiseOperator<DataT, El::Base<DataT>, D>>
{
  /** @name Private Types */
  ///@{

  using BaseType = Cloneable<AbsOperator<DataT, D>,
                             ElementwiseOperator<DataT, El::Base<DataT>, D>>;

  using LocalInputTensorType = typename BaseType::LocalInputTensorType;
  using LocalOutputTensorType = typename BaseType::LocalOutputTensorType;
  using ConstLocalInputTensorType =
    typename BaseType::ConstLocalInputTensorType;
  using ConstLocalOutputTensorType =
    typename BaseType::ConstLocalOutputTensorType;

  ///@}

public:
  /** @name Lifecycle */
  ///@{

  AbsOperator() = default;
  AbsOperator(AbsOperator&&) = default;
  AbsOperator(AbsOperator const&) = default;
  AbsOperator& operator=(AbsOperator&&) = default;
  AbsOperator& operator=(AbsOperator const&) = default;
  ~AbsOperator() = default;

  ///@}
  /** @name Queries */
  ///@{

  std::string get_type() const final { return "abs"; }
  int get_backprop_requirements() const final
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }

  ///@}
  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar)
  {
    using OperatorType = ElementwiseOperator<DataT, El::Base<DataT>, D>;
    ar(::cereal::make_nvp("DataTypeOperator",
                          ::cereal::base_class<OperatorType>(this)));
  }

  ///@}

private:
  /** @brief Local forward compute function */
  virtual void
  fp_compute_local(std::vector<ConstLocalInputTensorType> input,
                   std::vector<LocalOutputTensorType> output) const final;

  /** @brief Local backward compute function */
  void bp_compute_local(
    std::vector<ConstLocalInputTensorType> input,
    std::vector<ConstLocalOutputTensorType> gradient_wrt_output,
    std::vector<LocalInputTensorType> gradient_wrt_input) const final;

  void set_proto_params(lbann_data::Operator& msg) const final
  {
    msg.mutable_parameters()->PackFrom(lbann_data::AbsOperator{});
  }

  void do_fill_description(description& desc) const final {}
}; // class AbsOperator

#ifndef LBANN_ABS_OPERATOR_INSTANTIATE
#define PROTO_DEVICE(T, D) extern template class AbsOperator<T, D>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_ABS_OPERATOR_INSTANTIATE

} // namespace lbann

#endif // LBANN_OPERATORS_MATH_ABS_HPP_INCLUDED
