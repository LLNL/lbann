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

#ifndef LBANN_OPERATORS_ELEMENTWISE_OPERATOR_HPP_INCLUDED
#define LBANN_OPERATORS_ELEMENTWISE_OPERATOR_HPP_INCLUDED

#include "lbann/operators/operator.hpp"

#include <cereal/cereal.hpp>

#include <functional>
#include <iterator>
#include <type_traits>

namespace lbann {

/** @brief Element-wise specific tensor operation sub-class.
 *
 *  This layer manages some of the
 */
template <typename InputT, typename OutputT, El::Device D>
class ElementwiseOperator
  : public AbstractCloneableBase<ElementwiseOperator<InputT, OutputT, D>,
                                 Operator<InputT, OutputT, D>>
{
public:
  /** @name Public Types */
  ///@{

  using BaseType =
    Cloneable<HasAbstractFunction<ElementwiseOperator<InputT, OutputT, D>>,
              Operator<InputT, OutputT, D>>;

  using InputTensorType = typename BaseType::InputTensorType;
  using OutputTensorType = typename BaseType::OutputTensorType;
  using ConstInputTensorType = typename BaseType::ConstInputTensorType;
  using ConstOutputTensorType = typename BaseType::ConstOutputTensorType;

  using LocalInputTensorType = utils::TensorView<InputT, D>;
  using LocalOutputTensorType = utils::TensorView<OutputT, D>;
  using ConstLocalInputTensorType = utils::ConstTensorView<InputT, D>;
  using ConstLocalOutputTensorType = utils::ConstTensorView<OutputT, D>;

  ///@}

public:
  ElementwiseOperator() = default;
  virtual ~ElementwiseOperator() = default;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar)
  {
    ar(cereal::base_class<Operator<InputT, OutputT, D>>(this));
  };

  ///@}

  /** @name Virtual compute interface */
  ///@{
  using BaseType::bp_compute;
  using BaseType::fp_compute;

  template <typename TensorViewType>
  static auto get_local_tensor_views(std::vector<TensorViewType> const& in)
  {
    using LocalViewType = std::decay_t<decltype(in[0].local_data())>;

    std::vector<LocalViewType> local_views;
    local_views.reserve(in.size());
    std::transform(cbegin(in),
                   cend(in),
                   std::back_inserter(local_views),
                   [](auto const& x) { return x.local_data(); });
    return local_views;
  }

  /** @brief Apply operator's forward operation.
   *  @details Given the input tensors, the output tensors are
   *           populated with computed values.
   */
  void fp_compute(std::vector<ConstInputTensorType> const& inputs,
                  std::vector<OutputTensorType> const& outputs) const final
  {
    return fp_compute_local(get_local_tensor_views(inputs),
                            get_local_tensor_views(outputs));
  }

  // ===========================================================
  // Back prop compute function
  // ===========================================================

  /** @brief Compute operator's "backward" operation
   *  @details Given the inputs, outputs, and gradient w.r.t. output
   *           tensors, the gradient w.r.t. input tensors are
   *           populated with the computed values.
   */
  void bp_compute(
    std::vector<ConstInputTensorType> const& inputs,
    std::vector<ConstOutputTensorType> const& gradient_wrt_outputs,
    std::vector<InputTensorType> const& gradient_wrt_inputs) const final
  {
    return bp_compute_local(get_local_tensor_views(inputs),
                            get_local_tensor_views(gradient_wrt_outputs),
                            get_local_tensor_views(gradient_wrt_inputs));
  }

  ///@}

protected:
  /** @name Lifecycle management. */
  ///@{
  ElementwiseOperator(ElementwiseOperator const&) = default;
  ElementwiseOperator& operator=(ElementwiseOperator const&) = default;
  ElementwiseOperator(ElementwiseOperator&&) = default;
  ElementwiseOperator& operator=(ElementwiseOperator&&) = default;
  ///@}

  /** @name Local compute interface */
  ///@{

  /** @brief Local forward compute function */
  virtual void
  fp_compute_local(std::vector<ConstLocalInputTensorType> input,
                   std::vector<LocalOutputTensorType> output) const = 0;

  /** @brief Local backward compute function */
  virtual void bp_compute_local(
    std::vector<ConstLocalInputTensorType> input,
    std::vector<ConstLocalOutputTensorType> gradient_wrt_output,
    std::vector<LocalInputTensorType> gradient_wrt_input) const = 0;

  ///@}

}; // class ElementwiseOperator

} // namespace lbann
#endif // LBANN_OPERATORS_ELEMENTWISE_OPERATOR_HPP_INCLUDED
