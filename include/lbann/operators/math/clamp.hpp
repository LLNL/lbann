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

#ifndef LBANN_OPERATORS_MATH_CLAMP_HPP_INCLUDED
#define LBANN_OPERATORS_MATH_CLAMP_HPP_INCLUDED

#include "lbann_config.hpp"

#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/utils/cloneable.hpp"

#include "lbann/proto/operators.pb.h"

#include <h2/meta/Core.hpp>

namespace lbann {

/** @brief Constrain values to a range.
 *
 *  @f[
 *    \text{clamp}(x; \text{min}, \text{max}) =
 *      \begin{cases}
 *        \text{min} & x \leq \text{min}           \\
 *        x          & \text{min} < x < \text{max} \\
 *        \text{max} & x \geq \text{max}
 *      \end{cases}
 *  @f]
 */
template <typename DataT, El::Device D>
class ClampOperator final
  : public Cloneable<ClampOperator<DataT, D>,
                     ElementwiseOperator<DataT, DataT, D>>
{
#ifdef LBANN_HAS_GPU_FP16
  using CompareType =
    h2::meta::IfThenElse<std::is_same_v<DataT, fp16>, float, DataT>;
#else
  using CompareType = DataT;
#endif

  /** @name Private Types */
  ///@{

  using BaseType =
    Cloneable<ClampOperator<DataT, D>, ElementwiseOperator<DataT, DataT, D>>;

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

  ClampOperator(double min, double max)
    : m_min{El::To<DataT>(min)}, m_max{El::To<DataT>(max)}
  {
    LBANN_ASSERT(CompareType(m_min) <= CompareType(m_max));
  }
  ClampOperator(ClampOperator&&) = default;
  ClampOperator(ClampOperator const&) = default;

  ClampOperator& operator=(ClampOperator&&) = default;
  ClampOperator& operator=(ClampOperator const&) = default;

  ~ClampOperator() = default;

  ///@}
  /** @name Queries */
  ///@{

  std::string get_type() const final { return "clamp"; }
  int get_backprop_requirements() const final
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }
  DataT get_min() const noexcept { return m_min; }
  DataT get_max() const noexcept { return m_max; }

  ///@}
  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar)
  {
    using OperatorType = ElementwiseOperator<DataT, DataT, D>;
    ar(::cereal::make_nvp("ElementwiseOperator",
                          ::cereal::base_class<OperatorType>(this)),
       CEREAL_NVP(m_min),
       CEREAL_NVP(m_max));
  }

  ///@}

protected:
  friend class cereal::access;
  ClampOperator() : ClampOperator(El::To<DataT>(0), El::To<DataT>(1)) {}

private:
  /** @brief Local forward compute function */
  void fp_compute_local(std::vector<ConstLocalInputTensorType> input,
                        std::vector<LocalOutputTensorType> output) const final;

  /** @brief Local backward compute function */
  void bp_compute_local(
    std::vector<ConstLocalInputTensorType> input,
    std::vector<ConstLocalOutputTensorType> gradient_wrt_output,
    std::vector<LocalInputTensorType> gradient_wrt_input) const final;

  void set_proto_params(lbann_data::Operator& msg) const final
  {
    lbann_data::ClampOperator clamp_msg;
    clamp_msg.set_min(m_min);
    clamp_msg.set_max(m_max);
    msg.mutable_parameters()->PackFrom(clamp_msg);
  }

  void do_fill_description(description& desc) const final
  {
    std::ostringstream oss;
    oss << "[" << m_min << "," << m_max << "]";
    desc.add("Range", oss.str());
  }

private:
  /** Minimum output. */
  DataT m_min;
  /** Maximum output. */
  DataT m_max;
}; // class ClampOperator

#ifndef LBANN_CLAMP_OPERATOR_INSTANTIATE
#define PROTO_DEVICE(T, D) extern template class ClampOperator<T, D>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_CLAMP_OPERATOR_INSTANTIATE

} // namespace lbann

#endif // LBANN_OPERATORS_MATH_CLAMP_HPP_INCLUDED
