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

#ifndef LBANN_OPERATORS_MATH_CLAMP_HPP_INCLUDED
#define LBANN_OPERATORS_MATH_CLAMP_HPP_INCLUDED

#include "lbann/operators/elementwise_operator.hpp"

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
template <typename TensorDataType>
class ClampOperator :
    public Cloneable<ClampOperator<TensorDataType>,
                     ElementwiseOperator<TensorDataType>> {
#ifdef LBANN_HAS_GPU_FP16
  using CompareType = typename std::conditional<std::is_same<TensorDataType, fp16>::value, float, TensorDataType>::type;
#else
  using CompareType = TensorDataType;
#endif

  /** @name Public Types */
  ///@{

  /** @brief The local tensor type expected in this object. */
  using AbsMatrixType = El::AbstractMatrix<TensorDataType>;

  using CPUMatrixType = El::Matrix<TensorDataType, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
  using GPUMatrixType = El::Matrix<TensorDataType, El::Device::GPU>;
#endif // LBANN_HAS_GPU

  using BaseType = Cloneable<ClampOperator<TensorDataType>, ElementwiseOperator<TensorDataType>>;
  ///@}

public:
  ClampOperator(TensorDataType min, TensorDataType max)
    : m_min(min), m_max(max) {
    if (CompareType(m_min) > CompareType(m_max)) {
      std::stringstream err;
      err << "[" << m_min << "," << m_max << "] is an invalid range";
      LBANN_ERROR(err.str());
    }
  }

  std::string get_type() const override { return "clamp"; }

  description get_description() const override {
    auto desc = Operator<TensorDataType>::get_description();
    std::stringstream ss;
    ss << "[" << m_min << "," << m_max << "]";
    desc.add("Range", ss.str());
    return desc;
  }

  void write_proto(lbann_data::Operator* proto) const override {}

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  friend class cereal::access;
  ClampOperator()
    : ClampOperator(El::To<TensorDataType>(0),El::To<TensorDataType>(1))
  {}

  /** CPU-specific function instantiations */
  void fp_compute_local(std::vector<CPUMatrixType const*> inputs,
                        std::vector<CPUMatrixType*> outputs) const override;

  void bp_compute_local(std::vector<CPUMatrixType const*> inputs,
                        std::vector<CPUMatrixType const*> gradient_wrt_outputs,
                        std::vector<CPUMatrixType*> gradient_wrt_inputs) const override;

#ifdef LBANN_HAS_GPU
  /** GPU-specific function instantiations */
  void fp_compute_local(std::vector<GPUMatrixType const*> inputs,
                        std::vector<GPUMatrixType*> outputs) const override;

  void bp_compute_local(std::vector<GPUMatrixType const*> inputs,
                        std::vector<GPUMatrixType const*> gradient_wrt_outputs,
                        std::vector<GPUMatrixType*> gradient_wrt_inputs) const override;
#endif // LBANN_HAS_GPU


private:
  /** Minimum output. */
  TensorDataType m_min;
  /** Maximum output. */
  TensorDataType m_max;

};

#ifndef LBANN_CLAMP_OPERATOR_INSTANTIATE

#define PROTO(T)             \
  extern template class ClampOperator<T>

#include "lbann/macros/instantiate.hpp"
#undef PROTO

#endif // LBANN_CLAMP_OPERATOR_INSTANTIATE

} // namespace lbann

#endif // LBANN_OPERATORS_MATH_CLAMP_HPP_INCLUDED
