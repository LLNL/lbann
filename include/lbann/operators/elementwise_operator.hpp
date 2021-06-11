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

#include "lbann/operators/data_type_operator.hpp"

namespace lbann {

/**
 * @brief Element-wise specific tensor operation sub-class.
 *
 * Specialize an operator class for element-wise operations.
 */
template <typename InputTensorDataType,
          typename OutputTensorDataType = InputTensorDataType>
class ElementwiseOperator :
    public Cloneable<HasAbstractFunction<ElementwiseOperator<InputTensorDataType, OutputTensorDataType>>,
                                         DataTypeOperator<InputTensorDataType, OutputTensorDataType>> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using InputAbsDistMatrixType = El::AbstractDistMatrix<InputTensorDataType>;
  using OutputAbsDistMatrixType = El::AbstractDistMatrix<OutputTensorDataType>;

  /** @brief The local tensor type expected in this object. */
  using InputAbsMatrixType = El::AbstractMatrix<InputTensorDataType>;
  using OutputAbsMatrixType = El::AbstractMatrix<OutputTensorDataType>;

  using InputCPUMatrixType = El::Matrix<InputTensorDataType, El::Device::CPU>;
  using OutputCPUMatrixType = El::Matrix<OutputTensorDataType, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
  using InputGPUMatrixType = El::Matrix<InputTensorDataType, El::Device::GPU>;
  using OutputGPUMatrixType = El::Matrix<OutputTensorDataType, El::Device::GPU>;
#endif // LBANN_HAS_GPU

  using BaseType =
    Cloneable<HasAbstractFunction<
                ElementwiseOperator<InputTensorDataType, OutputTensorDataType>>,
              DataTypeOperator<InputTensorDataType, OutputTensorDataType>>;
  ///@}

public:
  ElementwiseOperator() = default;
  ElementwiseOperator(const ElementwiseOperator<InputTensorDataType, OutputTensorDataType>& other) = default;
  ElementwiseOperator& operator=(const ElementwiseOperator<InputTensorDataType, OutputTensorDataType>& other) = default;
  virtual ~ElementwiseOperator() = default;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar) {};

  ///@}

protected:

  // ===========================================================
  // Distributed compute functions
  // ===========================================================
  using BaseType::fp_compute;
  using BaseType::bp_compute;

  void fp_compute(std::vector<InputAbsDistMatrixType const*>& input,
                  std::vector<OutputAbsDistMatrixType*>& output) const final;

  void bp_compute(std::vector<InputAbsDistMatrixType const*>& input,
                  std::vector<OutputAbsDistMatrixType const*>& gradient_wrt_output,
                  std::vector<InputAbsDistMatrixType*>& gradient_wrt_input) const final;


  // ===========================================================
  // Local compute functions
  // ===========================================================
  /** @brief Local forward compute function
   */
  void fp_compute_local(std::vector<InputAbsMatrixType const*>& input,
                        std::vector<OutputAbsMatrixType*>& output) const;

  /** @brief Local backward compute function
   */
  void bp_compute_local(std::vector<InputAbsMatrixType const*>& input,
                        std::vector<OutputAbsMatrixType const*>& gradient_wrt_output,
                        std::vector<InputAbsMatrixType*>& gradient_wrt_input) const;

  /** CPU-specific function instantiations */
  /** @brief Refine the forward compute for CPU-specific data types
   */
  virtual void fp_compute_local(std::vector<InputCPUMatrixType const*>& input,
                                std::vector<OutputCPUMatrixType*>& output) const;

  /** @brief Refine the backward compute for CPU-specific data types
   */
  virtual void bp_compute_local(std::vector<InputCPUMatrixType const*>& input,
                                std::vector<OutputCPUMatrixType const*>& gradient_wrt_output,
                                std::vector<InputCPUMatrixType*>& gradient_wrt_input) const;

#ifdef LBANN_HAS_GPU
  /** GPU-specific function instantiations */
  /** @brief Refine the forward compute for GPU-specific data types
   */
  virtual void fp_compute_local(std::vector<InputGPUMatrixType const*>& input,
                                std::vector<OutputGPUMatrixType*>& output) const;

  /** @brief Refine the backward compute for GPU-specific data types
   */
  virtual void bp_compute_local(std::vector<InputGPUMatrixType const*>& input,
                                std::vector<OutputGPUMatrixType const*>& gradient_wrt_output,
                                std::vector<InputGPUMatrixType*>& gradient_wrt_input) const;
#endif // LBANN_HAS_GPU

};


#ifndef LBANN_ELEMENTWISE_OPERATOR_INSTANTIATE
#define PROTO(T)                                \
  extern template class ElementwiseOperator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#endif // LBANN_ELEMENTWISE_OPERATOR_INSTANTIATE

} // namespace lbann

#endif // LBANN_OPERATORS_ELEMENTWISE_OPERATOR_HPP_INCLUDED
