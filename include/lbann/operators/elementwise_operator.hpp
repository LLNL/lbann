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
  void fp_compute(InputAbsDistMatrixType const& input,
                  OutputAbsDistMatrixType& output) const override;


  void bp_compute(InputAbsDistMatrixType const& input,
                  OutputAbsDistMatrixType const& gradient_wrt_output,
                  InputAbsDistMatrixType& gradient_wrt_input) const override;


  // ===========================================================
  // Local compute functions
  // ===========================================================
  /** @brief Local forward compute function
   */
  virtual void fp_compute_local(InputAbsMatrixType const& input,
                                OutputAbsMatrixType& output) const;

  /** @brief Local backward compute function
   */
  virtual void bp_compute_local(InputAbsMatrixType const& input,
                                OutputAbsMatrixType const& gradient_wrt_output,
                                InputAbsMatrixType& gradient_wrt_input) const;

  /** CPU-specific function instantiations */
  /** @brief Refine the forward compute for CPU-specific data types
   */
  virtual void fp_compute_local(InputCPUMatrixType const& input,
                                OutputCPUMatrixType& output) const;

  /** @brief Refine the backward compute for CPU-specific data types
   */
  virtual void bp_compute_local(const InputCPUMatrixType& input,
                                const OutputCPUMatrixType& gradient_wrt_output,
                                InputCPUMatrixType& gradient_wrt_input) const;

#ifdef LBANN_HAS_GPU
  /** GPU-specific function instantiations */
  /** @brief Refine the forward compute for GPU-specific data types
   */
  virtual void fp_compute_local(InputGPUMatrixType const& input,
                                OutputGPUMatrixType& output) const;

  /** @brief Refine the backward compute for GPU-specific data types
   */
  virtual void bp_compute_local(const InputGPUMatrixType& input,
                                const OutputGPUMatrixType& gradient_wrt_output,
                                InputGPUMatrixType& gradient_wrt_input) const;
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
