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

#ifndef LBANN_OPERATORS_DATA_TYPE_OPERATOR_HPP_INCLUDED
#define LBANN_OPERATORS_DATA_TYPE_OPERATOR_HPP_INCLUDED

#include "lbann/operators/operator.hpp"

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>

namespace cereal
{
  class access;
}// namespace cereal

namespace lbann {

using supported_operator_data_type = h2::meta::TL<
#ifdef LBANN_HAS_GPU_FP16
  fp16,
#endif
#ifdef LBANN_HAS_HALF
  cpu_fp16,
#endif
  float, double>;

/**
 * @brief Data type specific tensor operation sub-class.
 *
 * Specialize an operator class for unique input and output data types.
 */
template <typename InputTensorDataType,
          typename OutputTensorDataType = InputTensorDataType>
class DataTypeOperator :
    public Cloneable<HasAbstractFunction<DataTypeOperator<InputTensorDataType, OutputTensorDataType>>,
                                         Operator> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using InputAbsDistMatrixType = El::AbstractDistMatrix<InputTensorDataType>;
  using OutputAbsDistMatrixType = El::AbstractDistMatrix<OutputTensorDataType>;

  /** @brief The local tensor type expected in this object. */
  using InputAbsMatrixType = El::AbstractMatrix<InputTensorDataType>;
  using OutputAbsMatrixType = El::AbstractMatrix<OutputTensorDataType>;

  ///@}

public:
  static_assert(
    h2::meta::tlist::MemberV<InputTensorDataType, supported_operator_data_type>(),
    "Must use a supported input type.");
  static_assert(
    h2::meta::tlist::MemberV<OutputTensorDataType, supported_operator_data_type>(),
    "Must use a supported output type.");

  DataTypeOperator() = default;
  DataTypeOperator(const DataTypeOperator<InputTensorDataType, OutputTensorDataType>& other) = default;
  DataTypeOperator& operator=(const DataTypeOperator<InputTensorDataType, OutputTensorDataType>& other) = default;
  virtual ~DataTypeOperator() = default;

  /** Get a string representing the operator datatype
   */
  std::string get_datatype_name() const override {
    return TypeName<OutputTensorDataType>();
  };

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar) {};

  ///@}

protected:

  // ===========================================================
  // Forward prop compute function
  // ===========================================================

  void fp_compute(BaseDistMat const& input, BaseDistMat& output) const override;

  /** @brief Refine the forward compute for specific data types
   */
  virtual void fp_compute(InputAbsDistMatrixType const& input,
                          OutputAbsDistMatrixType& output) const = 0;

  // ===========================================================
  // Back prop compute function
  // ===========================================================

  void bp_compute(BaseDistMat const& input,
                  BaseDistMat const& gradient_wrt_output,
                  BaseDistMat& gradient_wrt_input) const override;

  /** @brief Refine the backward compute for specific data types
   */
  virtual void bp_compute(InputAbsDistMatrixType const& input,
                          OutputAbsDistMatrixType const& gradient_wrt_output,
                          InputAbsDistMatrixType& gradient_wrt_input) const {};

};


#ifndef LBANN_DATA_TYPE_OPERATOR_INSTANTIATE
#define PROTO(T)                                \
  extern template class DataTypeOperator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#endif // LBANN_DATA_TYPE_OPERATOR_INSTANTIATE

} // namespace lbann

#endif // LBANN_OPERATORS_DATA_TYPE_OPERATOR_HPP_INCLUDED
