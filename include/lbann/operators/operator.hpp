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

#ifndef LBANN_OPERATORS_OPERATOR_HPP_INCLUDED
#define LBANN_OPERATORS_OPERATOR_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/typename.hpp"
#include <string>
#include <vector>

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>

/** @brief A utility macro for easily defining default-constructed sub-class
 *  builders.*/
#define LBANN_DEFINE_OPERATOR_BUILDER(OPERATOR_NAME)                          \
  template <typename TensorDataType> \
  std::unique_ptr<Operator<TensorDataType>> build_##OPERATOR_NAME##_operator_from_pbuf( \
    lbann_data::Operator const&)

/** @brief A utility macro for easily defining "default" builders.
 *  @note Must be called inside lbann namespace.
 */
#define LBANN_OPERATOR_DEFAULT_BUILDER(OPERATOR_NAME) \
  template <typename TensorDataType> \
  std::unique_ptr<Operator<TensorDataType>> build_##OPERATOR_NAME##_operator_from_pbuf( \
    lbann_data::Operator const&)                         \
  {                                                                     \
    using OperatorType = OPERATOR_NAME##_operator<TensorDataType>; \
    return make_unique<OperatorType>();                                \
  }

/** @brief A utility macro for easily adding ETI for operator builders
 *  @note Must be called inside lbann namespace.
 */
#define LBANN_OPERATOR_BUILDER_ETI(OPERATOR_NAME, T)                  \
  template std::unique_ptr<Operator<T>>                               \
  build_##OPERATOR_NAME##_operator_from_pbuf<T>( \
    lbann_data::Operator const&)

// Forward-declare protobuf classes
namespace lbann_data {
class Operator;
}


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
 * @brief Neural network tensor operation.
 *
 * An operator defines a mathematical function that that supports both
 * forward and possibly backward operations. In the forward direction,
 * it takes a vector of input tensors and produces a vector of output
 * tensors.  In the backward direction they implement the differentiation
 * of the forward operation, applying the function to the operator's
 * forward inputs and gradient with respect to the outputs, to compute
 * the gradient with respect to the input.
 * Operators act as curried functions, they can have state that
 * is defined during construction but do not hold internal state.
 * A operator should also be able to take objective function gradients
 * w.r.t. the outputs ("previous error signals") and compute the objective
 * function gradients w.r.t. the inputs ("error signals"). This allows
 * the model to perform automatic differentiation.
 *
 * Operator's are specified for unique input and output data types.
 */
template <typename InputTensorDataType,
          typename OutputTensorDataType = InputTensorDataType>
class Operator :
    public Cloneable<HasAbstractFunction<Operator<InputTensorDataType, OutputTensorDataType>>> {
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

  Operator() = default;
  Operator(const Operator<InputTensorDataType, OutputTensorDataType>& other) = default;
  Operator& operator=(const Operator<InputTensorDataType, OutputTensorDataType>& other) = default;
  virtual ~Operator() = default;

  /** @brief Get the operator type's name.
   *  A operator type name should be brief, human-readable description of
   *  the operator's mathematical operation.
   */
  virtual std::string get_type() const = 0;

  /** @brief Get a string representing the operator datatype */
  virtual std::string get_datatype_name() const {
    return TypeName<OutputTensorDataType>();
  };

  /** @brief Human-readable description. */
  virtual description get_description() const;

  /** @brief Write operator to proto file */
  virtual void write_proto(lbann_data::Operator* proto) const = 0;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar) {};

  ///@}

  // ===========================================================
  // Forward prop compute function
  // ===========================================================

  /** @brief Apply operator's forward operation.
   *  Given the input tensors, the output tensors are populated with
   *  computed values.
   */
  virtual void fp_compute(std::vector<InputAbsDistMatrixType const*> inputs,
                          std::vector<OutputAbsDistMatrixType*> outputs) const = 0;

  // ===========================================================
  // Back prop compute function
  // ===========================================================

  /** @brief Compute operator's "backward" operation
   *  Given the inputs, outputs, and gradient w.r.t. output tensors,
   *  the gradient w.r.t. input tensors are populated with the
   *  computed values.
   */
  virtual void bp_compute(std::vector<InputAbsDistMatrixType const*> inputs,
                          std::vector<OutputAbsDistMatrixType const*> gradient_wrt_outputs,
                          std::vector<InputAbsDistMatrixType*> gradient_wrt_inputs) const {};

};


#ifndef LBANN_OPERATOR_INSTANTIATE
#define PROTO(T)                                \
  extern template class Operator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#endif // LBANN_OPERATOR_INSTANTIATE

} // namespace lbann

#endif // LBANN_OPERATORS_OPERATOR_HPP_INCLUDED
