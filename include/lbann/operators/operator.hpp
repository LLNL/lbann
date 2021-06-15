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

/** @brief A utility macro for easily defining default-constructed sub-class
 *  builders.*/
#define LBANN_DEFINE_OPERATOR_BUILDER(OPERATOR_NAME)                          \
  template <typename TensorDataType> \
  std::unique_ptr<Operator> build_##OPERATOR_NAME##_operator_from_pbuf( \
    lbann_data::Operator const&)

/** @brief A utility macro for easily defining "default" builders.
 *  @note Must be called inside lbann namespace.
 */
#define LBANN_OPERATOR_DEFAULT_BUILDER(OPERATOR_NAME) \
  template <typename TensorDataType> \
  std::unique_ptr<Operator> build_##OPERATOR_NAME##_operator_from_pbuf(          \
    lbann_data::Operator const&)                         \
  {                                                                     \
    using OperatorType = OPERATOR_NAME##_operator<TensorDataType>; \
    return make_unique<OperatorType>();                                \
  }

/** @brief A utility macro for easily adding ETI for operator builders
 *  @note Must be called inside lbann namespace.
 */
#define LBANN_OPERATOR_BUILDER_ETI(OPERATOR_NAME, T)                  \
  template std::unique_ptr<Operator>                                       \
  build_##OPERATOR_NAME##_operator_from_pbuf<T>( \
    lbann_data::Operator const&)

// Forward-declare protobuf classes
namespace lbann_data {
class Operator;
}

namespace lbann {

// Forward declarations

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
 */
class Operator: public Cloneable<HasAbstractFunction<Operator>> {
public:

  Operator();
  Operator(const Operator& other) = default;
  Operator& operator=(const Operator& other) = default;
  virtual ~Operator() = default;

  /** @brief Get the operator type's name.
   *  A operator type name should be brief, human-readable description of
   *  the operator's mathematical operation.
   */
  virtual std::string get_type() const = 0;
  /** @brief Get the operator instance's name.
   *  Each operator in a model should have a unique, preferably
   *  human-readable, name.
   */
  inline std::string get_name() const { return m_name; }
  /** @brief Set the operator instance's name.
   *  Each operator in a model should have a unique, preferably
   *  human-readable, name.
   */
  inline void set_name(const std::string name) { m_name = name; }
  /** @brief Get a string representing the operator datatype */
  virtual std::string get_datatype_name() const {
    return TypeName<DataType>();
  };

  /** @brief Human-readable description. */
  virtual description get_description() const;

  /** @brief Write operator to proto file */
  virtual void write_proto(lbann_data::Operator* proto) const = 0;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  // ===========================================================
  // Forward prop compute function
  // ===========================================================

  /** @brief Apply operator's forward operation.
   *  Given the input tensors, the output tensors are populated with
   *  computed values.
   */
  virtual void fp_compute(std::vector<BaseDistMat const*>& inputs,
                          std::vector<BaseDistMat*>& outputs) const = 0;

  // ===========================================================
  // Back prop compute function
  // ===========================================================

  /** @brief Compute operator's "backward" operation
   *  Given the inputs, outputs, and gradient w.r.t. output tensors,
   *  the gradient w.r.t. input tensors are populated with the
   *  computed values.
   */
  virtual void bp_compute(std::vector<BaseDistMat const*>& inputs,
                          std::vector<BaseDistMat const*>& gradient_wrt_outputs,
                          std::vector<BaseDistMat*>& gradient_wrt_inputs) const {};

protected:

  // ===========================================================
  // Protected class members
  // ===========================================================

  /** @brief Operator instance's name.
   *  Each operator in a model should have a unique, preferably
   *  human-readable, name.
   */
  std::string m_name;
};

} // namespace lbann

#endif // LBANN_OPERATORS_OPERATOR_HPP_INCLUDED
