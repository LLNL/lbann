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

#ifndef LBANN_OPERATORS_OPERATOR_HPP_INCLUDED
#define LBANN_OPERATORS_OPERATOR_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/describable.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/protobuf_serializable.hpp"
#include "lbann/utils/tensor.hpp"
#include "lbann/utils/typename.hpp"

#include "lbann/proto/operators.pb.h"

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>

#include <google/protobuf/message.h>

#include <string>
#include <vector>

namespace lbann {

using supported_operator_data_type = h2::meta::TL<
#ifdef LBANN_HAS_GPU_FP16
  fp16,
#endif
#ifdef LBANN_HAS_HALF
  cpu_fp16,
#endif
  float,
  double,
  El::Complex<float>,
  El::Complex<double>>;

/** @brief Neural network tensor operation.
 *
 *  An operator defines a mathematical function that that supports
 *  both forward and possibly backward operations. In the forward
 *  direction, it takes a vector of input tensors and produces a
 *  vector of output tensors.  In the backward direction they
 *  implement the differentiation of the forward operation, applying
 *  the function to the operator's forward inputs and gradient with
 *  respect to the outputs, to compute the gradient with respect to
 *  the input.
 *
 *  Operators act as curried functions, they can have state that is
 *  defined during construction but do not hold internal state.  A
 *  operator should also be able to take objective function gradients
 *  w.r.t. the outputs ("previous error signals") and compute the
 *  objective function gradients w.r.t. the inputs ("error
 *  signals"). This allows the model to perform automatic
 *  differentiation.
 *
 *  Operators are specified for unique input and output data types.
 */
template <typename InputT, typename OutputT, El::Device D>
class Operator : public AbstractCloneableBase<Operator<InputT, OutputT, D>>,
                 public Describable,
                 public ProtobufSerializable
{
public:
  /** @name Public Types */
  ///@{
  using InputTensorType = utils::DistTensorView<InputT, D>;
  using OutputTensorType = utils::DistTensorView<OutputT, D>;
  using ConstInputTensorType = utils::ConstDistTensorView<InputT, D>;
  using ConstOutputTensorType = utils::ConstDistTensorView<OutputT, D>;
  ///@}

public:
  static_assert(
    h2::meta::tlist::MemberV<InputT, supported_operator_data_type>(),
    "Must use a supported input type.");
  static_assert(
    h2::meta::tlist::MemberV<OutputT, supported_operator_data_type>(),
    "Must use a supported output type.");

  /** @brief Constructor */
  Operator() = default;
  /** @brief Destructor */
  virtual ~Operator() = default;

  /** @brief Get the operator type's name.
   *  @details A operator type name should be brief, human-readable
   *           description of the operator's mathematical operation.
   */
  virtual std::string get_type() const = 0;

  /**
   * @brief Returns the necessary tensors for computing backpropagation
   * for this operator.
   */
  virtual int get_backprop_requirements() const
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }

  /** @brief Get the description of the operator. */
  Description get_description() const override;

  /** @brief Write a protobuf description of the operator.
   *
   *  This should be able to be passed back to LBANN to exactly recreate this
   *  operator from protobuf.
   */
  void write_proto(google::protobuf::Message& msg) const;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}
  /** @name Computational interface */
  ///@{

  /** @brief Apply operator's forward operation.
   *  @details Given the input tensors, the output tensors are
   *           populated with computed values.
   */
  virtual void
  fp_compute(std::vector<ConstInputTensorType> const& inputs,
             std::vector<OutputTensorType> const& outputs) const = 0;

  /** @brief Compute operator's "backward" operation
   *  @details Given the inputs and gradient w.r.t. output
   *           tensors, the gradient w.r.t. input tensors are
   *           populated with the computed values.
   */
  virtual void
  bp_compute(std::vector<ConstInputTensorType> const& inputs,
             std::vector<ConstOutputTensorType> const& gradient_wrt_outputs,
             std::vector<InputTensorType> const& gradient_wrt_inputs) const;
  ///@}

protected:
  Operator(Operator&& other) noexcept = default;
  Operator& operator=(Operator&& other) noexcept = default;
  Operator(Operator const& other) = default;
  Operator& operator=(Operator const& other) = default;

private:
  /** @brief Fill the concrete operator parameters. */
  virtual void set_proto_params(lbann_data::Operator&) const = 0;
  /** @brief Concrete operator description. */
  virtual void do_fill_description(Description&) const = 0;
};

template <typename InputT, typename OutputT, El::Device D>
void Operator<InputT, OutputT, D>::write_proto(
  google::protobuf::Message& msg) const
{
  lbann_data::Operator operator_msg;
  operator_msg.set_input_datatype(proto::ProtoDataType<InputT>);
  operator_msg.set_output_datatype(proto::ProtoDataType<OutputT>);
  operator_msg.set_device_allocation(proto::ProtoDevice<D>);

  this->set_proto_params(operator_msg);

  msg.CopyFrom(operator_msg);
}

template <typename InputT, typename OutputT, El::Device D>
Description Operator<InputT, OutputT, D>::get_description() const
{

  // Construct description object
  Description desc(this->get_type());

  // DataType
  desc.add("Input data type", TypeName<InputT>());
  desc.add("Output data type", TypeName<OutputT>());

  this->do_fill_description(desc);

  return desc;
}

template <typename InputT, typename OutputT, El::Device D>
void Operator<InputT, OutputT, D>::bp_compute(
  std::vector<ConstInputTensorType> const&,
  std::vector<ConstOutputTensorType> const&,
  std::vector<InputTensorType> const&) const
{}

template <typename InputT, typename OutputT, El::Device D>
template <typename ArchiveT>
void Operator<InputT, OutputT, D>::serialize(ArchiveT& ar)
{}

} // namespace lbann
#endif // LBANN_OPERATORS_OPERATOR_HPP_INCLUDED
