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
#ifndef LBANN_PROTO_OPERATOR_FACTORY_IMPL_HPP_INCLUDED
#define LBANN_PROTO_OPERATOR_FACTORY_IMPL_HPP_INCLUDED

#include "lbann_config.hpp"

#include "lbann/proto/factories.hpp"
#include "lbann/proto/operator_factory.hpp"

#include "lbann/operators/activations/activation_builders.hpp"
#include "lbann/operators/loss/loss_builders.hpp"
#include "lbann/operators/math/math_builders.hpp"
#include "lbann/operators/operator.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/utils/protobuf.hpp"

namespace lbann {
namespace proto {
namespace details {

template <typename InT, typename OutT, El::Device D>
OperatorFactory<InT, OutT, D> build_default_factory()
{
  OperatorFactory<InT, OutT, D> factory;

#define LBANN_REGISTER_BUILDER(OP_NAME, OP_LOWER)                              \
  factory.register_builder(#OP_NAME "Operator",                                \
                           build_##OP_LOWER##_operator<InT, D>)

  if constexpr (std::is_same_v<InT, OutT>) {
    LBANN_REGISTER_BUILDER(Acos, acos);
    LBANN_REGISTER_BUILDER(Acosh, acosh);
    LBANN_REGISTER_BUILDER(Add, add);
    LBANN_REGISTER_BUILDER(AddConstant, add_constant);
    LBANN_REGISTER_BUILDER(Asin, asin);
    LBANN_REGISTER_BUILDER(Asinh, asinh);
    LBANN_REGISTER_BUILDER(Atan, atan);
    LBANN_REGISTER_BUILDER(Atanh, atanh);
    LBANN_REGISTER_BUILDER(BinaryCrossEntropy, binary_cross_entropy);
    LBANN_REGISTER_BUILDER(BooleanAccuracy, boolean_accuracy);
    LBANN_REGISTER_BUILDER(BooleanFalseNegative, boolean_false_negative);
    LBANN_REGISTER_BUILDER(BooleanFalsePositive, boolean_false_positive);
    LBANN_REGISTER_BUILDER(Ceil, ceil);
    LBANN_REGISTER_BUILDER(Clamp, clamp);
    LBANN_REGISTER_BUILDER(ConstantSubtract, constant_subtract);
    LBANN_REGISTER_BUILDER(Cos, cos);
    LBANN_REGISTER_BUILDER(Cosh, cosh);
    LBANN_REGISTER_BUILDER(Divide, divide);
    LBANN_REGISTER_BUILDER(Equal, equal);
    LBANN_REGISTER_BUILDER(EqualConstant, equal_constant);
    LBANN_REGISTER_BUILDER(Erf, erf);
    LBANN_REGISTER_BUILDER(ErfInv, erfinv);
    LBANN_REGISTER_BUILDER(Exp, exp);
    LBANN_REGISTER_BUILDER(Expm1, expm1);
    LBANN_REGISTER_BUILDER(Floor, floor);
    LBANN_REGISTER_BUILDER(Gelu, gelu);
    LBANN_REGISTER_BUILDER(GeluNew, gelunew);
    LBANN_REGISTER_BUILDER(Greater, greater);
    LBANN_REGISTER_BUILDER(GreaterConstant, greater_constant);
    LBANN_REGISTER_BUILDER(GreaterEqual, greater_equal);
    LBANN_REGISTER_BUILDER(GreaterEqualConstant, greater_equal_constant);
    LBANN_REGISTER_BUILDER(Less, less);
    LBANN_REGISTER_BUILDER(LessConstant, less_constant);
    LBANN_REGISTER_BUILDER(LessEqual, less_equal);
    LBANN_REGISTER_BUILDER(LessEqualConstant, less_equal_constant);
    LBANN_REGISTER_BUILDER(Log, log);
    LBANN_REGISTER_BUILDER(Log1p, log1p);
    LBANN_REGISTER_BUILDER(LogSigmoid, log_sigmoid);
    LBANN_REGISTER_BUILDER(LogicalAnd, logical_and);
    LBANN_REGISTER_BUILDER(LogicalNot, logical_not);
    LBANN_REGISTER_BUILDER(LogicalOr, logical_or);
    LBANN_REGISTER_BUILDER(LogicalXor, logical_xor);
    LBANN_REGISTER_BUILDER(Max, max);
    LBANN_REGISTER_BUILDER(MaxConstant, max_constant);
    LBANN_REGISTER_BUILDER(Min, min);
    LBANN_REGISTER_BUILDER(MinConstant, min_constant);
    LBANN_REGISTER_BUILDER(Mod, mod);
    LBANN_REGISTER_BUILDER(Multiply, multiply);
    LBANN_REGISTER_BUILDER(Negative, negative);
    LBANN_REGISTER_BUILDER(NotEqual, not_equal);
    LBANN_REGISTER_BUILDER(NotEqualConstant, not_equal_constant);
    LBANN_REGISTER_BUILDER(Pow, pow);
    LBANN_REGISTER_BUILDER(Reciprocal, reciprocal);
    LBANN_REGISTER_BUILDER(Round, round);
    LBANN_REGISTER_BUILDER(Rsqrt, rsqrt);
    LBANN_REGISTER_BUILDER(SafeDivide, safe_divide);
    LBANN_REGISTER_BUILDER(SafeReciprocal, safe_reciprocal);
    LBANN_REGISTER_BUILDER(Scale, scale);
    LBANN_REGISTER_BUILDER(Select, select);
    LBANN_REGISTER_BUILDER(Selu, selu);
    LBANN_REGISTER_BUILDER(Sigmoid, sigmoid);
    LBANN_REGISTER_BUILDER(SigmoidBinaryCrossEntropy,
                           sigmoid_binary_cross_entropy);
    LBANN_REGISTER_BUILDER(Sign, sign);
    LBANN_REGISTER_BUILDER(Sin, sin);
    LBANN_REGISTER_BUILDER(Sinh, sinh);
    LBANN_REGISTER_BUILDER(Softplus, softplus);
    LBANN_REGISTER_BUILDER(Softsign, softsign);
    LBANN_REGISTER_BUILDER(Sqrt, sqrt);
    LBANN_REGISTER_BUILDER(Square, square);
    LBANN_REGISTER_BUILDER(SquaredDifference, squared_difference);
    LBANN_REGISTER_BUILDER(Subtract, subtract);
    LBANN_REGISTER_BUILDER(SubtractConstant, subtract_constant);
    LBANN_REGISTER_BUILDER(Tan, tan);
    LBANN_REGISTER_BUILDER(Tanh, tanh);
  }

  if constexpr (std::is_same_v<OutT, El::Base<InT>>) {
    factory.register_builder("AbsOperator", build_abs_operator<InT, D>);
  }

  return factory;
}

} // namespace details
} // namespace proto
} // namespace lbann

template <typename InT, typename OutT, El::Device D>
auto lbann::proto::get_operator_factory() -> OperatorFactory<InT, OutT, D>&
{
  static auto factory = details::build_default_factory<InT, OutT, D>();
  return factory;
}

template <typename InputT, typename OutputT, El::Device D>
auto lbann::proto::construct_operator(lbann_data::Operator const& msg)
  -> std::unique_ptr<Operator<InputT, OutputT, D>>
{
  LBANN_ASSERT(ProtoDataType<InputT> == msg.input_datatype());
  LBANN_ASSERT(ProtoDataType<OutputT> == msg.output_datatype());
  LBANN_ASSERT(ProtoDevice<D> ==
               proto::resolve_default_device(msg.device_allocation()));

  auto const name = protobuf::message_type(msg.parameters());
  return get_operator_factory<InputT, OutputT, D>().create_object(name, msg);
}

#ifndef LBANN_INSTANTIATE_DEFAULT_OPERATOR_FACTORIES
namespace lbann {
#define PROTO_DEVICE(T, DEVICE)                                                \
  extern template proto::OperatorFactory<T, T, DEVICE>&                        \
  proto::get_operator_factory();                                               \
  extern template std::unique_ptr<Operator<T, T, DEVICE>>                      \
  proto::construct_operator(lbann_data::Operator const& msg)
#include <lbann/macros/instantiate_device.hpp>
} // namespace lbann
#endif //  LBANN_INSTANTIATE_DEFAULT_OPERATOR_FACTORIES
#endif // LBANN_PROTO_OPERATOR_FACTORY_IMPL_HPP_INCLUDED
