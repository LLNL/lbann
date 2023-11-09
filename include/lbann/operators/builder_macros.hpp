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
#ifndef LBANN_INCLUDE_LBANN_OPERATORS_BUILDER_MACROS_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_BUILDER_MACROS_HPP_INCLUDED

#include "lbann/operators/operator.hpp"
#include "lbann/proto/datatype_helpers.hpp"

#include "lbann/proto/operators.pb.h"

#include <memory>

// Forward declaration
namespace lbann_data {
class Operator;
} // namespace lbann_data

namespace lbann {
namespace details {

template <typename InputT, typename OutputT, El::Device D>
void AssertConsistentTypeParameters(lbann_data::Operator const& op)
{
  LBANN_ASSERT(proto::ProtoDataType<InputT> ==
               proto::resolve_default_datatype(op.input_datatype()));
  LBANN_ASSERT(proto::ProtoDataType<OutputT> ==
               proto::resolve_default_datatype(op.output_datatype()));
  LBANN_ASSERT(proto::ProtoDevice<D> ==
               proto::resolve_default_device(op.device_allocation()));
}

} // namespace details
} // namespace lbann

/** @brief A utility macro for adding a builder declaration for a single-type
 *         operator.
 *  @note Must be called inside lbann namespace.
 */
#define LBANN_DECLARE_OPERATOR_BUILDER(OP_NAME)                                \
  template <typename DataT, El::Device D>                                      \
  std::unique_ptr<Operator<DataT, DataT, D>> build_##OP_NAME##_operator(       \
    lbann_data::Operator const& op)

/** @brief A utility macro for easily adding a default builder with
 *         dynamic type-checking assertions.
 *
 *  Type-checking is only done with Debug builds.
 *
 *  @note Must *NOT* be called inside lbann namespace.
 */
#define LBANN_DEFINE_OPERATOR_BUILDER(OP_LOWER, OP_NAME)                       \
  template <typename DataT, El::Device D>                                      \
  std::unique_ptr<lbann::Operator<DataT, DataT, D>>                            \
    lbann::build_##OP_LOWER##_operator(lbann_data::Operator const& op)         \
  {                                                                            \
    details::AssertConsistentTypeParameters<DataT, DataT, D>(op);              \
    return std::make_unique<OP_NAME##Operator<DataT, D>>();                    \
  }

/** @brief A utility macro for easily adding ETI for operator builders
 *  @note Must *NOT* be called inside lbann namespace.
 */
#define LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(OPERATOR_NAME, T, D)            \
  template std::unique_ptr<lbann::Operator<T, T, D>>                           \
    lbann::build_##OPERATOR_NAME##_operator<T, D>(lbann_data::Operator const&)

#endif // LBANN_INCLUDE_LBANN_OPERATORS_BUILDER_MACROS_HPP_INCLUDED
