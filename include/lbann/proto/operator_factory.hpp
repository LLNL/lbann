////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_PROTO_OPERATOR_FACTORY_HPP_INCLUDED
#define LBANN_PROTO_OPERATOR_FACTORY_HPP_INCLUDED

#include "lbann/operators/operator.hpp"
#include "lbann/utils/factory.hpp"

#include <string>

namespace lbann {
namespace proto {

// Define the factory type.
template <typename InT, typename OutT, El::Device D>
using OperatorFactory =
  generic_factory<Operator<InT, OutT, D>,
                  std::string,
                  generate_builder_type<Operator<InT, OutT, D>,
                                        lbann_data::Operator const&>>;

/** @brief Access the global operator factory for these types.
 *
 *  LBANN will include instantiations for the case that
 *  `std::is_same_v<InT, OutT>` and the data type is `float`,
 *  `double`, or the FP16 type that matches the selected device.
 */
template <typename InT, typename OutT, El::Device D>
OperatorFactory<InT, OutT, D>& get_operator_factory();

} // namespace proto
} // namespace lbann
#endif // LBANN_PROTO_OPERATOR_FACTORY_HPP_INCLUDED
