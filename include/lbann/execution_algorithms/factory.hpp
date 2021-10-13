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
#ifndef LBANN_EXECUTION_ALGORITHMS_FACTORY_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_FACTORY_HPP_INCLUDED

#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/factory_error_policies.hpp"
#include "lbann/utils/make_abstract.hpp"

#include <h2/meta/typelist/TypeList.hpp>

#include <google/protobuf/message.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace lbann {

/** @brief Factory for constructing training algorithms from protobuf
 *         messages.
 */
using TrainingAlgorithmFactory = generic_factory<
  TrainingAlgorithm,
  std::string,
  proto::generate_builder_type<TrainingAlgorithm,
                               google::protobuf::Message const&>>;

/** @brief The builder type used to create a new training algorithm.
 */
using TrainingAlgorithmBuilder =
  typename TrainingAlgorithmFactory::builder_type;

/** @brief The trainining algorithm factory key. */
using TrainingAlgorithmKey = typename TrainingAlgorithmFactory::id_type;

/** @brief Register a new training algorithm with the default factory.
 *  @param[in] key The identifier for the training algorithm.
 *  @param[in] builder The builder for the training algorithm.
 */
void register_new_training_algorithm(TrainingAlgorithmKey key,
                                     TrainingAlgorithmBuilder builder);

} // namespace lbann

/** @brief Create a new training_algorithm instance.
 *  @param[in] params A protobuf message describing the algorithm.
 *  @return A newly-constructed training algorithm.
 */
template <>
std::unique_ptr<lbann::TrainingAlgorithm>
lbann::make_abstract<lbann::TrainingAlgorithm>(
  google::protobuf::Message const& params);

#endif // LBANN_EXECUTION_ALGORITHMS_FACTORY_HPP_INCLUDED
