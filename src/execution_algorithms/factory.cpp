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
#include "lbann/execution_algorithms/factory.hpp"
#include "lbann/execution_algorithms/kfac.hpp"
#include "lbann/execution_algorithms/ltfb.hpp"
#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"
#include "lbann/utils/make_abstract.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/training_algorithm.pb.h"
#include <google/protobuf/message.h>
#include <memory>

namespace {

lbann::TrainingAlgorithmFactory build_default_factory()
{
  lbann::TrainingAlgorithmFactory fact;
  fact.register_builder("SGD", lbann::make<lbann::SGDTrainingAlgorithm>);
  fact.register_builder("LTFB", lbann::make<lbann::LTFB>);
  fact.register_builder("KFAC", lbann::make<lbann::KFAC>);
  return fact;
}

lbann::TrainingAlgorithmFactory& get_factory()
{
  static lbann::TrainingAlgorithmFactory fact = build_default_factory();
  return fact;
}

} // namespace

void lbann::register_new_training_algorithm(TrainingAlgorithmKey key,
                                            TrainingAlgorithmBuilder builder)
{
  get_factory().register_builder(std::move(key), std::move(builder));
}

template <>
std::unique_ptr<lbann::TrainingAlgorithm>
lbann::make_abstract<lbann::TrainingAlgorithm>(
  google::protobuf::Message const& params)
{
  auto const& algo_params =
    dynamic_cast<lbann_data::TrainingAlgorithm const&>(params);
  return get_factory().create_object(
    protobuf::message_type(algo_params.parameters()),
    params);
}
