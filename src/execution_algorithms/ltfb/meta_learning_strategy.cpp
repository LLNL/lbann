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
#include "lbann/execution_algorithms/ltfb/meta_learning_strategy.hpp"
#include "lbann/execution_algorithms/ltfb/random_pairwise_exchange.hpp"
#include "lbann/execution_algorithms/ltfb/regularized_evolution.hpp"
#include "lbann/execution_algorithms/ltfb/truncation_selection_exchange.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/utils/make_abstract.hpp"

#include <google/protobuf/message.h>
#include <training_algorithm.pb.h>

namespace {

lbann::ltfb::MetaLearningStrategyFactory build_default_factory()
{
  using namespace lbann::ltfb;
  MetaLearningStrategyFactory factory;
  factory.register_builder("RandomPairwiseExchange",
                           lbann::make<RandomPairwiseExchange>);
  factory.register_builder("TruncationSelectionExchange",
                           lbann::make<TruncationSelectionExchange>);
  factory.register_builder("RegularizedEvolution",
                           lbann::make<RegularizedEvolution>);
  return factory;
}

lbann::ltfb::MetaLearningStrategyFactory& get_factory()
{
  static lbann::ltfb::MetaLearningStrategyFactory factory =
    build_default_factory();
  return factory;
}

} // namespace

void lbann::ltfb::register_new_metalearning_strategy(
  MetaLearningStrategyKey key,
  MetaLearningStrategyBuilder builder)
{
  get_factory().register_builder(std::move(key), std::move(builder));
}

void lbann::ltfb::unregister_metalearning_strategy(
  MetaLearningStrategyKey const& key)
{
  get_factory().unregister(key);
}

template <>
std::unique_ptr<lbann::ltfb::MetaLearningStrategy>
lbann::make_abstract<lbann::ltfb::MetaLearningStrategy>(
  google::protobuf::Message const& params)
{
  return get_factory().create_object(
    proto::helpers::message_type(
      dynamic_cast<google::protobuf::Any const&>(params)),
    params);
}
