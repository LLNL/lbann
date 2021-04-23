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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_META_LEARNING_STRATEGY_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_META_LEARNING_STRATEGY_HPP_INCLUDED

#include "execution_context.hpp"

#include "lbann/proto/helpers.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/make_abstract.hpp"

#include <google/protobuf/message.h>

namespace lbann {
// forward declarations
class data_coordinator;
class model;

namespace ltfb {

/** @class MetaLearningStrategy
 *  @brief Base class describing a family of meta-learning methods
 *
 *  The current use-case for this is the LTFB family of algorithms, in
 *  which this strategy is used in tandem with a "trainer-local"
 *  training algorithm. Specifically, in the case of LTFB, the
 *  meta-learning strategy is applied *after* the local training
 *  algorithm to combine and postprocess the output of multiple
 *  trainers.
 */
class MetaLearningStrategy
  : public Cloneable<HasAbstractFunction<MetaLearningStrategy>>
{
public:
  virtual ~MetaLearningStrategy() noexcept = default;
  /** @brief Pick the next model according to this metaheuristic.
   *
   *  The fundamental assumption here is that each trainer contributes
   *  one model to the metaheuristic evaluation -- this model might be
   *  meaningful in some way, or the concrete implementation can choose
   *  to ignore it completely. It is further assumed that this model's
   *  `lbann_comm` object describes the entire multi-trainer ecosystem
   *  in which this algorithm is participating (which is why there's
   *  not a distinct `lbann_comm` argument).
   *
   *  @param[in,out] m On input, this trainer's candidate model. On
   *                 output, the updated model, either to e returned
   *                 from the training algorithm or to seed the next
   *                 outer loop.
   *  @param[in,out] ctxt The execution context for the outer LTFB
   *                 algorithm.
   *  @param[in,out] dc The data coordinator for this trainer.
   */
  virtual void select_next(model& m,
                           ltfb::ExecutionContext& ctxt,
                           data_coordinator& dc) const = 0;
}; // class MetaLearningStrategy

/** @brief A factory for constructing MetaLearningStrategy objects from Protobuf
 *         messages.
 */
using MetaLearningStrategyFactory = generic_factory<
  MetaLearningStrategy,
  std::string,
  proto::generate_builder_type<MetaLearningStrategy,
                               google::protobuf::Message const&>>;

using MetaLearningStrategyKey = MetaLearningStrategyFactory::id_type;
using MetaLearningStrategyBuilder = MetaLearningStrategyFactory::builder_type;

void register_new_metalearning_strategy(MetaLearningStrategyKey key,
                                        MetaLearningStrategyBuilder builder);

void unregister_metalearning_strategy(MetaLearningStrategyKey const& key);

} // namespace ltfb

template <>
std::unique_ptr<ltfb::MetaLearningStrategy>
make_abstract<ltfb::MetaLearningStrategy>(const google::protobuf::Message &msg);

} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_META_LEARNING_STRATEGY_HPP_INCLUDED
