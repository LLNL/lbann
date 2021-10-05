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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_REGULARIZED_EVOLUTION_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_REGULARIZED_EVOLUTION_HPP_INCLUDED

#include "mutation_strategy.hpp"

#include "meta_learning_strategy.hpp"

#include <google/protobuf/message.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace lbann {
namespace ltfb {

/** @class RegularizedEvolution
 *  This is a meta-learning strategy in population-based training.
 *  A sample of trainers is chosen from a population in every tournament.
 *  The best trainer is chosen from that sample according to an evaluation
 * metric. Then the model from that best trainer is mutated and replaces the
 * oldest model.
 */

class RegularizedEvolution final
  : public Cloneable<RegularizedEvolution, MetaLearningStrategy>
{
public:
  enum class metric_strategy
  {
    LOWER_IS_BETTER,
    HIGHER_IS_BETTER,
  }; // enum class metric_strategy

public:
  RegularizedEvolution(std::string metric_name,
                       metric_strategy winner_strategy,
                       std::unique_ptr<MutationStrategy> mutate_algo,
                       int sample_size);
  ~RegularizedEvolution() = default;
  RegularizedEvolution(RegularizedEvolution const& other);

  void select_next(model& m,
                   ltfb::ExecutionContext& ctxt,
                   data_coordinator& dc) const final;

private:
  /** @brief Get the value of the given metric from the model. */
  EvalType
  evaluate_model(model& m, ExecutionContext& ctxt, data_coordinator& dc) const;

private:
  /** @brief The strategy for mutation of a model
   *
   *  When a trainer loses in a LTFB tournament, the winning model is
   *  copied over to it and this mutation strategy is applied to the
   *  copied model to explore a new model. This is relevant to neural
   *  architecture search (NAS).
   */
  std::unique_ptr<MutationStrategy> m_mutate_algo;

  /** @brief Name of the metric for evaluation
   */
  std::string m_metric_name;

  /** @brief Strategy to consider for evaluating the metric
   *  e.g., HIGHER_IS_BETTER or LOWER_IS_BETTER
   */
  metric_strategy m_metric_strategy;

  /** @brief The size of the sample to choose from the population in every step
   */
  int m_sample_size;

}; // class RegularizedEvolution

} // namespace ltfb

/** @name Builder functions */
///@{

/** @brief Concrete product builder for RegularizedEvolution. */
template <>
std::unique_ptr<ltfb::RegularizedEvolution>
make(google::protobuf::Message const&);

///@}

} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_REGULARIZED_EVOLUTION_HPP_INCLUDED
