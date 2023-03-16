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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_TRUNCATION_SELECTION_EXCHANGE_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_TRUNCATION_SELECTION_EXCHANGE_HPP_INCLUDED

#include "meta_learning_strategy.hpp"

#include <google/protobuf/message.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace lbann {
namespace ltfb {

/** @class TruncationSelectionExchange
 *
 *  A variant of classic LTFB or exploitation mechanism in
 *  population-based training. All trainers in the population set are
 *  ranked using specified evaluation metric. Model parameters,
 *  training hyperparameters and or topologies of any trainer in the
 *  bottom rank is replaced by that of a (random) trainer in the top
 *  rank.
 */
class TruncationSelectionExchange final
  : public Cloneable<TruncationSelectionExchange, MetaLearningStrategy>
{
public:
  enum class metric_strategy
  {
    LOWER_IS_BETTER,
    HIGHER_IS_BETTER,
  }; // enum class metric_strategy

public:
  /** @name Life-cycle management */
  ///@{
  /** @brief Constructor
   *  @param[in] metric_name The name of the metric to use for
   *                         evaluation. A metric with this name must
   *                         exist in the model passed to apply().
   *  @param[in] winner_strategy Strategy for determining the winner
   *                             of a tournament.
   *  @param[in] truncation_k Partitions ranking list to
   *                          top(winners)/bottom(losers)
   */
  TruncationSelectionExchange(std::string metric_name,
                              metric_strategy winner_strategy,
                              int truncation_k);

  /** @brief Constructor
   *  @param[in] metrics The metric/strategy pairs. A metric with each
   *                     given name must exist in the model passed to
   *                     apply().
   *  @param[in] truncation_k Partitions ranking list to
   *                          top(winners)/bottom(losers)
   */
  TruncationSelectionExchange(
    std::unordered_map<std::string, metric_strategy> metrics,
    int truncation_k);

  ~TruncationSelectionExchange() = default;
  TruncationSelectionExchange(TruncationSelectionExchange const& other);
  ///@}

  /** @brief Engage in a ranked tournament with a population of trainers.

   *  @param[in,out] m On input, the locally computed model. On
   *                 output, the selected (winning model) with respect to the
   *                 tournament.
   *  @param[in,out] ctxt The execution context for the outer LTFB
   *                 wrapper.
   *  @param[in,out] dc The data source for the tournament.
   */
  void select_next(model& m,
                   ltfb::LTFBExecutionContext& ctxt,
                   data_coordinator& dc) const final;

private:
  /** @brief Get the value of the given metric from the model. */
  EvalType evaluate_model(model& m,
                          LTFBExecutionContext& ctxt,
                          data_coordinator& dc) const;

private:
  /** @brief The list of metric/strategy pairs.
   *
   *  Each metric gets its own strategy.
   *  Note, only one metric and strategy pair is currently supported.
   *  List (map) is for compatibility with classic LTFB (RPE)
   *  And as a placeholder for when multiple metrics are needed.
   */
  std::unordered_map<std::string, metric_strategy> m_metrics;

  /* partitions ranking of trainers to top (winners) and below (loosers)
   */
  int m_truncation_k;

}; // class TruncationSelectionExchange

} // namespace ltfb

/** @name Builder functions */
///@{

/** @brief Concrete builder for TruncationSelectionExchange. */
template <>
std::unique_ptr<ltfb::TruncationSelectionExchange>
make(google::protobuf::Message const&);

///@}

} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_TRUNCATION_SELECTION_EXCHANGE_HPP_INCLUDED
