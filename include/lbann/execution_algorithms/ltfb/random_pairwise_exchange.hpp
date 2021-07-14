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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_RANDOM_PAIRWISE_EXCHANGE_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_RANDOM_PAIRWISE_EXCHANGE_HPP_INCLUDED

#include "meta_learning_strategy.hpp"

#include <google/protobuf/message.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#ifdef LBANN_HAS_DYAD
#include "dyad_stream_api.hpp"
#endif // LBANN_HAS_DYAD

namespace lbann {
namespace ltfb {

/** @class RandomPairwiseExchange
 *  @brief The original LTFB algorithm.
 *
 *  In this case, a collection of metrics are provided (by name) and
 *  the winner of the tournament is chosen based on the metric
 *  strategy associated with each of them. The assumption is that the
 *  metric outputs a single scalar and the winner is chosen by a
 *  simple ">" or "<" operation.
 *
 *  The tournament partners are chosen according to an internal
 *  algorithm. Partners exchange their models with each other,
 *  according to the selected communication scheme, and the winner at
 *  each rank is chosen based on evaluating the metrics on the local
 *  trainer's data (one consequence being that each partner might
 *  select a different winner). The partner model must win ALL of the
 *  metrics to be declared the tournament winner.
 */
class RandomPairwiseExchange final
  : public Cloneable<RandomPairwiseExchange, MetaLearningStrategy>
{
public:
  /** @class ExchangeStrategy
   *  @brief A method for exchanging models with a partner trainer.
   *
   *  This class is an implementation detail of the
   *  "RandomPairwiseExchange" MetaLearningStrategy described below. It
   *  is not for general use.
   */
  class ExchangeStrategy
    : public Cloneable<HasAbstractFunction<ExchangeStrategy>>
  {
  public:
    /** @brief Construct with weights names
     *  @param[in] weights_names Names of weights to exchange. If
     *                           empty, then all weights are
     *                           exchanged.
     */
    ExchangeStrategy(std::set<std::string> weights_names)
      : m_weights_names{std::move(weights_names)}
    {}
    virtual ~ExchangeStrategy() = default;

    /** @brief Get the model from a partner trainer.
     *  @param[in] m The local model. This is for the "sendrecv_weights"
     *             strategy.  For other strategies, it's a proxy for the
     *             lbann_comm.
     *  @param[in] partner_trainer The ID of the partner trainer.
     *  @param[in] step The LTFB step ID.
     *
     *  @fixme The step parameter is only used by CheckpointFile; we
     *  should consider alternatives that don't clutter the API.
     */
    virtual std::unique_ptr<model>
    get_partner_model(model const& m, El::Int partner_trainer, size_t step) = 0;
    // Better API, but complicates "sendrecv_weights":
    // virtual std::unique_ptr<model> get_partner_model(
    //   lbann_comm const& c, El::Int partner_trainer);
  protected:
    /** @brief Access weights_names. */
    std::set<std::string> const& weights_names() const noexcept
    {
      return m_weights_names;
    }

  private:
    std::set<std::string> m_weights_names;
  }; // class ExchangeStrategy

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
   *             evaluation. A metric with this name must exist in the
   *             model passed to apply().
   *  @param[in] winner_strategy Strategy for determining the winner
   *             of a tournament.
   *  @param[in] comm_algo Algorithm for exchanging models.
   */
  RandomPairwiseExchange(std::string metric_name,
                         metric_strategy winner_strategy,
                         std::unique_ptr<ExchangeStrategy> comm_algo);

  /** @brief Constructor
   *  @param[in] metrics The list of metric/strategy pairs. A metric
   *             with each given name must exist in the model passed
   *             to apply(). The local model is favored. The partner
   *             model must win ALL of the metric comparisons to be
   *             declared the winner.
   *  @param[in] comm_algo Algorithm for exchanging models.
   */
  RandomPairwiseExchange(
    std::unordered_map<std::string, metric_strategy> metrics,
    std::unique_ptr<ExchangeStrategy> comm_algo);

  ~RandomPairwiseExchange() = default;
  RandomPairwiseExchange(RandomPairwiseExchange const& other);
  ///@}

  /** @brief Engage in a tournament with a partner trainer.

   *  @param[in,out] m On input, the locally computed model. On
   *                 output, the winning model with respect to the
   *                 tournament.
   *  @param[in,out] ctxt The execution context for the outer LTFB
   *                 wrapper.
   *  @param[in,out] dc The data source for the tournament.
   */
  void select_next(model& m,
                   ltfb::ExecutionContext& ctxt,
                   data_coordinator& dc) const final;

private:
  /** @brief Get the value of the given metric from the model. */
  std::unordered_map<std::string, EvalType>
  evaluate_model(model& m, ExecutionContext& ctxt, data_coordinator& dc) const;
  /** @brief Generate a new trainer partner from the comm. */
  El::Int get_partner_trainer(lbann_comm const& c) const noexcept;
  /** @brief Evaluate the output of two models according to the input
   *         metric strategies.
   *
   *  The local model is preferred if possible (to avoid a model
   *  move). That is, "<=" or ">=" is used.
   *
   *  @param[in] local_scores The metric outputs of the local model.
   *  @param[in] partner_scores The metric output of the remote model.
   *  @returns true if the local model is favored.
   */
  bool local_is_better(
    std::unordered_map<std::string, EvalType> const& local_scores,
    std::unordered_map<std::string, EvalType> const& partner_scores) const;

private:
  /** @brief The list of metric/strategy pairs.
   *
   *  Each metric gets its own strategy. A partner model must win
   *  every metric to be declared the tournament winner.
   */
  std::unordered_map<std::string, metric_strategy> m_metrics;

  /** @brief The strategy for exchanging two models.
   *
   *  This is largely an implementation detail of moving models
   *  around. It shouldn't be used to alter the algorithmic goings-on
   *  of the method. Ideally, it should disappear as we converge on a
   *  "universally best method" for moving things around (let's
   *  pretend such a thing exists; it makes me feel better, anyway).
   */
  std::unique_ptr<ExchangeStrategy> m_comm_algo;

}; // class RandomPairwiseExchange

/** @class SendRecvWeights
 *  @brief Exchange model weights directly using sendrecvs.
 *  @todo More general approach to exchange optimizer state. Currently
 *  only SGD and Adam are supported.
 */
class SendRecvWeights final
  : public Cloneable<SendRecvWeights, RandomPairwiseExchange::ExchangeStrategy>
{
  using BaseType =
    Cloneable<SendRecvWeights, RandomPairwiseExchange::ExchangeStrategy>;

public:
  /** @brief Construct from weights names
   *  @param[in] weights_names Names of weights to exchange. If empty,
   *                           then all weights are exchanged.
   *  @param[in] exchange_hyperparameters Exchange optimizer
   *                                      hyperparameters.
   */
  SendRecvWeights(std::set<std::string> const& weights_names,
                  bool exchange_hyperparameters);

  /** @brief Construct from weights names
   *  @param[in] weights_names Names of weights to exchange. If empty,
   *                           then all weights are exchanged.
   *  @param[in] exchange_hyperparameters Exchange optimizer
   *                                      hyperparameters.
   */
  SendRecvWeights(std::set<std::string>&& weights_names,
                  bool exchange_hyperparameters);

  SendRecvWeights(SendRecvWeights const&) = default;
  SendRecvWeights(SendRecvWeights&&) = default;

  std::unique_ptr<model> get_partner_model(model const& m,
                                           El::Int partner_trainer,
                                           size_t /*step*/) final;

private:
  bool exchange_hyperparams_;
}; // class SendRecvWeights

/// See @c lbann::callbacks::ltfb::communication_algorithm::checkpoint_file
class CheckpointFile final
  : public Cloneable<CheckpointFile, RandomPairwiseExchange::ExchangeStrategy>
{
  using BaseType =
    Cloneable<CheckpointFile, RandomPairwiseExchange::ExchangeStrategy>;

public:
  CheckpointFile(std::set<std::string> const& weights_names,
                 std::string const& ckpt_basedir);
  CheckpointFile(std::set<std::string>&& weights_names,
                 std::string const& ckpt_basedir);
  std::unique_ptr<model>
  get_partner_model(model const& m, El::Int partner_trainer, size_t step) final;

private:
  std::string ckpt_basedir_;
}; // class CheckpointFile

#ifdef LBANN_HAS_DYAD
/// See @c lbann::callbacks::ltfb::communication_algorithm::checkpoint_file_dyad
class CheckpointFileDyad final
  : public Cloneable<CheckpointFileDyad, RandomPairwiseExchange::ExchangeStrategy>
{
  using BaseType =
    Cloneable<CheckpointFileDyad, RandomPairwiseExchange::ExchangeStrategy>;

public:
  CheckpointFileDyad(std::set<std::string> const& weights_names,
                     const dyad::dyad_params& dparams);
  CheckpointFileDyad(std::set<std::string>&& weights_names,
                     const dyad::dyad_params& dparams);
  std::unique_ptr<model>
  get_partner_model(model const& m, El::Int partner_trainer, size_t step) final;

  dyad::dyad_stream_core& get_dyad();

private:
  std::string ckpt_basedir_;

  dyad::dyad_stream_core m_dyad;
}; // class CheckpointFileDyad
#endif // LBANN_HAS_DYAD

class CheckpointBinary final
  : public Cloneable<CheckpointBinary, RandomPairwiseExchange::ExchangeStrategy>
{
  using BaseType =
    Cloneable<CheckpointBinary, RandomPairwiseExchange::ExchangeStrategy>;

public:
  CheckpointBinary(std::set<std::string> const& weights_names);
  CheckpointBinary(std::set<std::string>&& weights_names);
  std::unique_ptr<model> get_partner_model(model const& m,
                                           El::Int partner_trainer,
                                           size_t /*step*/) final;
}; // class CheckpointBinary

} // namespace ltfb

/** @name Builder functions */
///@{

/** @brief Concrete product builder for RandomPairwiseExchange. */
template <>
std::unique_ptr<ltfb::RandomPairwiseExchange>
make(google::protobuf::Message const&);

///@}

} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_RANDOM_PAIRWISE_EXCHANGE_HPP_INCLUDED
