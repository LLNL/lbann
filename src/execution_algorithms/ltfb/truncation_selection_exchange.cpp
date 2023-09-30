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

#include "lbann/execution_algorithms/ltfb/truncation_selection_exchange.hpp"

#include "checkpoint_common.hpp"

#include "lbann/base.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/weights/data_type_weights_impl.hpp"

#include "lbann/proto/training_algorithm.pb.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace lbann {
namespace ltfb {
namespace {

bool low_score_wins(TruncationSelectionExchange::metric_strategy strategy)
{
  using MetricStrategy = TruncationSelectionExchange::metric_strategy;
  switch (strategy) {
  case MetricStrategy::LOWER_IS_BETTER:
    return true;
  case MetricStrategy::HIGHER_IS_BETTER:
    return false;
  default:
    LBANN_ERROR("Invalid metric strategy!");
  }
  return true; // Silence compiler warning about no return.
}

} // namespace

// TruncationSelectionExchange implementation

TruncationSelectionExchange::TruncationSelectionExchange(
  std::unordered_map<std::string, metric_strategy> metrics,
  int truncation_k)
  : m_metrics{std::move(metrics)}, m_truncation_k{std::move(truncation_k)}
{
  LBANN_ASSERT(m_metrics.size() ==
               1); // only single (1) metric is supported at this time
}

TruncationSelectionExchange::TruncationSelectionExchange(
  std::string metric_name,
  metric_strategy winner_strategy,
  int truncation_k)
  : TruncationSelectionExchange({{std::move(metric_name), winner_strategy}},
                                truncation_k)
{}

TruncationSelectionExchange::TruncationSelectionExchange(
  TruncationSelectionExchange const& other)
  : m_metrics{other.m_metrics}, m_truncation_k{other.m_truncation_k}
{}

EvalType TruncationSelectionExchange::evaluate_model(model& m,
                                                     LTFBExecutionContext& ctxt,
                                                     data_coordinator& dc) const
{
  // Make sure data readers finish asynchronous work
  const auto original_mode = ctxt.get_execution_mode();
  dc.collect_background_data_fetch(original_mode);

  // Can use validation if it is global
  if (!dc.is_execution_mode_valid(execution_mode::tournament)) {
    LBANN_ERROR("LTFB truncation selection requires ",
                to_string(execution_mode::tournament),
                " execution mode");
  }

  // Mark the data store as loading - Note that this is a temporary fix
  // for the current use of the tournament
  dc.mark_data_store_explicitly_loading(execution_mode::tournament);

  // Evaluate model on validation set
  get_trainer().evaluate(&m, execution_mode::tournament);

  // Get metric values
  bool found_metric = false;
  EvalType score = 0.f;
  std::string metric_name;
  for (const auto& met : m.get_metrics()) {
    metric_name = met->name();
    if (m_metrics.count(metric_name)) {
      found_metric = true;
      score += met->get_mean_value(execution_mode::tournament);
      break;
    }
  }

  // sanity check
  if (!found_metric) {
    LBANN_ERROR("could not find metric \"",
                metric_name,
                "\" "
                "in model \"",
                m.get_name(),
                "\"");
  }

  dc.make_data_store_preloaded(execution_mode::tournament);

  // Clean up and return metric score
  m.reset_mode(ctxt, original_mode);
  dc.reset_mode(ctxt);
  return score;
}

void TruncationSelectionExchange::select_next(model& m,
                                              ltfb::LTFBExecutionContext& ctxt,
                                              data_coordinator& dc) const
{
  auto const& comm = *(m.get_comm());
  const unsigned int num_trainers = comm.get_num_trainers();
  const int trainer_id = comm.get_trainer_rank();
  auto const step = ctxt.get_step();

  auto score = evaluate_model(m, ctxt, dc);

  // epsilon, avoid (close) duplicity in score
  score += 0.00000001 * trainer_id;
  // trainer master computes trainer metric score rank/position
  std::vector<EvalType> score_list(num_trainers);
  comm.trainer_barrier();
  if (comm.am_trainer_master()) {
    comm.all_gather<EvalType>(score, score_list, comm.get_intertrainer_comm());
  }
  // Communicate trainer score list from trainer master processes
  comm.trainer_broadcast(comm.get_trainer_master(),
                         score_list.data(),
                         num_trainers);
  std::vector<EvalType> top_scores = score_list;
  // top-k in an ascending order
  // supports of singular metric value (for now)
  auto met_strategy = m_metrics.begin()->second;
  if (low_score_wins(met_strategy))
    std::sort(top_scores.begin(), top_scores.end(), std::less<EvalType>());
  // top-k in an descending order
  else
    std::sort(top_scores.begin(), top_scores.end(), std::greater<EvalType>());
  auto itr1 = std::adjacent_find(top_scores.begin(), top_scores.end());
  if (itr1 != top_scores.end()) {
    LBANN_ERROR("truncation tournament exchange currently works if trainers "
                "scores are unique");
  }

  auto itr2 =
    std::find(top_scores.begin(), top_scores.end(), score_list[trainer_id]);
  auto trainer_score_pos = std::distance(top_scores.begin(), itr2);

  if (trainer_score_pos < m_truncation_k) {
    // Winner (in top-k)
    // for each loosing trainer
    for (unsigned int i = m_truncation_k; i < num_trainers; i++) {
      if (trainer_score_pos == i % m_truncation_k) {
        // One of partners is trainer that owns score at top_scores[i]
        auto dest = std::distance(
          score_list.begin(),
          std::find(score_list.begin(), score_list.end(), top_scores[i]));

        auto model_string = pack(m);
        if (comm.am_trainer_master()) {
          send_string(comm, model_string, dest);
          std::cout << "In LTFB TSE step " << step << ", trainer " << trainer_id
                    << " with score " << score_list[trainer_id];
          std::cout << " sends model to trainer  " << dest << " with score "
                    << score_list[dest] << std::endl;
        }
      }
    }
  }
  else { // not in top-k, receive
    auto src =
      std::distance(score_list.begin(),
                    std::find(score_list.begin(),
                              score_list.end(),
                              top_scores[trainer_score_pos % m_truncation_k]));

    std::string rcv_str;
    if (comm.am_trainer_master()) {
      rcv_str = recv_string(comm, src);
      std::cout << "In LTFB TSE step " << step << ", trainer " << trainer_id
                << " with score " << score_list[trainer_id];
      std::cout << " receives model from trainer " << src << " with score "
                << score_list[src] << std::endl;
    }

    unpack(m, rcv_str);
    auto& trainer = get_trainer();
    m.setup(trainer.get_max_mini_batch_size(),
            trainer.get_grids(),
            /*force*/ true);
  }
}

} // namespace ltfb
} // namespace lbann

namespace {
lbann::ltfb::TruncationSelectionExchange::metric_strategy
to_lbann(lbann_data::TruncationSelectionExchange::MetricStrategy strategy)
{
  using LBANNEnumType =
    lbann::ltfb::TruncationSelectionExchange::metric_strategy;
  using ProtoEnumType = lbann_data::TruncationSelectionExchange::MetricStrategy;
  switch (strategy) {
  case ProtoEnumType::
    TruncationSelectionExchange_MetricStrategy_LOWER_IS_BETTER:
    return LBANNEnumType::LOWER_IS_BETTER;
  case ProtoEnumType::
    TruncationSelectionExchange_MetricStrategy_HIGHER_IS_BETTER:
    return LBANNEnumType::HIGHER_IS_BETTER;
  default:
    LBANN_ERROR("Unknown enum value: ", static_cast<int>(strategy));
  }
  return LBANNEnumType::LOWER_IS_BETTER;
}

} // namespace

/** @brief Builder function for TruncationSelectionExchange. */
template <>
std::unique_ptr<lbann::ltfb::TruncationSelectionExchange>
lbann::make<lbann::ltfb::TruncationSelectionExchange>(
  google::protobuf::Message const& msg_in)
{
  auto const& params = dynamic_cast<google::protobuf::Any const&>(msg_in);
  lbann_data::TruncationSelectionExchange msg;
  LBANN_ASSERT(params.UnpackTo(&msg));

  // Copy the metric map into LBANN format.
  using MetricStrategy = ltfb::TruncationSelectionExchange::metric_strategy;
  std::unordered_map<std::string, MetricStrategy> metric_map;
  std::transform(msg.metric_name_strategy_map().cbegin(),
                 msg.metric_name_strategy_map().cend(),
                 std::inserter(metric_map, metric_map.end()),
                 [](auto const& kvp) {
                   using MapType =
                     std::unordered_map<std::string, MetricStrategy>;
                   using ValueType = typename MapType::value_type;
                   return ValueType{kvp.first, to_lbann(kvp.second)};
                 });

  return std::make_unique<lbann::ltfb::TruncationSelectionExchange>(
    std::move(metric_map),
    msg.truncation_k());
}
