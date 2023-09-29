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

#include "lbann/execution_algorithms/ltfb/mutation_strategy.hpp"

#include "lbann/execution_algorithms/ltfb/regularized_evolution.hpp"

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

// RegularizedEvolution Implementation

RegularizedEvolution::RegularizedEvolution(
  std::string metric_name,
  metric_strategy winner_strategy,
  std::unique_ptr<MutationStrategy> mutate_algo,
  int sample_size)
  : m_mutate_algo{std::move(mutate_algo)},
    m_metric_name{std::move(metric_name)},
    m_metric_strategy{std::move(winner_strategy)},
    m_sample_size{std::move(sample_size)}
{}

RegularizedEvolution::RegularizedEvolution(RegularizedEvolution const& other)
  : m_mutate_algo{other.m_mutate_algo->clone()},
    m_metric_name{other.m_metric_name},
    m_metric_strategy{other.m_metric_strategy},
    m_sample_size{other.m_sample_size}
{}

EvalType RegularizedEvolution::evaluate_model(model& m,
                                              LTFBExecutionContext& ctxt,
                                              data_coordinator& dc) const

{
  // Make sure data readers finish asynchronous work
  const auto original_mode = ctxt.get_execution_mode();
  dc.collect_background_data_fetch(original_mode);

  // Can use validation if it is global
  if (!dc.is_execution_mode_valid(execution_mode::tournament)) {
    LBANN_ERROR("Regularized Evolution requires ",
                to_string(execution_mode::tournament),
                " execution mode");
  }

  // Mark the data store as loading - Note that this is a temporary fix
  // for the current use of the tournament
  dc.mark_data_store_explicitly_loading(execution_mode::tournament);

  // Evaluate model on test (or validation?) set
  get_trainer().evaluate(&m, execution_mode::tournament);

  // Get metric values
  bool found_metric = false;
  EvalType score = 0.f;
  std::string metric_name;
  for (const auto& met : m.get_metrics()) {
    metric_name = met->name();
    if (metric_name == m_metric_name) {
      found_metric = true;
      score = met->get_mean_value(execution_mode::tournament);
      break;
    }
  }

  // sanity check
  if (!found_metric) {
    LBANN_ERROR("could not find metric \"",
                m_metric_name,
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

void RegularizedEvolution::select_next(model& m,
                                       ltfb::LTFBExecutionContext& ctxt,
                                       data_coordinator& dc) const
{
  auto const& comm = *(m.get_comm());
  const unsigned int num_trainers = comm.get_num_trainers();
  const int trainer_id = comm.get_trainer_rank();
  auto const step = ctxt.get_step();

  std::vector<unsigned> sample_trainers(num_trainers);
  if (comm.am_world_master()) {
    std::iota(begin(sample_trainers), end(sample_trainers), 0U);
    std::shuffle(sample_trainers.begin(),
                 sample_trainers.end(),
                 get_ltfb_generator());

    // Print trainers selected in sample
    std::cout << "Trainers in sample at step " << step << " -";
    for (int i = 0; i < m_sample_size; i++)
      std::cout << " " << sample_trainers[i];
    std::cout << std::endl;
  }
  comm.world_broadcast(comm.get_world_master(),
                       sample_trainers.data(),
                       num_trainers);

  El::Int score = evaluate_model(m, ctxt, dc);

  // AllGather scores from all trainers
  std::vector<EvalType> score_list_all(num_trainers);
  comm.trainer_barrier();
  if (comm.am_trainer_master()) {
    comm.all_gather<EvalType>(score,
                              score_list_all,
                              comm.get_intertrainer_comm());
  }

  // Use scores only for samples selected from sample_trainers above
  // and place them in the same order as in sample_trainers
  std::vector<EvalType> score_list_samples(m_sample_size);
  for (int i = 0; i < m_sample_size; i++) {
    score_list_samples[i] = score_list_all[sample_trainers[i]];
  }

  // Communicate sample score list from trainer master to other procs in
  // trainer
  comm.trainer_broadcast(comm.get_trainer_master(),
                         score_list_samples.data(),
                         m_sample_size);

  // Find winning trainer in sample according to metric strategy
  El::Int winner_id;
  if (m_metric_strategy ==
      RegularizedEvolution::metric_strategy::HIGHER_IS_BETTER)
    winner_id = sample_trainers[std::distance(
      score_list_samples.begin(),
      std::max_element(score_list_samples.begin(), score_list_samples.end()))];
  else if (m_metric_strategy ==
           RegularizedEvolution::metric_strategy::LOWER_IS_BETTER)
    winner_id = sample_trainers[std::distance(
      score_list_samples.begin(),
      std::min_element(score_list_samples.begin(), score_list_samples.end()))];
  else
    LBANN_ERROR("Invalid metric strategy!");

  // Find oldest trainer - cycle through trainer ids
  El::Int oldest_id = step % num_trainers;

  // Print winning and oldest model
  if (comm.am_world_master()) {
    std::cout << "Winner - " << winner_id << ", Oldest - " << oldest_id
              << std::endl;
  }

  if (trainer_id == winner_id) {

    if (winner_id != oldest_id) {
      auto model_string = pack(m);
      if (comm.am_trainer_master()) {
        send_string(comm, model_string, oldest_id);
        std::cout << "In Reg Evo step " << step << ", trainer " << trainer_id
                  << " with score " << score_list_all[trainer_id]
                  << " sends model to trainer " << oldest_id << std::endl;
      }
    }
  }

  if (trainer_id == oldest_id) {

    if (winner_id != oldest_id) {
      std::string rcv_str;
      if (comm.am_trainer_master()) {
        rcv_str = recv_string(comm, winner_id);
        std::cout << "In Reg Evo step " << step << ", trainer " << trainer_id
                  << " receives model from trainer " << winner_id << std::endl;
      }

      unpack(m, rcv_str);
    }

    // Mutating oldest model
    m_mutate_algo->mutate(m, step);

    auto& trainer = get_trainer();
    auto&& metadata = trainer.get_data_coordinator().get_dr_metadata();
    m.setup(trainer.get_max_mini_batch_size(),
            metadata,
            trainer.get_grids(),
            /*force*/ true);
  }
}

} // namespace ltfb
} // namespace lbann

namespace {

lbann::ltfb::RegularizedEvolution::metric_strategy
to_lbann(lbann_data::RegularizedEvolution::MetricStrategy strategy)
{
  using LBANNEnumType = lbann::ltfb::RegularizedEvolution::metric_strategy;
  using ProtoEnumType = lbann_data::RegularizedEvolution::MetricStrategy;
  switch (strategy) {
  case ProtoEnumType::RegularizedEvolution_MetricStrategy_LOWER_IS_BETTER:
    return LBANNEnumType::LOWER_IS_BETTER;
  case ProtoEnumType::RegularizedEvolution_MetricStrategy_HIGHER_IS_BETTER:
    return LBANNEnumType::HIGHER_IS_BETTER;
  default:
    LBANN_ERROR("Unknown enum value: ", static_cast<int>(strategy));
  }
  return LBANNEnumType::LOWER_IS_BETTER;
}

} // namespace

/** @brief Builder function for RegularizedEvolution. */
template <>
std::unique_ptr<lbann::ltfb::RegularizedEvolution>
lbann::make<lbann::ltfb::RegularizedEvolution>(
  google::protobuf::Message const& msg_in)
{
  auto const& params = dynamic_cast<google::protobuf::Any const&>(msg_in);
  lbann_data::RegularizedEvolution msg;
  LBANN_ASSERT(params.UnpackTo(&msg));

  using MutationStrategyType = lbann::ltfb::MutationStrategy;

  return std::make_unique<lbann::ltfb::RegularizedEvolution>(
    msg.metric_name(),
    to_lbann(msg.metric_strategy()),
    make_abstract<MutationStrategyType>(msg.mutation_strategy()),
    msg.sample_size());
}
