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

#include "lbann/execution_algorithms/ltfb/mutation_strategy.hpp"

#include "lbann/execution_algorithms/ltfb/regularized_evolution.hpp"

//#include "lbann/execution_algorithms/ltfb/random_pairwise_exchange.hpp"

#include "lbann/base.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/models/directed_acyclic_graph.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/weights/data_type_weights_impl.hpp"

#include <training_algorithm.pb.h>

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

// Pack model to ship off
std::string pack(model const& m)
{
  std::ostringstream oss;
  {
    RootedBinaryOutputArchive ar(oss, m.get_comm()->get_trainer_grid());
    ar(m);
  }
  return oss.str();
}

// Send a string to the root of the destination trainer
void send_string(lbann_comm const& comm,
                 std::string const& str,
                 int destination_trainer)
{
  size_t size = str.length();
  comm.send(&size, 1, destination_trainer, /*rank=*/0);
  comm.send(reinterpret_cast<El::byte const*>(str.data()),
            size,
            destination_trainer,
            /*rank=*/0);
}

// Receive a string from the root of src_trainer
std::string recv_string(lbann_comm const& comm, int src_trainer)
{
  size_t size = 0;
  comm.recv(&size, 1, src_trainer);
  std::string buf;
  buf.resize(size);
  comm.recv(reinterpret_cast<El::byte*>(buf.data()), size, src_trainer);
  return buf;
}

// Unpack received model
void unpack(model& m, std::string const& str)
{
  std::istringstream iss(str);
  {
    RootedBinaryInputArchive ar(iss, m.get_comm()->get_trainer_grid());
    ar(m);
  }
}

} // namespace

// RegularizedEvolution Implementation

RegularizedEvolution::RegularizedEvolution(
  std::unordered_map<std::string, metric_strategy> metrics,
  std::unique_ptr<MutationStrategy> mutate_algo)
  : m_metrics{std::move(metrics)}, m_mutate_algo{std::move(mutate_algo)}
{
  LBANN_ASSERT(m_metrics.size());
}

RegularizedEvolution::RegularizedEvolution(
  std::string metric_name,
  metric_strategy winner_strategy,
  std::unique_ptr<MutationStrategy> mutate_algo)
  : RegularizedEvolution({{std::move(metric_name), winner_strategy}},
                         std::move(mutate_algo))
{}

RegularizedEvolution::RegularizedEvolution(RegularizedEvolution const& other)
  : m_metrics{other.m_metrics}, m_mutate_algo{other.m_mutate_algo->clone()}
{}

EvalType RegularizedEvolution::evaluate_model(model& m,
                                              ExecutionContext& ctxt,
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
  m.mark_data_store_explicitly_loading(execution_mode::tournament);

  // Evaluate model on test (or validation?) set
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

  m.make_data_store_preloaded(execution_mode::testing);

  // Clean up and return metric score
  m.reset_mode(ctxt, original_mode);
  dc.reset_mode(ctxt);
  return score;
}

void RegularizedEvolution::select_next(model& m,
                                       ltfb::ExecutionContext& ctxt,
                                       data_coordinator& dc) const
{
  // Find the best model trainer
  // Find the oldest model trainer (highest age)
  // Copy the best model trainer to oldest model trainer, mutate it and set its
  // age to 0

  auto const& comm = *(m.get_comm());
  const unsigned int num_trainers = comm.get_num_trainers();
  const int trainer_id = comm.get_trainer_rank();
  auto const step = ctxt.get_step();

  // Choose sample S<=P
  El::Int S = 3; //num_trainers;
  std::vector<EvalType> sample_trainers(num_trainers);
  if (comm.am_world_master()) {
    for (unsigned int i = 0; i < num_trainers; i++)
      sample_trainers[i] = i;
    std::shuffle(sample_trainers.begin(),
                 sample_trainers.end(),
                 get_ltfb_generator());
  }
  comm.world_broadcast(comm.get_world_master(),
                       sample_trainers.data(),
                       num_trainers);

  // if rank within first S, send score , else score = 0
  auto it =
    std::find(sample_trainers.begin(), sample_trainers.end(), trainer_id);

  El::Int score;
  // If in sample, send true score
  if (std::distance(sample_trainers.begin(), it) < S) {
    score = evaluate_model(m, ctxt, dc);
  }
  // Else send dummy score
  else {
    score = 0;
  }

  std::vector<EvalType> score_list(num_trainers);
  comm.trainer_barrier();
  if (comm.am_trainer_master()) {
    comm.all_gather<EvalType>(score, score_list, comm.get_intertrainer_comm());
  }
  // Communicate trainer score list from trainer master to other procs in
  // trainer
  comm.trainer_broadcast(comm.get_trainer_master(),
                         score_list.data(),
                         num_trainers);

  // Find winning trainer (in sample)
  El::Int winner_id =
    std::distance(score_list.begin(),
                  std::max_element(score_list.begin(), score_list.end()));

  // Find oldest trainer - cycle through trainer ids
  El::Int oldest_id = step % num_trainers;

  // DEBUG
  std::cout << "Winner - " << winner_id << ", Oldest - " << oldest_id
            << std::endl;

  if (trainer_id == winner_id) {

    if (winner_id != oldest_id) {
      auto model_string = pack(m);
      if (comm.am_trainer_master()) {
        send_string(comm, model_string, oldest_id);
        std::cout << "In Reg Evo step " << step << ", trainer " << trainer_id
                  << " with score " << score_list[trainer_id];
        std::cout << " sends model to trainer " << oldest_id << std::endl;
      }
    }
  }

  if (trainer_id == oldest_id) {

    auto partner_model_ptr = m.copy_model();
    auto& partner_model = *partner_model_ptr;

    if (winner_id != oldest_id) {
      std::string rcv_str;
      if (comm.am_trainer_master()) {
        rcv_str = recv_string(comm, winner_id);
        std::cout << "In Reg Evo step " << step << ", trainer " << trainer_id;
        std::cout << " receives model from trainer " << winner_id << std::endl;
      }

      unpack(partner_model, rcv_str);
    }

    // Mutating oldest model
    m_mutate_algo->mutate(partner_model, step);

    auto& trainer = get_trainer();
    auto&& metadata = trainer.get_data_coordinator().get_dr_metadata();
    m.setup(trainer.get_max_mini_batch_size(),
            metadata,
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

template <>
std::unique_ptr<lbann::ltfb::RegularizedEvolution>
lbann::make<lbann::ltfb::RegularizedEvolution>(
  google::protobuf::Message const& msg_in)
{
  auto const& params = dynamic_cast<google::protobuf::Any const&>(msg_in);
  lbann_data::RegularizedEvolution msg;
  LBANN_ASSERT(params.UnpackTo(&msg));

  // Copy the metric map into LBANN format.
  using MetricStrategy = ltfb::RegularizedEvolution::metric_strategy;
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

  using MutationStrategyType = lbann::ltfb::MutationStrategy;

  return make_unique<lbann::ltfb::RegularizedEvolution>(
    std::move(metric_map),
    make_abstract<MutationStrategyType>(msg.mutation_strategy()));
}
