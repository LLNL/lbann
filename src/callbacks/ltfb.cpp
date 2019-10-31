////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/ltfb.hpp"
#include "lbann/callbacks/imcomm.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/optimizers/sgd.hpp"
#include "lbann/optimizers/adam.hpp"
#include "lbann/proto/factories.hpp"

#include <callbacks.pb.h>

#include <sstream>
#include <string>
#include <tuple>

namespace lbann {
namespace callback {

namespace {

/** @brief Generate partner trainer assignments.
 *
 *  Requires a scatter from the world master process. If there are an
 *  odd number of trainers, one of them is partnered with itself.
 */
template <typename TensorDataType>
El::Int get_partner_trainer(lbann_comm& comm,
                            const std::string& message_prefix) {
  if (comm.am_world_master()) { // Root process

    // Assign partner trainers
    // Note: The first trainer in 'trainers' is paired with the
    // second, the third with the fourth, and so on. If there are an
    // odd number of trainers, the last one is partnered with itself.
    const El::Int num_trainers = comm.get_num_trainers();
    const El::Int procs_per_trainer = comm.get_procs_per_trainer();
    std::vector<El::Int> trainers(num_trainers);
    std::iota(trainers.begin(), trainers.end(), 0);
    std::shuffle(trainers.begin(), trainers.end(), get_fast_generator());

    // Print partner assignments to standard output
    std::stringstream msg;
    msg << message_prefix << "tournament partners -";
    for (El::Int i = 0; i < num_trainers; i += 2) {
      msg << (i > 0 ? "," : "")
          << " {" << trainers[i];
      if (i+1 < num_trainers) {
        msg << "," << trainers[i+1];
      }
      msg << "}";
    }
    msg << "\n";
    std::cout << msg.str();

    // Send partner assignments to all processes
    std::vector<El::Int> send_buffer(num_trainers * procs_per_trainer);
    for (El::Int i = 0; i < num_trainers; i += 2) {
      const auto& trainer1 = trainers[i];
      const auto& trainer2 = (i+1 < num_trainers) ? trainers[i+1] : trainer1;
      std::fill_n(&send_buffer[trainer1 * procs_per_trainer],
                  procs_per_trainer, trainer2);
      std::fill_n(&send_buffer[trainer2 * procs_per_trainer],
                  procs_per_trainer, trainer1);
    }
    return comm.scatter(send_buffer.data(), comm.get_world_comm());

  } else { // Non-root process
    return comm.scatter<El::Int>(comm.get_world_master(),
                                 comm.get_world_comm());
  }
}

/// See @c lbann::callbacks::ltfb::communication_algorithm::sendrecv_weights
namespace sendrecv_weights {

/** @param weights_names    Names of weights to exchange. If empty,
 *                          then all weights are exchanged.
 *  @param send_weights     Weights values sent to partner.
 *  @param recv_weights     Weights values recieved from partner.
 */
template <typename TensorDataType>
void exchange_models(lbann_comm& comm,
                     El::Int partner_trainer,
                     const std::set<std::string>& weights_names,
                     const std::vector<weights<TensorDataType>*>& send_weights,
                     std::vector<weights<TensorDataType>*>& recv_weights,
                     bool exchange_hyperparameters) {

  // Get partner process
  const El::Int rank_in_trainer = comm.get_rank_in_trainer();
  const El::Int procs_per_trainer = comm.get_procs_per_trainer();
  const El::Int partner_rank_in_world = (partner_trainer * procs_per_trainer
                                         + rank_in_trainer);

  // Exchange weights with partner
  for (size_t i = 0; i < send_weights.size(); ++i) {
    const auto& send = *send_weights[i];
    auto& recv = *recv_weights[i];
    if (weights_names.empty()
        || (std::find(weights_names.begin(), weights_names.end(),
                      send.get_name())
            != weights_names.end())) {

      // Exchange weights values
      El::SendRecv(send.get_values().LockedMatrix(),
                   recv.get_values().Matrix(),
                   comm.get_world_comm(),
                   partner_rank_in_world,
                   partner_rank_in_world);

      // Exchange optimizer state
      const auto* send_opt = send.get_optimizer();
      auto* recv_opt = recv.get_optimizer();
      const auto* send_sgd = dynamic_cast<const sgd<TensorDataType>*>(send_opt);
      auto* recv_sgd = dynamic_cast<sgd<TensorDataType>*>(recv_opt);
      if (send_sgd != nullptr && recv_sgd != nullptr) {
        if(exchange_hyperparameters) {
          using hyperparameters_type = std::tuple<TensorDataType, TensorDataType, bool>;
          hyperparameters_type hyperparameters(send_sgd->get_learning_rate(),
                                               send_sgd->get_momentum(),
                                               send_sgd->using_nesterov());
          El::mpi::SendRecv(reinterpret_cast<El::byte*>(&hyperparameters),
                            sizeof(hyperparameters_type),
                            partner_rank_in_world,
                            partner_rank_in_world,
                            comm.get_world_comm(),
                            El::SyncInfo<El::Device::CPU>{});
          recv_sgd->set_learning_rate(std::get<0>(hyperparameters));
          recv_sgd->set_momentum(std::get<1>(hyperparameters));
          recv_sgd->set_nesterov(std::get<2>(hyperparameters));
        }
        El::SendRecv(send_sgd->get_velocity().LockedMatrix(),
                     recv_sgd->get_velocity().Matrix(),
                     comm.get_world_comm(),
                     partner_rank_in_world,
                     partner_rank_in_world);
      }
      const auto* send_adam = dynamic_cast<const adam<TensorDataType>*>(send_opt);
      auto* recv_adam = dynamic_cast<adam<TensorDataType>*>(recv_opt);
      if (send_adam != nullptr && recv_adam != nullptr) {
        if(exchange_hyperparameters) {
          using hyperparameters_type = std::tuple<TensorDataType, TensorDataType, TensorDataType,
                                                  TensorDataType, TensorDataType, TensorDataType>;
          hyperparameters_type hyperparameters(send_adam->get_learning_rate(),
                                               send_adam->get_beta1(),
                                               send_adam->get_beta2(),
                                               send_adam->get_eps(),
                                               send_adam->get_current_beta1(),
                                               send_adam->get_current_beta2());
          El::mpi::SendRecv(reinterpret_cast<El::byte*>(&hyperparameters),
                            sizeof(hyperparameters_type),
                            partner_rank_in_world,
                            partner_rank_in_world,
                            comm.get_world_comm(),
                            El::SyncInfo<El::Device::CPU>{});
          recv_adam->set_learning_rate(std::get<0>(hyperparameters));
          recv_adam->set_beta1(std::get<1>(hyperparameters));
          recv_adam->set_beta2(std::get<2>(hyperparameters));
          recv_adam->set_eps(std::get<3>(hyperparameters));
          recv_adam->set_current_beta1(std::get<4>(hyperparameters));
          recv_adam->set_current_beta2(std::get<5>(hyperparameters));
          El::SendRecv(send_adam->get_moment1().LockedMatrix(),
                       recv_adam->get_moment1().Matrix(),
                       comm.get_world_comm(),
                       partner_rank_in_world,
                       partner_rank_in_world);
        }
        El::SendRecv(send_adam->get_moment2().LockedMatrix(),
                     recv_adam->get_moment2().Matrix(),
                     comm.get_world_comm(),
                     partner_rank_in_world,
                     partner_rank_in_world);
      }

    }
  }

}

} // namespace sendrecv_weights

/// See @c lbann::callbacks::ltfb::communication_algorithm::checkpoint_file
namespace checkpoint_file {

/** @param weights_names    Names of weights to exchange. If empty,
 *                          then all weights are exchanged.
 *  @param local_weight     Copies of weights. Used to restore weights
 *                          that we don't want to exchange.
 */
template <typename TensorDataType>
void exchange_models(lbann_comm& comm,
                     El::Int partner_trainer,
                     model& m,
                     const std::set<std::string>& weights_names,
                     const std::vector<weights<TensorDataType>*>& local_weights) {

  // Checkpoint directories
  const auto& c = m.get_execution_context();
  const auto local_trainer = comm.get_trainer_rank();
  const auto step = c.get_step();
  const std::string send_dir = (m.get_name()
                                + "_trainer" + std::to_string(local_trainer)
                                + "_step" + std::to_string(step));
  const std::string recv_dir = (m.get_name()
                                + "_trainer" + std::to_string(partner_trainer)
                                + "_step" + std::to_string(step));

  // Save model checkpoint
  persist p;
  p.set_cb_type(callback_type::model_only);
  if (comm.am_trainer_master()) {
    p.open_checkpoint(send_dir);
  } else {
    p.m_checkpoint_dir = send_dir;
  }
  m.save_to_checkpoint_shared(p);
  p.close_checkpoint();

  // Synchronize with partner trainer
  {
    const auto rank_in_trainer = comm.get_rank_in_trainer();
    TensorDataType send = false, recv = false;
    comm.sendrecv(&send, 1, partner_trainer, rank_in_trainer,
                  &recv, 1, partner_trainer, rank_in_trainer,
                  El::SyncInfo<El::Device::CPU>{});
  }

  // Load model checkpoint from partner trainer
  p.set_cb_type(callback_type::model_only);
  if (comm.am_trainer_master()) {
    p.open_restart(recv_dir);
  } else {
    p.m_checkpoint_dir = recv_dir;
  }
  m.load_from_checkpoint_shared(p);
  if (comm.am_trainer_master()) {
    p.close_restart();
  }

  // Restore weights that shouldn't be exchanged
  if (!weights_names.empty()) {
    const auto& model_weights = m.get_weights();
    for (size_t i = 0; i < model_weights.size(); ++i) {
      if (std::find(weights_names.begin(),
                    weights_names.end(),
                    model_weights[i]->get_name())
          == weights_names.end()) {
        *model_weights[i] = *local_weights[i];
      }
    }
  }

}

template <typename TensorDataType>
void restore_local_model(lbann_comm& comm, model& m) {

  // Checkpoint directories
  const auto& c = m.get_execution_context();
  const auto local_trainer = comm.get_trainer_rank();
  const auto step = c.get_step();
  const std::string checkpoint_dir = (m.get_name()
                                      + "_trainer" + std::to_string(local_trainer)
                                      + "_step" + std::to_string(step));

  // Load local model checkpoint
  persist p;
  p.set_cb_type(callback_type::model_only);
  if (comm.am_trainer_master()) {
    p.open_restart(checkpoint_dir);
  } else {
    p.m_checkpoint_dir = checkpoint_dir;
  }
  m.load_from_checkpoint_shared(p);
  if (comm.am_trainer_master()) {
    p.close_restart();
  }

}
} // namespace checkpoint_file

/** Get mean metric value with validation set. */
template <typename TensorDataType>
EvalType evaluate(model& m, const std::string& metric_name) {
  auto& c = m.get_execution_context();
  // Make sure data readers finish asynchronous work
  const auto original_mode = c.get_execution_mode();
  m.collect_background_data_fetch(original_mode);

  // Mark the data store as loading - Note that this is a temporary fix
  // for the current use of the tournament
  m.mark_data_store_explicitly_loading(execution_mode::validation);

  // Evaluate model on validation set
  c.get_trainer().evaluate(&m, execution_mode::validation);

  // Get metric value
  bool found_metric = false;
  EvalType metric_value = 0;
  for (const auto& met : m.get_metrics()) {
    if (met->name() == metric_name) {
      found_metric = true;
      metric_value = met->get_mean_value(execution_mode::validation);
      break;
    }
  }
  if (!found_metric) {
    LBANN_ERROR("could not find metric \"",metric_name,"\" ",
                "in model \"",m.get_name(),"\"");
  }

  // Mark the data store as loaded - Note that this is a temporary fix
  // for the current use of the tournament
  m.make_data_store_preloaded(execution_mode::validation);

  // Clean up and return metric value
  c.set_execution_mode(original_mode);
  return metric_value;

}

} // namespace <anon>

template <typename TensorDataType>
ltfb<TensorDataType>::ltfb(El::Int batch_interval,
           std::string metric_name,
           std::set<std::string> weights_names,
           bool low_score_wins,
           communication_algorithm comm_algo,
           bool exchange_hyperparameters)
  : data_type_callback<TensorDataType>(batch_interval),
    m_metric_name(std::move(metric_name)),
    m_weights_names(std::move(weights_names)),
    m_low_score_wins(low_score_wins),
    m_comm_algo(comm_algo),
    m_exchange_hyperparameters(exchange_hyperparameters) {}

template <typename TensorDataType>
ltfb<TensorDataType>::ltfb(const ltfb<TensorDataType>& other) :
  data_type_callback<TensorDataType>(other),
  m_metric_name(other.m_metric_name),
  m_weights_names(other.m_weights_names),
  m_low_score_wins(other.m_low_score_wins),
  m_comm_algo(other.m_comm_algo),
  m_exchange_hyperparameters(other.m_exchange_hyperparameters) {

  // Deep copy
  m_workspace_weights.clear();
  m_workspace_weights.reserve(other.m_workspace_weights.size());
  for (const auto& w : other.m_workspace_weights) {
    m_workspace_weights.emplace_back(w->copy());
  }

}

template <typename TensorDataType>
ltfb<TensorDataType>& ltfb<TensorDataType>::operator=(const ltfb<TensorDataType>& other) {
  data_type_callback<TensorDataType>::operator=(other);

  // Shallow copies
  m_metric_name = other.m_metric_name;
  m_weights_names = other.m_weights_names;
  m_low_score_wins = other.m_low_score_wins;
  m_comm_algo = other.m_comm_algo;
  m_exchange_hyperparameters = other.m_exchange_hyperparameters;

  // Deep copy
  m_workspace_weights.clear();
  m_workspace_weights.reserve(other.m_workspace_weights.size());
  for (const auto& w : other.m_workspace_weights) {
    m_workspace_weights.emplace_back(w->copy());
  }

  return *this;
}

template <typename TensorDataType>
void ltfb<TensorDataType>::setup(model *m) {

  // Create workspace objects
  const auto& model_weights = m->get_weights();
  m_workspace_weights.clear();
  m_workspace_weights.reserve(model_weights.size());
  for (const auto& w : model_weights) {
    m_workspace_weights.emplace_back(w->copy());
  }

  // Make sure model does not have inter-trainer communication callback
  for (auto&& cb : m->get_callbacks()) {
    if (dynamic_cast<imcomm<TensorDataType>*>(cb) != nullptr) {
      LBANN_ERROR("Detected both LTFB and imcomm callbacks. ");
    }
  }

}

template <typename TensorDataType>
void ltfb<TensorDataType>::on_train_begin(model *m) {
  auto&& comm = *m->get_comm();

  if (comm.am_world_master()) {
    std::cout << "starting synchronizing trainers...\n";
  }
  double tm1 = get_time();
  /// Make sure that all of the trainers are ready to go before starting
  comm.intertrainer_barrier();

  if (comm.am_world_master()) {
    std::cout << "synchronizing trainers... " << get_time()-tm1 <<"s\n";
  }
}

template <typename TensorDataType>
void ltfb<TensorDataType>::on_batch_begin(model *m) {
  const auto& c = m->get_execution_context();
  auto&& comm = *m->get_comm();

  // Check whether to start LTFB round
  const auto mode = c.get_execution_mode();
  const auto step = c.get_step();
  if (mode != execution_mode::training || step == 0) { return; }

  // Print message
  const auto message_prefix = (std::string{} + "LTFB ("
                               + "model \"" + m->get_name() + "\", "
                               + "step " + std::to_string(step)
                               + "): ");
  if (comm.am_world_master()) {
    std::cout << message_prefix + "starting tournament...\n";
  }

  // Determine partner model for tournament
  const El::Int local_trainer = comm.get_trainer_rank();
  const El::Int partner_trainer
    = get_partner_trainer<TensorDataType>(comm, message_prefix);

  // Evaluate local model
  if (comm.am_world_master()) {
    std::cout << message_prefix + "evaluating local model...\n";
  }
  const auto local_score = evaluate<TensorDataType>(*m, m_metric_name);

  // Store local model data
  auto&& model_weights = m->get_weights();
  std::vector<weights<TensorDataType>*> local_weights;
  for (size_t i = 0; i < model_weights.size(); ++i) {
    local_weights.push_back(m_workspace_weights[i].get());
    *local_weights[i] = *model_weights[i];
  }

  // Exchange model data with partner trainer
  if (comm.am_world_master()) {
    std::cout << message_prefix + "exchanging model data...\n";
  }
  switch (m_comm_algo) {
  case communication_algorithm::sendrecv_weights:
    sendrecv_weights::exchange_models<TensorDataType>(comm,
                                                      partner_trainer,
                                                      m_weights_names,
                                                      local_weights,
                                                      model_weights,
                                                      m_exchange_hyperparameters);
    break;
  case communication_algorithm::checkpoint_file:
    checkpoint_file::exchange_models<TensorDataType>(comm,
                                                     partner_trainer,
                                                     *m,
                                                     m_weights_names,
                                                     local_weights);
    break;
  default:
    LBANN_ERROR("invalid LTFB communication algorithm");
  }

  // Evaluate partner model
  if (comm.am_world_master()) {
    std::cout << message_prefix + "evaluating partner model...\n";
  }
  const auto& partner_score = evaluate(*m, m_metric_name);

  // Choose tournament winner
  // Note: restore local model data if it got a better score.
  El::Int tournament_winner = partner_trainer;
  if ((m_low_score_wins && local_score <= partner_score) ||
      (!m_low_score_wins && local_score >= partner_score)) {
    tournament_winner = local_trainer;
    switch (m_comm_algo) {
    case communication_algorithm::sendrecv_weights:
      for (size_t i = 0; i < model_weights.size(); ++i) {
        *model_weights[i] = *local_weights[i];
      }
      break;
    case communication_algorithm::checkpoint_file:
      checkpoint_file::restore_local_model<TensorDataType>(comm, *m);
      break;
    default:
      LBANN_ERROR("invalid LTFB communication algorithm");
    }
  }

  // Report tournament winner
  if (comm.am_trainer_master()) {
    std::stringstream msg;
    msg << message_prefix
        << "trainer " << local_trainer << " "
        << "selected model from trainer " << tournament_winner
        << " (trainer " << local_trainer << " score "
        << "= " << local_score << ", "
        << "trainer " << partner_trainer << " score "
        << "= " << partner_score << ")" << "\n";
    std::cout << msg.str();
  }

}

template <typename TensorDataType>
typename ltfb<TensorDataType>::communication_algorithm
ltfb<TensorDataType>::string_to_comm_algo(const std::string& str) {
  if (str.empty() || str == "sendrecv_weights") {
    return communication_algorithm::sendrecv_weights;
  }
  if (str == "checkpoint_file") {
    return communication_algorithm::checkpoint_file;
  }

  // Invalid LTFB communication algorithm
  LBANN_ERROR("invalid LTFB communication algorithm (",str,")");
  return communication_algorithm::sendrecv_weights;

}

template <typename TensorDataType>
std::unique_ptr<callback_base>
build_ltfb_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackLTFB&>(proto_msg);
  return make_unique<ltfb<TensorDataType>>(
    params.batch_interval(),
    params.metric(),
    parse_set<std::string>(params.weights()),
    params.low_score_wins(),
    ltfb<TensorDataType>::string_to_comm_algo(params.communication_algorithm()),
    params.exchange_hyperparameters());
}

} // namespace callback
} // namespace lbann
