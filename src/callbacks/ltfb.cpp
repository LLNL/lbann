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
#include "lbann/utils/random_number_generators.hpp"
#include "lbann/optimizers/sgd.hpp"
#include "lbann/optimizers/adam.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/weights/data_type_weights.hpp"

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
El::Int get_partner_trainer(lbann_comm& comm,
                            const std::string& message_prefix) {
    // Assign partner trainers
    // Note: The first trainer in 'trainers' is paired with the
    // second, the third with the fourth, and so on. If there are an
    // odd number of trainers, the last one is partnered with itself.
    const El::Int num_trainers = comm.get_num_trainers();
    std::vector<El::Int> trainers(num_trainers);
    std::iota(trainers.begin(), trainers.end(), 0);
    // Everyone use a special RNG that is only for LTFB so that they
    // can all communicate the same pairs without communication
    std::shuffle(trainers.begin(), trainers.end(), get_ltfb_generator());

    if (comm.am_world_master()) { // Root process
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
      std::cout << msg.str() << std::endl << std::flush;
    }

    // Setup partner assignments for all processes
    std::vector<El::Int> send_buffer(num_trainers);
    for (El::Int i = 0; i < num_trainers; i += 2) {
      const auto& trainer1 = trainers[i];
      const auto& trainer2 = (i+1 < num_trainers) ? trainers[i+1] : trainer1;
      send_buffer[trainer1] = trainer2;
      send_buffer[trainer2] = trainer1;
    }
    return send_buffer[comm.get_trainer_rank()];
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
                     const std::vector<data_type_weights<TensorDataType>*>& send_weights,
                     std::vector<data_type_weights<TensorDataType>*>& recv_weights,
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
        }
        El::SendRecv(send_adam->get_moment1().LockedMatrix(),
                     recv_adam->get_moment1().Matrix(),
                     comm.get_world_comm(),
                     partner_rank_in_world,
                     partner_rank_in_world);
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
                     El::Int step,
                     const std::set<std::string>& weights_names,
                     const std::vector<data_type_weights<TensorDataType>*>& local_weights,
                     const std::string& ckpt_basedir) {

  // Checkpoint directories
  const auto basedir = (ckpt_basedir.empty()?
                          std::string("") :
                          add_delimiter(ckpt_basedir));
  const auto local_trainer = comm.get_trainer_rank();
  const std::string send_dir = (basedir
                                + m.get_name()
                                + "_trainer" + std::to_string(local_trainer)
                                + "_step" + std::to_string(step));
  const std::string recv_dir = (basedir
                                + m.get_name()
                                + "_trainer" + std::to_string(partner_trainer)
                                + "_step" + std::to_string(step));

  // Save model checkpoint
  {
    persist p;
    p.set_cb_type(callback_type::model_only);
    p.open_checkpoint(send_dir, comm.am_trainer_master());
    comm.trainer_barrier();
    m.save_to_checkpoint_shared(p);
    p.close_checkpoint();
  }

  // Synchronize with partner trainer
  comm.trainer_barrier();
  if (comm.am_trainer_master()) {
    int send{0}, recv{0};
    comm.sendrecv(&send, 1, partner_trainer, 0,
                  &recv, 1, partner_trainer, 0,
                  El::SyncInfo<El::Device::CPU>{});
  }
  comm.trainer_barrier();

  // Load model checkpoint from partner trainer
  {
    persist p;
    p.set_cb_type(callback_type::model_only);
    p.open_restart(recv_dir);
    m.load_from_checkpoint_shared(p);
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
        using dtw_type = data_type_weights<TensorDataType>;
        dynamic_cast<dtw_type&>(*model_weights[i]) = *local_weights[i];
      }
    }
  }

}

void restore_local_model(lbann_comm& comm,
                         model& m,
                         El::Int step,
                         const std::string& ckpt_basedir) {

  // Checkpoint directories
  const auto basedir = (ckpt_basedir.empty()?
                          std::string("") :
                          add_delimiter(ckpt_basedir));
  const auto local_trainer = comm.get_trainer_rank();
  const std::string checkpoint_dir = (basedir
                                      + m.get_name()
                                      + "_trainer" + std::to_string(local_trainer)
                                      + "_step" + std::to_string(step));

  // Load local model checkpoint
  persist p;
  p.set_cb_type(callback_type::model_only);
  p.open_restart(checkpoint_dir);
  m.load_from_checkpoint_shared(p);
  p.close_restart();

}

} // namespace checkpoint_file

/** Get mean metric value with validation set. */
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
  m.reset_mode(c, original_mode);
  c.get_trainer().get_data_coordinator().reset_mode(c);
  return metric_value;

}

} // namespace <anon>

ltfb::ltfb(El::Int batch_interval,
           std::string metric_name,
           std::set<std::string> weights_names,
           bool low_score_wins,
           communication_algorithm comm_algo,
           const std::string& ckpt_basedir,
           bool exchange_hyperparameters)
  : callback_base(batch_interval),
    m_metric_name(std::move(metric_name)),
    m_weights_names(std::move(weights_names)),
    m_low_score_wins(low_score_wins),
    m_comm_algo(comm_algo),
    m_ckpt_basedir(ckpt_basedir),
    m_exchange_hyperparameters(exchange_hyperparameters) {}

ltfb::ltfb(const ltfb& other) :
  callback_base(other),
  m_metric_name(other.m_metric_name),
  m_weights_names(other.m_weights_names),
  m_low_score_wins(other.m_low_score_wins),
  m_comm_algo(other.m_comm_algo),
  m_ckpt_basedir(other.m_ckpt_basedir),
  m_exchange_hyperparameters(other.m_exchange_hyperparameters) {

  // Deep copy
  m_workspace_weights.clear();
  m_workspace_weights.reserve(other.m_workspace_weights.size());
  for (const auto& w : other.m_workspace_weights) {
    m_workspace_weights.emplace_back(w->clone());
  }

}

ltfb& ltfb::operator=(const ltfb& other) {
  callback_base::operator=(other);

  // Shallow copies
  m_metric_name = other.m_metric_name;
  m_weights_names = other.m_weights_names;
  m_low_score_wins = other.m_low_score_wins;
  m_comm_algo = other.m_comm_algo;
  m_ckpt_basedir = other.m_ckpt_basedir;
  m_exchange_hyperparameters = other.m_exchange_hyperparameters;

  // Deep copy
  m_workspace_weights.clear();
  m_workspace_weights.reserve(other.m_workspace_weights.size());
  for (const auto& w : other.m_workspace_weights) {
    m_workspace_weights.emplace_back(w->clone());
  }

  return *this;
}

void ltfb::setup(model *m) {

  // Create workspace objects
  const auto& model_weights = m->get_weights();
  m_workspace_weights.clear();
  m_workspace_weights.reserve(model_weights.size());
  for (const auto& w : model_weights) {
    m_workspace_weights.emplace_back(w->clone());
  }

  // Make sure model does not have inter-trainer communication callback
  for (auto&& cb : m->get_callbacks()) {
    if (dynamic_cast<imcomm*>(cb) != nullptr) {
      LBANN_ERROR("Detected both LTFB and imcomm callbacks. ");
    }
  }

}

void ltfb::on_train_begin(model *m) {
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

void ltfb::on_batch_begin(model *m) {
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
  const El::Int partner_trainer = get_partner_trainer(comm, message_prefix);

  // Evaluate local model
  if (comm.am_world_master()) {
    std::cout << message_prefix + "evaluating local model...\n";
  }
  const auto local_score = evaluate(*m, m_metric_name);

  // Store local model data
  auto&& model_weights_tmp = m->get_weights();
  std::vector<data_type_weights<DataType>*> local_weights, model_weights;
  local_weights.reserve(model_weights_tmp.size());
  model_weights.reserve(model_weights_tmp.size());
  for (size_t i = 0; i < model_weights_tmp.size(); ++i) {
    auto* wsp = dynamic_cast<data_type_weights<DataType>*>(
      m_workspace_weights[i].get());
    auto* mlw = dynamic_cast<data_type_weights<DataType>*>(
      model_weights_tmp[i]);
    if (!wsp || !mlw)
      LBANN_ERROR("Detected bad weights");
    local_weights.push_back(wsp);
    model_weights.push_back(mlw);
    *local_weights.back() = *model_weights.back();
  }

  // Exchange model data with partner trainer
  if (comm.am_world_master()) {
    std::cout << message_prefix + "exchanging model data...\n";
  }
  switch (m_comm_algo) {
  case communication_algorithm::sendrecv_weights:
    sendrecv_weights::exchange_models(comm,
                                      partner_trainer,
                                      m_weights_names,
                                      local_weights,
                                      model_weights,
                                      m_exchange_hyperparameters);
    break;
  case communication_algorithm::checkpoint_file:
    checkpoint_file::exchange_models(comm,
                                     partner_trainer,
                                     *m,
                                     step,
                                     m_weights_names,
                                     local_weights,
                                     m_ckpt_basedir);
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
  if ((m_low_score_wins && local_score <= partner_score)
      || (!m_low_score_wins && local_score >= partner_score)
      || (!std::isnan(local_score) && std::isnan(partner_score))) {
    tournament_winner = local_trainer;
    switch (m_comm_algo) {
    case communication_algorithm::sendrecv_weights:
      for (size_t i = 0; i < model_weights.size(); ++i) {
        *model_weights[i] = *local_weights[i];
      }
      break;
    case communication_algorithm::checkpoint_file:
      checkpoint_file::restore_local_model(comm, *m, step, m_ckpt_basedir);
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
    std::cout << msg.str() << std::flush;
  }
}

typename ltfb::communication_algorithm
ltfb::string_to_comm_algo(const std::string& str) {
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

void ltfb::set_ckpt_basedir(const std::string& dir) {
  m_ckpt_basedir = dir;
}

std::string ltfb::get_ckpt_basedir() const {
  return m_ckpt_basedir;
}

std::unique_ptr<callback_base>
build_ltfb_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackLTFB&>(proto_msg);
  return make_unique<ltfb>(
    params.batch_interval(),
    params.metric(),
    parse_set<std::string>(params.weights()),
    params.low_score_wins(),
    ltfb::string_to_comm_algo(params.communication_algorithm()),
    params.checkpoint_basedir(),
    params.exchange_hyperparameters());
}

} // namespace callback
} // namespace lbann
