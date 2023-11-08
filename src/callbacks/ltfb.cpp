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

#include "lbann/callbacks/ltfb.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/optimizers/adam.hpp"
#include "lbann/optimizers/sgd.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/random_number_generators.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lbann {
namespace callback {

/** @class LTFBCommunicationAlgorithm
 *  @brief Exchange model information with partners.
 */
class LTFBCommunicationAlgorithm
  : public Cloneable<HasAbstractFunction<LTFBCommunicationAlgorithm>>
{
public:
  /** @brief Default constructor.
   *
   *  All weights within a model are exchanged.
   */
  LTFBCommunicationAlgorithm() = default;

  /** @brief Construct with names of weights
   *  @param[in] weights_names Names of weights to exchange. If empty,
   *                           then all weights are exchanged.
   */
  LTFBCommunicationAlgorithm(std::set<std::string> const& weights_names)
    : weights_names_{weights_names}
  {}

  /** @brief Construct with names of weights
   *  @param[in] weights_names Names of weights to exchange. If empty,
   *                           then all weights are exchanged.
   */
  LTFBCommunicationAlgorithm(std::set<std::string>&& weights_names)
    : weights_names_{std::move(weights_names)}
  {}

  /** @brief Add communication algorithm data to prototext */
  virtual void
  write_comm_algo_proto(lbann_data::Callback_CallbackLTFB& msg) const = 0;

  virtual ~LTFBCommunicationAlgorithm() noexcept = default;

  /** @brief Exchange data in model with a partner.
   *
   *  @param[in,out] m Model to send to partner trainer. On output it
   *                   will be overwritten with model recieved from
   *                   partner trainer.
   *  @param[in] partner_trainer Index of the partner trainer with
   *                             which to exchange models.
   *  @param[in] step The step ID at which the exchange occurs.
   */
  virtual void
  exchange_models(model& m, El::Int partner_trainer, El::Int step) const = 0;

protected:
  LTFBCommunicationAlgorithm(LTFBCommunicationAlgorithm const&) = default;
  LTFBCommunicationAlgorithm(LTFBCommunicationAlgorithm&&) = default;
  /** @name Data access and query */
  ///@{
  auto weights_names() const noexcept -> std::set<std::string> const&
  {
    return weights_names_;
  }
  bool has_weights_names() const noexcept { return !weights_names_.empty(); }
  ///@}
private:
  std::set<std::string> weights_names_;
}; // class LTFBCommunicationAlgorithm

namespace {

/** @brief Inter-trainer communication scheme for LTFB.
 *
 *  The specifics of these algorithms are experimental and will be
 *  in flux.
 */
enum class comm_algorithm
{
  /** @brief Directly exchange weights values with sendrecv.
   *
   *  Corresponding ranks in partner trainers will iterate through
   *  their weights and exchange values with sendrecvs.
   *
   *  Notes:
   *    - Requires all models to be identical aside from their
   *      weights values, so this is not suitable for hyperparameter
   *      or model architecture exploration.
   *    - Optimizer state is not exchanged, so there may be wonky
   *      learning behavior immediately after a tournament.
   *    - Optimal if communication performance between ranks is
   *      uniform and independent. If intra-trainer communication is
   *      fast or if communication performance is sensitive to
   *      network traffic, it may be advantageous to gather model
   *      data on the trainer master ranks and only perform
   *      inter-trainer communication between them.
   */
  sendrecv_weights,

  /** @brief Save and load model data with checkpoint files.
   *
   *  Notes:
   *    - Supports hyperparameter exploration.
   *    - This approach is temporary and experimental, since going
   *      through the file system is very suboptimal. When a wire
   *      format for model checkpoints is developed, it should be
   *      used instead.
   */
  checkpoint_file,

  /** @brief Transfer model on the wire.
   *
   *  The entire model is saved, as though doing a
   *  checkpoint. Instead of writing to disk, a binary stream is
   *  used. After gathering the binary stream, it is
   *  exchanged. There is an implicit assumption that the
   *  participating processes are binary-compatible (e.g., for
   *  issues like endianness).
   */
  checkpoint_binary,
}; // enum comm_algorithm
comm_algorithm string_to_comm_algo(const std::string& str)
{
  if (str.empty() || str == "sendrecv_weights") {
    return comm_algorithm::sendrecv_weights;
  }
  else if (str == "checkpoint_file") {
    return comm_algorithm::checkpoint_file;
  }
  else if (str == "checkpoint_binary") {
    return comm_algorithm::checkpoint_binary;
  }
  // Invalid LTFB communication algorithm
  LBANN_ERROR("invalid LTFB communication algorithm (", str, ")");
  return comm_algorithm::sendrecv_weights;
}

/** @brief Generate partner trainer assignments.
 *
 *  Requires a scatter from the world master process. If there are an
 *  odd number of trainers, one of them is partnered with itself.
 */
El::Int get_partner_trainer(lbann_comm& comm, const std::string& message_prefix)
{
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
    auto const& arg_parser = global_argument_parser();
    bool const ltfb_verbose = arg_parser.get<bool>(LBANN_OPTION_LTFB_VERBOSE);
    bool skipped_reporting_trainers = false;
    // Print partner assignments to standard output
    std::ostringstream msg;
    msg << message_prefix << "tournament partners -";
    for (El::Int i = 0; i < num_trainers; i += 2) {
      // By default only print out 3 pairs of trainer mappings unless
      // LTFB has verbose reporting
      if (i < 3 || i == (num_trainers - 2) || i == (num_trainers - 1) ||
          ltfb_verbose) {
        msg << (i > 0 && !skipped_reporting_trainers ? "," : "") << " {"
            << trainers[i];
        if (i + 1 < num_trainers) {
          msg << "," << trainers[i + 1];
        }
        msg << "}";
      }
      else if (!skipped_reporting_trainers) {
        msg << " ...";
        skipped_reporting_trainers = true;
      }
    }
    msg << "\n";
    std::cout << msg.str() << std::endl;
  }

  // Setup partner assignments for all processes
  std::vector<El::Int> send_buffer(num_trainers);
  for (El::Int i = 0; i < num_trainers; i += 2) {
    const auto& trainer1 = trainers[i];
    const auto& trainer2 = (i + 1 < num_trainers) ? trainers[i + 1] : trainer1;
    send_buffer[trainer1] = trainer2;
    send_buffer[trainer2] = trainer1;
  }
  return send_buffer[comm.get_trainer_rank()];
}

void restore_model_weights(
  model& m,
  std::unordered_map<std::string, std::unique_ptr<weights>>& restore_weights)
{
  // Restore weights that shouldn't be exchanged
  if (restore_weights.empty())
    return;

  // FIXME: Generalize this; enable ptr move??
  for (auto w : m.get_weights()) {
    if (restore_weights.count(w->get_name()) > 0) {
      using TensorDataType = DataType;
      using WeightsType = data_type_weights<TensorDataType>;
      dynamic_cast<WeightsType&>(*w) =
        dynamic_cast<WeightsType&>(*restore_weights[w->get_name()]);
    }
  }
}

std::string sendrecv_string(lbann_comm const& c,
                            std::string const& src,
                            El::Int partner_trainer)
{
#ifdef LBANN_HAS_ALUMINUM
  El::mpi::EnsureComm<size_t, El::Collective::SENDRECV>(
    c.get_world_comm(),
    El::SyncInfo<El::Device::CPU>{});
#endif

  if (!c.am_trainer_master())
    return "";

  // Exchange sizes
  size_t my_size = src.size();
  size_t other_size = src.max_size() + 1;
  c.sendrecv(&my_size,
             1,
             partner_trainer,
             0,
             &other_size,
             1,
             partner_trainer,
             0,
             El::SyncInfo<El::Device::CPU>{});

  // Exchange strings
  std::string tgt(other_size, '\0');

  auto const* send_buf = reinterpret_cast<El::byte const*>(src.data());
  auto* recv_buf = reinterpret_cast<El::byte*>(tgt.data());

  // Get the max blk size
  int constexpr max_blk_size_int = std::numeric_limits<int>::max();
  std::size_t constexpr max_blk_size_size_t = max_blk_size_int;

  while (my_size || other_size) {
    int const this_blk_send_size =
      (my_size > max_blk_size_size_t ? max_blk_size_int : my_size);
    int const this_blk_recv_size =
      (other_size > max_blk_size_size_t ? max_blk_size_int : other_size);

    c.sendrecv(send_buf,
               this_blk_send_size,
               partner_trainer,
               0,
               recv_buf,
               this_blk_recv_size,
               partner_trainer,
               0,
               El::SyncInfo<El::Device::CPU>{});

    send_buf += this_blk_send_size;
    recv_buf += this_blk_recv_size;
    my_size =
      (my_size > max_blk_size_size_t ? my_size - max_blk_size_size_t : 0);
    other_size =
      (other_size > max_blk_size_size_t ? other_size - max_blk_size_size_t : 0);
  }
  return tgt;
}

template <typename T>
void exchange(lbann_comm const& c, T& object, El::Int partner_trainer)
{
  std::ostringstream oss;
  {
    RootedBinaryOutputArchive ar(oss, c.get_trainer_grid());
    c.trainer_barrier();
    ar(object);
  }
  c.trainer_barrier(); // I don't think this is necessary
  {
    std::istringstream iss{sendrecv_string(c, oss.str(), partner_trainer)};
    RootedBinaryInputArchive ar(iss, c.get_trainer_grid());
    ar(object);
  }
  c.trainer_barrier(); // I don't think this is necessary either
}

/** @class SendRecvWeights
 *  @brief Exchange model weights directly using sendrecvs.
 *  @todo More general approach to exchange optimizer state. Currently
 *  only SGD and Adam are supported.
 */
class SendRecvWeights final
  : public Cloneable<SendRecvWeights, LTFBCommunicationAlgorithm>
{
  using BaseType = Cloneable<SendRecvWeights, LTFBCommunicationAlgorithm>;

public:
  /** @brief Construct from weights names
   *  @param[in] weights_names Names of weights to exchange. If empty,
   *                           then all weights are exchanged.
   *  @param[in] exchange_hyperparameters Exchange optimizer
   *                                      hyperparameters.
   */
  SendRecvWeights(std::set<std::string> const& weights_names,
                  bool exchange_hyperparameters)
    : BaseType(weights_names), exchange_hyperparams_{exchange_hyperparameters}
  {}

  /** @brief Construct from weights names
   *  @param[in] weights_names Names of weights to exchange. If empty,
   *                           then all weights are exchanged.
   *  @param[in] exchange_hyperparameters Exchange optimizer
   *                                      hyperparameters.
   */
  SendRecvWeights(std::set<std::string>&& weights_names,
                  bool exchange_hyperparameters)
    : BaseType(std::move(weights_names)),
      exchange_hyperparams_{exchange_hyperparameters}
  {}

  SendRecvWeights(SendRecvWeights const&) = default;
  SendRecvWeights(SendRecvWeights&&) = default;

  virtual void
  write_comm_algo_proto(lbann_data::Callback_CallbackLTFB& msg) const
  {
    msg.set_weights(protobuf::to_space_sep_string(this->weights_names()));
    msg.set_communication_algorithm("sendrecv_weights");
    msg.set_exchange_hyperparameters(exchange_hyperparams_);

    // Not used for sendrecv_weights
    // msg.set_checkpoint_basedir("");
  }

  /** @todo This function is way too long. */
  void exchange_models(model& m,
                       El::Int partner_trainer,
                       El::Int /*step*/) const final
  {
    auto&& comm = *m.get_comm();

    // Get partner process
    const El::Int rank_in_trainer = comm.get_rank_in_trainer();
    const El::Int procs_per_trainer = comm.get_procs_per_trainer();
    const El::Int partner_rank_in_world =
      (partner_trainer * procs_per_trainer + rank_in_trainer);

    // Exchange weights with partner
    for (auto&& w_ptr : m.get_weights()) {
      // Skip weights if name isn't in list
      auto const& weights_names = this->weights_names();
      if (this->has_weights_names() &&
          (weights_names.find(w_ptr->get_name()) == weights_names.cend())) {
        continue;
      }

      // Exchange weights values
      using TensorDataType = DataType;
      using WeightsType = data_type_weights<TensorDataType>;
      auto& recv_weights = dynamic_cast<WeightsType&>(*w_ptr);
      auto send_weights = recv_weights;
      El::SendRecv(send_weights.get_values_sharded().LockedMatrix(),
                   recv_weights.get_values_sharded().Matrix(),
                   comm.get_world_comm(),
                   partner_rank_in_world,
                   partner_rank_in_world);

      // If the two weights objects use different optimizers across
      // the set of trainers, we need to be careful about how we
      // exchange the data.

      // Skip if there is no optimizer.
      // FIXME (trb 04/14/2021): Could we hit a situation in which the
      // weights in Trainer I has an optimizer and the corresponding
      // weights for Trainer J does not?
      optimizer* send_opt = send_weights.get_optimizer();
      if (!send_opt)
        continue;

      bool const do_binary_exchange =
        exchange_hyperparams_ ||
        !have_same_optimizer_type(comm, *send_opt, partner_trainer);

      if (do_binary_exchange) {
        // Since we cannot get at the unique pointer directly, we make
        // a copy:
        auto opt_up = send_opt->clone();
        exchange(comm, opt_up, partner_trainer);
        opt_up->setup(&recv_weights);
        recv_weights.set_optimizer(std::move(opt_up));
      }
      else {
        // Exchange SGD optimizer state
        using SGDType = sgd<TensorDataType>;
        auto* send_sgd = dynamic_cast<SGDType*>(send_weights.get_optimizer());
        auto* recv_sgd = dynamic_cast<SGDType*>(recv_weights.get_optimizer());
        if (send_sgd != nullptr && recv_sgd != nullptr) {
          El::SendRecv(send_sgd->get_velocity().LockedMatrix(),
                       recv_sgd->get_velocity().Matrix(),
                       comm.get_world_comm(),
                       partner_rank_in_world,
                       partner_rank_in_world);
          continue;
        }

        // Exchange Adam optimizer state
        using AdamType = adam<TensorDataType>;
        auto* send_adam = dynamic_cast<AdamType*>(send_weights.get_optimizer());
        auto* recv_adam = dynamic_cast<AdamType*>(recv_weights.get_optimizer());
        if (send_adam != nullptr && recv_adam != nullptr) {
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
          continue;
        }
        LBANN_WARNING("Unknown optimizer type. NO EXCHANGE.");
      }
    }
  }

private:
  bool have_same_optimizer_type(lbann_comm const& c,
                                optimizer const& opt,
                                El::Int partner_trainer) const noexcept
  {
    std::size_t const my_type_hash = typeid(opt).hash_code();
    std::size_t other_type_hash = -1;
    c.sendrecv(&my_type_hash,
               1,
               partner_trainer,
               0,
               &other_type_hash,
               1,
               partner_trainer,
               0,
               El::SyncInfo<El::Device::CPU>{});
    return my_type_hash == other_type_hash;
  }

private:
  bool exchange_hyperparams_;
}; // class SendRecvWeights

/// See @c lbann::callbacks::ltfb::communication_algorithm::checkpoint_file
class CheckpointFile final
  : public Cloneable<CheckpointFile, LTFBCommunicationAlgorithm>
{
  using BaseType = Cloneable<CheckpointFile, LTFBCommunicationAlgorithm>;

public:
  CheckpointFile(std::set<std::string> const& weights_names,
                 std::string const& ckpt_basedir)
    : BaseType(weights_names), ckpt_basedir_{ckpt_basedir}
  {}

  CheckpointFile(std::set<std::string>&& weights_names,
                 std::string const& ckpt_basedir)
    : BaseType(std::move(weights_names)), ckpt_basedir_{ckpt_basedir}
  {}

  virtual void
  write_comm_algo_proto(lbann_data::Callback_CallbackLTFB& msg) const
  {
    msg.set_weights(protobuf::to_space_sep_string(this->weights_names()));
    msg.set_communication_algorithm("checkpoint_file");
    msg.set_checkpoint_basedir(ckpt_basedir_);

    // Not used in checkpoint_file
    // msg.set_exchange_hyperparameters(bool_value);
  }

  void
  exchange_models(model& m, El::Int partner_trainer, El::Int step) const final
  {
    auto&& comm = *m.get_comm();
    // Keep track of weights that shouldn't be exchanged
    std::unordered_map<std::string, std::unique_ptr<weights>> restore_weights;
    if (this->has_weights_names()) {
      auto const& weights_names = this->weights_names();
      for (auto w : m.get_weights()) {
        if (weights_names.find(w->get_name()) == weights_names.cend()) {
          using TensorDataType = DataType;
          using WeightsType = data_type_weights<TensorDataType>;
          restore_weights[w->get_name()] =
            std::make_unique<WeightsType>(dynamic_cast<WeightsType&>(*w));
        }
      }
    }

    // Checkpoint directories
    const auto basedir =
      (ckpt_basedir_.empty() ? std::string("") : add_delimiter(ckpt_basedir_));
    const auto local_trainer = comm.get_trainer_rank();
    const std::string send_dir =
      (basedir + m.get_name() + "_trainer" + std::to_string(local_trainer) +
       "_step" + std::to_string(step));
    const std::string recv_dir =
      (basedir + m.get_name() + "_trainer" + std::to_string(partner_trainer) +
       "_step" + std::to_string(step));

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
      comm.sendrecv(&send,
                    1,
                    partner_trainer,
                    0,
                    &recv,
                    1,
                    partner_trainer,
                    0,
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

    /// @todo Should be unneeded, but we experience hangs without it
    comm.trainer_barrier();

    restore_model_weights(m, restore_weights);
  }

private:
  std::string ckpt_basedir_;
}; // class CheckpointFile

class CheckpointBinary final
  : public Cloneable<CheckpointBinary, LTFBCommunicationAlgorithm>
{
  using BaseType = Cloneable<CheckpointBinary, LTFBCommunicationAlgorithm>;

public:
  CheckpointBinary(std::set<std::string> const& weights_names)
    : BaseType(weights_names)
  {}
  CheckpointBinary(std::set<std::string>&& weights_names)
    : BaseType(std::move(weights_names))
  {}
  void exchange_models(model& m,
                       El::Int partner_trainer,
                       El::Int /*step*/) const final
  {
    auto&& comm = *m.get_comm();
    // Keep track of weights that shouldn't be exchanged
    std::unordered_map<std::string, std::unique_ptr<weights>> restore_weights;
    if (this->has_weights_names()) {
      auto const& weights_names = this->weights_names();
      for (auto w : m.get_weights()) {
        if (weights_names.find(w->get_name()) == weights_names.cend()) {
          using TensorDataType = DataType;
          using WeightsType = data_type_weights<TensorDataType>;
          restore_weights[w->get_name()] =
            std::make_unique<WeightsType>(dynamic_cast<WeightsType&>(*w));
        }
      }
    }
    exchange(comm, m, partner_trainer);
    restore_model_weights(m, restore_weights);
  }

  virtual void
  write_comm_algo_proto(lbann_data::Callback_CallbackLTFB& msg) const
  {
    msg.set_weights(protobuf::to_space_sep_string(this->weights_names()));
    msg.set_communication_algorithm("checkpoint_binary");

    // Not used in checkpoint_binary
    // msg.set_exchange_hyperparameters(bool_value);
    // msg.set_checkpoint_basedir("");
  }

}; // class CheckpointBinary

/** Get mean metric value with validation set. */
EvalType evaluate(model& m, const std::string& metric_name)
{
  auto& c = m.get_execution_context();
  // Make sure data readers finish asynchronous work
  const auto original_mode = c.get_execution_mode();
  data_coordinator& dc = get_trainer().get_data_coordinator();
  dc.collect_background_data_fetch(original_mode);

  if (!dc.is_execution_mode_valid(execution_mode::tournament)) {
    LBANN_ERROR("LTFB requires ",
                to_string(execution_mode::tournament),
                " execution mode");
  }
  // Mark the data store as loading - Note that this is a temporary fix
  // for the current use of the tournament
  dc.mark_data_store_explicitly_loading(execution_mode::tournament);

  // Evaluate model on validation set
  get_trainer().evaluate(&m, execution_mode::tournament);

  // Get metric value
  bool found_metric = false;
  EvalType metric_value = 0;
  for (const auto& met : m.get_metrics()) {
    if (met->name() == metric_name) {
      found_metric = true;
      metric_value = met->get_mean_value(execution_mode::tournament);
      break;
    }
  }
  if (!found_metric) {
    LBANN_ERROR("could not find metric \"",
                metric_name,
                "\" ",
                "in model \"",
                m.get_name(),
                "\"");
  }

  // Mark the data store as loaded - Note that this is a temporary fix
  // for the current use of the tournament
  dc.make_data_store_preloaded(execution_mode::tournament);

  // Clean up and return metric value
  m.reset_mode(c, original_mode);
  dc.reset_mode(c);
  return metric_value;
}

} // namespace

ltfb::ltfb(El::Int batch_interval,
           std::string metric_name,
           std::unique_ptr<LTFBCommunicationAlgorithm> algo,
           bool low_score_wins)
  : callback_base(batch_interval),
    m_metric_name{std::move(metric_name)},
    comm_algo_{std::move(algo)},
    m_low_score_wins{low_score_wins}
{}

ltfb::ltfb(const ltfb& other)
  : callback_base(other),
    m_metric_name(other.m_metric_name),
    comm_algo_(other.comm_algo_->clone()),
    m_low_score_wins(other.m_low_score_wins)
{}

ltfb& ltfb::operator=(const ltfb& other)
{
  callback_base::operator=(other);
  m_metric_name = other.m_metric_name;
  comm_algo_ = other.comm_algo_->clone();
  m_low_score_wins = other.m_low_score_wins;
  return *this;
}

void ltfb::on_train_begin(model* m)
{
  auto&& comm = *m->get_comm();

  if (comm.am_world_master()) {
    std::cout << "starting synchronizing trainers...\n";
  }
  double tm1 = get_time();
  /// Make sure that all of the trainers are ready to go before starting
  comm.intertrainer_barrier();

  if (comm.am_world_master()) {
    std::cout << "synchronizing trainers... " << get_time() - tm1 << "s\n";
  }
}

void ltfb::on_batch_begin(model* m)
{
  auto& local_model = *m;
  auto& context = local_model.get_execution_context();
  auto&& comm = *local_model.get_comm();

  // Check whether to start LTFB round
  const auto mode = context.get_execution_mode();
  const auto step = context.get_step();
  if (mode != execution_mode::training || step == 0) {
    return;
  }

  // Print message
  const auto message_prefix =
    (std::string{} + "LTFB (" + "model \"" + local_model.get_name() + "\", " +
     "step " + std::to_string(step) + "): ");
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
  auto local_score = evaluate(local_model, m_metric_name);

  // Get model from partner trainer
  if (comm.am_world_master()) {
    std::cout << message_prefix + "exchanging model data...\n";
  }

  model partner_model(local_model);
  if (comm_algo_)
    comm_algo_->exchange_models(partner_model, partner_trainer, step);
  else
    LBANN_ERROR("No communication algorithm.");

  // Evaluate partner model
  if (comm.am_world_master()) {
    std::cout << message_prefix + "evaluating partner model...\n";
  }
  auto partner_score = evaluate(partner_model, m_metric_name);

  // Choose tournament winner
  // Note: restore local model data if it got a better score.
  El::Int tournament_winner = local_trainer;
  if ((m_low_score_wins && partner_score <= local_score) ||
      (!m_low_score_wins && partner_score >= local_score) ||
      (!std::isfinite(local_score) && std::isfinite(partner_score))) {
    tournament_winner = partner_trainer;

    /// @todo Use move assignment operator once LTFB is moved into a
    /// training algorithm
    local_model.swap_layers(partner_model);
    local_model.swap_weights(partner_model);
    local_model.swap_metrics(partner_model);
    local_model.swap_objective_function(partner_model);
    auto& trainer_ = get_trainer();
    local_model.setup(trainer_.get_max_mini_batch_size(),
                      trainer_.get_grids(),
                      true);
  }

  // Report tournament winner
  if (comm.am_trainer_master()) {
    std::ostringstream msg;
    msg << message_prefix << "trainer " << local_trainer << " "
        << "selected model from trainer " << tournament_winner << " (trainer "
        << local_trainer << " score "
        << "= " << local_score << ", "
        << "trainer " << partner_trainer << " score "
        << "= " << partner_score << ")"
        << "\n";
    std::cout << msg.str() << std::flush;
  }
}

void ltfb::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_ltfb();
  msg->set_batch_interval(m_batch_interval);
  msg->set_metric(m_metric_name);
  msg->set_low_score_wins(m_low_score_wins);
  comm_algo_->write_comm_algo_proto(*msg);
}
std::unique_ptr<callback_base>
build_ltfb_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                              const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackLTFB&>(proto_msg);
  auto weights_list = parse_set<std::string>(params.weights());
  std::unique_ptr<LTFBCommunicationAlgorithm> algo;
  switch (string_to_comm_algo(params.communication_algorithm())) {
  case comm_algorithm::sendrecv_weights:
    algo = std::make_unique<SendRecvWeights>(std::move(weights_list),
                                             params.exchange_hyperparameters());
    break;
  case comm_algorithm::checkpoint_file:
    algo = std::make_unique<CheckpointFile>(std::move(weights_list),
                                            params.checkpoint_basedir());
    break;
  case comm_algorithm::checkpoint_binary:
    algo = std::make_unique<CheckpointBinary>(std::move(weights_list));
    break;
  }
  return std::make_unique<ltfb>(params.batch_interval(),
                                params.metric(),
                                std::move(algo),
                                params.low_score_wins());
}

} // namespace callback
} // namespace lbann
