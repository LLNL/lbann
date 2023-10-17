////////////////////////////////////////////////////////////////////////////
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
//
// checkpoint .hpp .cpp - Callback hooks to checkpoint model
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <memory>
#include <string>

namespace lbann {
namespace {
/**
 * Get the current execution context for the model.  However, if the current
 * execution context is not for training and there is a valid training
 * execution context, return the training context instead.
 *
 */
SGDExecutionContext const&
get_current_execution_context_with_training_override(model& m, trainer& t)
{
  const auto& c =
    dynamic_cast<SGDExecutionContext const&>(m.get_execution_context());
  if (c.get_execution_mode() != execution_mode::training &&
      t.execution_context_valid(m, execution_mode::training)) {
    return dynamic_cast<SGDExecutionContext const&>(
      t.get_execution_context(&m, execution_mode::training));
  }
  return c;
}

/**
 * When generating a checkpoint for non-training execution phases, the epoch
 * number should be pulled from the training context to provide a proper
 * ordering of the checkpoint.
 */
size_t get_epoch_with_training_override(model& m, trainer& t)
{
  return get_current_execution_context_with_training_override(m, t).get_epoch();
}

/**
 * When generating a checkpoint for non-training execution phases, the step
 * count should be pulled from the training context to provide a proper
 * ordering of the checkpoint.
 */
size_t get_step_with_training_override(model& m, trainer& t)
{
  return get_current_execution_context_with_training_override(m, t).get_step();
}
} // namespace

namespace callback {

// Load from checkpoint occurs during setup callbacks
void checkpoint::on_setup_begin(model* m) { reload_model(m); }

void checkpoint::setup(trainer* t)
{
  set_active_trainer(t);
  auto& p = get_active_trainer().get_persist_obj();
  p.set_cb_type(callback_type::invalid);
  reload_trainer(t);
}

// Restoring the execution context from checkpoint occurs during just
// before execution phase
void checkpoint::on_train_begin(model* m)
{
  auto& p = get_active_trainer().get_persist_obj();
  p.set_cb_type(callback_type::full_checkpoint);
  restart(m);
}

void checkpoint::on_train_end(model* m)
{
  auto& p = get_active_trainer().get_persist_obj();
  p.set_cb_type(callback_type::full_checkpoint);
  if (need_checkpoint(m, callback_phase::epoch)) {
    do_checkpoint(m, visitor_hook::execution_mode_end);
  }
  p.set_cb_type(callback_type::invalid);
}

// Interval defined with checkpoint_epochs or ckpt_dist_epochs
void checkpoint::on_epoch_begin(model* m)
{
  auto& p = get_active_trainer().get_persist_obj();
  p.set_cb_type(callback_type::full_checkpoint);
  if (need_checkpoint(m, callback_phase::epoch)) {
    do_checkpoint(m, visitor_hook::epoch_begin);
  }
  p.set_cb_type(callback_type::invalid);
}
// Interval defined with checkpoint_epochs or ckpt_dist_epochs
void checkpoint::on_validation_begin(model* m)
{
  auto& p = get_active_trainer().get_persist_obj();
  p.set_cb_type(callback_type::full_checkpoint);
  if (need_checkpoint(m, callback_phase::validation)) {
    do_checkpoint(m, visitor_hook::execution_mode_begin);
  }
  p.set_cb_type(callback_type::invalid);
}
// Interval defined with checkpoint_steps or ckpt_dist_steps
void checkpoint::on_batch_begin(model* m)
{
  auto& p = get_active_trainer().get_persist_obj();
  p.set_cb_type(callback_type::full_checkpoint);
  if (need_checkpoint(m, callback_phase::batch)) {
    do_checkpoint(m, visitor_hook::execution_mode_batch_begin);
  }
  p.set_cb_type(callback_type::invalid);
}

// Decide if we need to trigger a checkpoint for either mode, based on prototext
// defined intervals
bool checkpoint::need_checkpoint(model* m, callback_phase phase)
{
  /* TODO: since we're using clocks, this requires a bcast for each call,
   * we could use number of samples processed to make a local decision */
  // if none of our checkpoint conditions are set, assume we're not
  // checkpointing
  if (m_checkpoint_epochs == 0 && m_checkpoint_steps == 0 &&
      m_checkpoint_secs == 0.0 && m_ckpt_dist_epochs == 0 &&
      m_ckpt_dist_steps == 0) {
    return false;
  }
  // assume that we won't checkpoint
  m_checkpoint_shared = false;
  m_checkpoint_dist = false;
  lbann_comm* comm = m->get_comm();
  auto& t = this->get_active_trainer();
  size_t const cur_epoch = get_epoch_with_training_override(*m, t);
  size_t const cur_step = get_step_with_training_override(*m, t);
  // If we are at the end of a training epoch and the training epoch lands on
  // defined interval, ckpt
  if (!m_checkpoint_shared && m_checkpoint_epochs > 0 &&
      (phase == callback_phase::epoch || phase == callback_phase::validation)) {
    m_checkpoint_shared =
      (cur_epoch > 0) && (cur_epoch % m_checkpoint_epochs == 0);
  }

  if (!m_checkpoint_dist && m_ckpt_dist_epochs > 0 &&
      (phase == callback_phase::epoch || phase == callback_phase::validation)) {
    m_checkpoint_dist =
      (cur_epoch > 0) && (cur_epoch % m_ckpt_dist_epochs == 0);
  }

  // If we are at the end of a training mb step and the training mb step lands
  // on defined interval, trigger checkpoint
  if (!m_checkpoint_shared && m_checkpoint_steps > 0) {
    m_checkpoint_shared =
      (cur_step > 0) && (cur_step % m_checkpoint_steps == 0);
  }

  if (!m_checkpoint_dist && m_ckpt_dist_steps > 0) {
    m_checkpoint_dist = (cur_step > 0) && (cur_step % m_ckpt_dist_steps == 0);
  }

  // check the clock if time-based checkpoint is enabled
  if (!m_checkpoint_shared && m_checkpoint_secs != 0.0) {
    // have rank 0 determine whether we should checkpoint
    // to avoid issues with clock skew, we rely on rank 0 to make decision
    if (comm->am_trainer_master()) {
      // get the current time
      EvalType current = MPI_Wtime();
      // compute time next checkpoint is due
      EvalType next = m_checkpoint_last + m_checkpoint_secs;
      // determine whether it's time for a checkpoint
      m_checkpoint_shared = (current >= next);
    }
    comm->trainer_broadcast(0, m_checkpoint_shared);
  }
  // If either checkpoint version is triggered, return true, otherwise false.
  return (m_checkpoint_shared || m_checkpoint_dist);
}

// Checkpoint Shared/Distributed
bool checkpoint::do_checkpoint(model* m, visitor_hook hook)
{
  auto& p = get_active_trainer().get_persist_obj();
  auto& c = dynamic_cast<SGDExecutionContext&>(m->get_execution_context());
  auto& t = get_active_trainer();
  if (&t != &get_trainer()) {
    LBANN_ERROR("Mismatched trainers");
  }
  // if the checkpoint directory is not defined, bail
  if (get_checkpoint_dir().length() == 0 && m_per_rank_dir.length() == 0) {
    return false;
  }
  // time how long this takes
  // read current epoch and step counters from model
  El::Timer timer;
  std::string epochdir;
  std::string latest_file;
  size_t epoch = std::numeric_limits<size_t>::max();
  size_t step = std::numeric_limits<size_t>::max();
  lbann_comm* comm = m->get_comm();
  // TODO: we would want to prepend dir with the model name and model rank:
  // m->get_name() + '.' + std::to_string(comm->get_trainer_rank()) + '.'
  // However, rng state is not part of model state but that of the world.
  // So, it needs to be in the root folder.
  comm->trainer_barrier();
  // let user know we're saving a checkpoint
  if (comm->am_trainer_master()) {
    epoch = get_epoch_with_training_override(*m, t);
    step = get_step_with_training_override(*m, t);
    timer.Start();
    std::cout << "[" << m->get_name() << "." << comm->get_trainer_rank()
              << "] Checkpoint ["
              << (is_execution_mode_hook(hook)
                    ? to_string(hook, c.get_execution_mode())
                    : to_string(hook))
              << "] to " << get_checkpoint_dir() << " : epoch " << epoch
              << " step " << step << " ..." << std::endl;
    fflush(stdout);
  }
  comm->trainer_broadcast(0, epoch);
  comm->trainer_broadcast(0, step);

  // Distributed ckpt
  if (m_checkpoint_dist) {
    this->do_distributed_checkpoint(*comm,
                                    t,                      /* trainer */
                                    *m,                     /* model   */
                                    hook,                   /* visitor hook */
                                    c.get_execution_mode(), /* mode */
                                    p,                      /* persist */
                                    epoch,
                                    step);
  }
  // Shared checkpoint
  if (m_checkpoint_shared) {
    this->do_shared_checkpoint(*comm,
                               t,                      /* trainer */
                               *m,                     /* model   */
                               hook,                   /* visitor hook */
                               c.get_execution_mode(), /* mode */
                               p,                      /* persist */
                               epoch,
                               step);
  }

  uint64_t bytes_count = p.get_bytes();

  if (comm->am_trainer_master()) {
    EvalType secs = timer.Stop();
    EvalType bw = 0;
    if (secs > 0.0) {
      bw = EvalType(bytes_count) / (secs * 1024.0 * 1024.0);
    }
    std::cout << "[" << m->get_name() << "." << comm->get_trainer_rank()
              << "] Checkpoint ["
              << (is_execution_mode_hook(hook)
                    ? to_string(hook, c.get_execution_mode())
                    : to_string(hook))
              << "] to " << get_checkpoint_dir() << " complete: Epoch=" << epoch
              << " Step=" << step << " (" << secs << " secs, " << bytes_count
              << " bytes, " << bw << " MB/sec)" << std::endl;
    fflush(stdout);
  }
  // record last checkpoint time in case checkpoint_secs interval defined.
  m_checkpoint_last = MPI_Wtime();
  p.reset_bytes();
  return true;
}

std::string checkpoint::find_latest_checkpoint(lbann_comm& comm,
                                               const std::string& trainer_name,
                                               const std::string& alg_name,
                                               visitor_hook& hook,
                                               execution_mode& mode,
                                               size_t& epoch,
                                               size_t& step,
                                               bool& shared)
{
  constexpr unsigned int max_len_dirname = 1024;
  std::string dir;
  size_t epoch_dist = 0;
  size_t step_dist = 0;

  // Grab latest checkpoint information, checks for latest in dist and shared,
  // restarts from most recent between the two.
  if (comm.am_trainer_master()) {
    std::string latest_file;
    if (m_per_rank_dir.length()) {
      dir = get_distributed_checkpoint_rootdir();
      latest_file =
        get_last_distributed_checkpoint_filename(trainer_name, alg_name, dir);
      read_latest(latest_file, &hook, &mode, &epoch_dist, &step_dist);
    }
    if (get_restart_dir().length()) {
      dir = get_shared_checkpoint_rootdir();
      latest_file =
        get_last_shared_checkpoint_filename(trainer_name, alg_name, dir);
      read_latest(latest_file, &hook, &mode, &epoch, &step);
    }

    if (epoch > epoch_dist) {
      dir = get_shared_checkpoint_rootdir();
      shared = 1;
    }
    else if (epoch == epoch_dist && step > step_dist) {
      dir = get_shared_checkpoint_rootdir();
      shared = 1;
    }
    else {
      dir = get_distributed_checkpoint_rootdir();
      step = step_dist;
      epoch = epoch_dist;
      shared = 0;
    }
  }
  // Update other ranks on where we are loading from.
  // TODO: we would want to prepend dir with the model name and model rank:
  // m->get_name() + '.' + std::to_string(comm->get_trainer_rank()) + '.'
  header_t<max_len_dirname> header;
  std::memset(&header, 0x0, sizeof(header_t<max_len_dirname>));

  if (comm.am_trainer_master()) {
    header.hook = hook;
    header.mode = mode;
    header.epoch = epoch;
    header.step = step;
    header.shared = shared;
    dir.copy(header.dirname, dir.length(), 0);
  }

  comm.trainer_broadcast(0, header);

  if (!comm.am_trainer_master()) {
    hook = header.hook;
    mode = header.mode;
    epoch = header.epoch;
    step = header.step;
    shared = header.shared;
    dir = header.dirname;
  }
  return dir;
}

// Open latest Shared/Distributed checkpoint
bool checkpoint::open_latest_checkpoint(
  lbann_comm& comm,
  const std::string& task_label,
  const std::string& trainer_name,
  const std::string& alg_name,
  std::function<bool(persist&)> reload_shared_ckpt,
  std::function<bool(persist&)> reload_distributed_ckpt)
{
  // if the checkpoint directory is not defined, bail
  if (get_restart_dir().length() == 0 && m_per_rank_dir.length() == 0) {
    return false;
  }
  auto& p = get_active_trainer().get_persist_obj();

  // constexpr unsigned int max_len_dirname = 1024;
  // get top level directory
  // char dir[max_len_dirname];
  size_t epoch = std::numeric_limits<size_t>::max();
  size_t step = std::numeric_limits<size_t>::max();
  bool shared = true;
  visitor_hook hook;
  execution_mode mode;

  std::string dir = find_latest_checkpoint(comm,
                                           trainer_name,
                                           alg_name,
                                           hook,
                                           mode,
                                           epoch,
                                           step,
                                           shared);

  // if we couldn't find the latest epoch, just return
  if (epoch == std::numeric_limits<size_t>::max()) {
    return false;
  }
  // time how long this takes
  El::Timer timer;
  // let user know we're restarting from a checkpoint
  if (comm.am_trainer_master()) {
    timer.Start();
    std::cout << task_label << " from " << get_restart_dir() << " : hook "
              << (is_execution_mode_hook(hook) ? to_string(hook, mode)
                                               : to_string(hook))
              << " epoch " << epoch << " step " << step << " ..." << std::endl;
  }

  std::string epochdir;
  // Create dir to restart from based off last recorded checkpoint (or overriden
  // values in last.shared[distributed].checkpoint
  if (!shared) {
    epochdir = get_distributed_checkpoint_dirname(trainer_name,
                                                  alg_name,
                                                  comm.get_rank_in_trainer(),
                                                  dir,
                                                  hook,
                                                  mode,
                                                  epoch,
                                                  step);
    if (!file::directory_exists(epochdir)) {
      LBANN_WARNING(epochdir + " does not exist");
      return false;
    }
    p.open_restart(epochdir.c_str());
    if (!reload_distributed_ckpt(p))
      LBANN_WARNING("Unable to reload distributed checkpoint ", epochdir);
    p.close_restart();
  }
  else {
    epochdir = get_shared_checkpoint_dirname(trainer_name,
                                             alg_name,
                                             dir,
                                             hook,
                                             mode,
                                             epoch,
                                             step);

    if (!file::directory_exists(epochdir)) {
      LBANN_WARNING(epochdir + " does not exist");
      return false;
    }
    // if (comm->am_trainer_master()) {
    /// @todo For the moment let all ranks open the checkpoint files
    p.open_restart(epochdir.c_str());
    // } else {
    // // Ensure all ranks have access to checkpoint dir, needed for loading
    // rank specific rng state
    //   p.m_checkpoint_dir = epochdir;
    // }
    if (!reload_shared_ckpt(p))
      LBANN_WARNING("Unable to reload shared checkpoint ", epochdir);
    /// @todo For the moment let all ranks open the checkpoint files
    p.close_restart();
    // }
  }

  // close our checkpoint
  uint64_t bytes_count = p.get_bytes();
  // let user know we've completed reading our restart
  if (comm.am_trainer_master()) {
    EvalType secs = timer.Stop();
    EvalType bw = 0.0;
    if (secs > 0.0) {
      bw = EvalType(bytes_count) / (secs * 1024.0 * 1024.0);
    }
    std::cout << "[" << trainer_name << "] " << task_label << " from "
              << get_restart_dir() << " complete: Epoch=" << epoch
              << " Step=" << step << " (" << secs << " secs, " << bytes_count
              << " bytes, " << bw << " MB/sec)" << std::endl;
    fflush(stdout);
  }
  p.reset_bytes();
  return true;
}

// Reload a model from a Shared/Distributed checkpoint
bool checkpoint::reload_model(model* m)
{
  lbann::utils::grid_manager grid_raii(m->get_comm()->get_trainer_grid());
  return open_latest_checkpoint(
    *(m->get_comm()),
    "Reloading Model " + m->get_name(),
    get_active_trainer().get_name(),
    get_active_training_algorithm().get_type(),
    [m](persist& p_ref) { return m->load_from_checkpoint_shared(p_ref); },
    [m](persist& p_ref) { return m->load_from_checkpoint_distributed(p_ref); });
}

// Reload a model from a Shared/Distributed checkpoint
bool checkpoint::reload_trainer(trainer* t)
{
  return open_latest_checkpoint(
    *(t->get_comm()),
    "Reloading Trainer",
    t->get_name(),
    "sgd",
    [t](persist& p_ref) { return t->load_from_checkpoint_shared(p_ref); },
    [t](persist& p_ref) { return t->load_from_checkpoint_distributed(p_ref); });
}

// Restart previously saved Shared/Distributed execution contexts
bool checkpoint::restart(model* m)
{
  // This function needs to read the checkpoint to see what execution
  // contexts exists and create a valid execution context for each
  // one.
  // Then setup the model with the proper one
  auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  return open_latest_checkpoint(
    *(m->get_comm()),
    "Restarting",
    get_active_trainer().get_name(),
    get_active_training_algorithm().get_type(),
    [&m, &c](persist& p_ref) {
      return get_trainer().load_from_checkpoint_shared(*m, c);
    },
    [&m, &c](persist& p_ref) {
      return get_trainer().load_from_checkpoint_distributed(*m, c);
    });
}

void checkpoint::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_checkpoint();
  msg->set_checkpoint_dir(m_checkpoint_dir);
  msg->set_restart_dir(m_restart_dir);
  msg->set_checkpoint_epochs(m_checkpoint_epochs);
  msg->set_checkpoint_steps(m_checkpoint_steps);
  msg->set_checkpoint_secs(m_checkpoint_secs);
  msg->set_per_rank_dir(m_per_rank_dir);
  msg->set_ckpt_dist_epochs(m_ckpt_dist_epochs);
  msg->set_ckpt_dist_steps(m_ckpt_dist_steps);
}

void checkpoint::do_distributed_checkpoint(lbann_comm& comm,
                                           trainer& t,
                                           model& m,
                                           visitor_hook hook,
                                           execution_mode mode,
                                           persist& p,
                                           size_t epoch,
                                           size_t step)
{
  if (!m_checkpoint_dist)
    return;

  // Prepend per rank directory with shared checkpoint dir name
  // Per rank directory typically a cache location like node local SSDs
  std::string dir;
  {
    std::ostringstream dir_oss;
    if (m_per_rank_dir.length() != 0) {
      // @todo BVE FIXME this looks wrong  I think that the order
      // should be reversed
      dir_oss << m_per_rank_dir << "/";
    }
    dir_oss << this->get_checkpoint_dir();
    dir = dir_oss.str();
  }

  // make the base directory
  makedir(dir.c_str());

  // now create directories per ranks
  //
  // NOTE: training_algorithm::get_type() is not entirely correct
  // here. It could cause problems if multiple instances of the same
  // callback are checkpointed at the same time, but currently we
  // don't do this. The issue with "get_name()" is that we might not
  // know the name of the training algorithm on restore (since the
  // training algos themselves are not checkpointed).
  auto const epochdir = get_distributed_checkpoint_dirname(
    t.get_name(),
    this->get_active_training_algorithm().get_type(),
    comm.get_rank_in_trainer(),
    dir,
    hook,
    mode,
    epoch,
    step);

  // @todo BVE FIXME this should be refactored to only open the
  // checkpoints files that we care about
  p.open_checkpoint(epochdir.c_str(), true);

  // Make sure that the master has had a chance to create the directories
  comm.trainer_barrier();

  // Call top level save to checkpoint function in model, in turn
  // calls save to checkpoint functions for other model classes
  // (weights, layers)
  if ((p.get_cb_type() == callback_type::model_only) ||
      (p.get_cb_type() == callback_type::full_checkpoint)) {
    m.save_to_checkpoint_distributed(p);
  }
  if ((p.get_cb_type() == callback_type::execution_context_only) ||
      (p.get_cb_type() == callback_type::full_checkpoint)) {
    t.save_to_checkpoint_distributed();
  }
  p.close_checkpoint();

  // Print latest checkpoint to file
  if (comm.am_trainer_master()) {
    auto const latest_file = get_last_distributed_checkpoint_filename(
      t.get_name(),
      this->get_active_training_algorithm().get_type(),
      dir);
    write_latest(latest_file, hook, mode, epoch, step);
  }
}

void checkpoint::do_shared_checkpoint(lbann_comm& comm,
                                      trainer& t,
                                      model& m,
                                      visitor_hook hook,
                                      execution_mode mode,
                                      persist& p,
                                      size_t epoch,
                                      size_t step)
{
  if (!m_checkpoint_shared)
    return;

  auto const dir = this->get_checkpoint_dir();
  makedir(dir.c_str());

  auto const epochdir = get_shared_checkpoint_dirname(
    t.get_name(),
    this->get_active_training_algorithm().get_type(),
    dir,
    hook,
    mode,
    epoch,
    step);
  p.open_checkpoint(epochdir.c_str(), comm.am_trainer_master());

  // Make sure that the master has had a chance to create the directories
  comm.trainer_barrier();
  if ((p.get_cb_type() == callback_type::model_only) ||
      (p.get_cb_type() == callback_type::full_checkpoint)) {
    m.save_to_checkpoint_shared(p);
  }
  if ((p.get_cb_type() == callback_type::execution_context_only) ||
      (p.get_cb_type() == callback_type::full_checkpoint)) {
    t.save_to_checkpoint_shared();
  }

  // close our checkpoint
  p.close_checkpoint();
  if (comm.am_trainer_master()) {
    auto const latest_file = get_last_shared_checkpoint_filename(
      t.get_name(),
      this->get_active_training_algorithm().get_type(),
      dir);
    write_latest(latest_file, hook, mode, epoch, step);
  }
}

trainer& checkpoint::get_active_trainer()
{
  if (m_active_trainer == nullptr) {
    LBANN_ERROR("No active trainer for the checkpoint callback");
  }
  return *m_active_trainer;
}

TrainingAlgorithm& checkpoint::get_active_training_algorithm()
{
  if (m_active_training_algorithm == nullptr) {
    LBANN_ERROR("No active training algorithm for the checkpoint callback");
  }
  return *m_active_training_algorithm;
}

std::string get_trainer_checkpoint_dirname(const std::string& trainer_name,
                                           const std::string& dir)
{
  return build_string(dir, '/', trainer_name, '/');
}

std::string get_last_shared_checkpoint_filename(const std::string& alg_name,
                                                const std::string& dir)
{
  return build_string(dir, '/', alg_name, ".last.shared.checkpoint");
}

std::string get_last_shared_checkpoint_filename(const std::string& trainer_name,
                                                const std::string& alg_name,
                                                const std::string& dir)
{
  return get_last_shared_checkpoint_filename(
    alg_name,
    get_trainer_checkpoint_dirname(trainer_name, dir));
}

std::string get_shared_checkpoint_dirname(const std::string& alg_name,
                                          const std::string& dir,
                                          visitor_hook hook,
                                          execution_mode mode,
                                          size_t epoch,
                                          size_t step)
{
  return build_string(
    dir,
    '/',
    alg_name,
    ".shared.",
    (is_execution_mode_hook(hook) ? to_string(hook, mode) : to_string(hook)),
    ".epoch.",
    epoch,
    ".step.",
    step,
    '/');
}

std::string get_shared_checkpoint_dirname(const std::string& trainer_name,
                                          const std::string& alg_name,
                                          const std::string& dir,
                                          visitor_hook hook,
                                          execution_mode mode,
                                          size_t epoch,
                                          size_t step)
{
  return get_shared_checkpoint_dirname(
    alg_name,
    get_trainer_checkpoint_dirname(trainer_name, dir),
    hook,
    mode,
    epoch,
    step);
}

std::string
get_last_distributed_checkpoint_filename(const std::string& alg_name,
                                         const std::string& dir)
{
  return build_string(dir, '/', alg_name, ".last.distributed.checkpoint");
}

std::string
get_last_distributed_checkpoint_filename(const std::string& trainer_name,
                                         const std::string& alg_name,
                                         const std::string& dir)
{
  return get_last_distributed_checkpoint_filename(
    alg_name,
    get_trainer_checkpoint_dirname(trainer_name, dir));
}

std::string get_distributed_checkpoint_dirname(const std::string& alg_name,
                                               const int rank_in_trainer,
                                               const std::string& dir,
                                               visitor_hook hook,
                                               execution_mode mode,
                                               size_t epoch,
                                               size_t step)
{
  return build_string(
    dir,
    '/',
    alg_name,
    ".rank.",
    rank_in_trainer,
    ".distributed.",
    (is_execution_mode_hook(hook) ? to_string(hook, mode) : to_string(hook)),
    ".epoch.",
    epoch,
    ".step.",
    step,
    '/');
}

std::string get_distributed_checkpoint_dirname(const std::string& trainer_name,
                                               const std::string& alg_name,
                                               const int rank_in_trainer,
                                               const std::string& dir,
                                               visitor_hook hook,
                                               execution_mode mode,
                                               size_t epoch,
                                               size_t step)
{
  return get_distributed_checkpoint_dirname(
    alg_name,
    rank_in_trainer,
    get_trainer_checkpoint_dirname(trainer_name, dir),
    hook,
    mode,
    epoch,
    step);
}

// Print last checkpoint to file, used to determine which checkpoint to load
// from.
bool write_latest(std::string filename,
                  visitor_hook hook,
                  execution_mode mode,
                  size_t epoch,
                  size_t train)
{
  // open the file for writing
  int fd = openwrite(filename.c_str());
  if (fd != -1) {
    char field[256];
    std::string hookStr =
      is_execution_mode_hook(hook) ? to_string(hook, mode) : to_string(hook);
    sprintf(field,
            "hook=%s epoch=%ld step=%ld\n",
            hookStr.c_str(),
            epoch,
            train);
    write_string(fd, filename.c_str(), field, strlen(field));
    // close our file
    closewrite(fd, filename.c_str());
  }
  return true;
}

/** \brief Reads the "latest" file and returns the epoch number and
 *        sample offset for most recent checkpoint
 */
bool read_latest(std::string filename,
                 visitor_hook* hook,
                 execution_mode* mode,
                 size_t* epochLast,
                 size_t* trainLast)
{
  // assume we don't have a file, we'll return -1 in that case
  *epochLast = -1;
  *trainLast = -1;
  *mode = execution_mode::invalid;
  *hook = visitor_hook::invalid;
  // open the file for reading
  int fd = openread(filename.c_str());
  if (fd != -1) {
    // read epoch from file
    char field[256];
    read_string(fd, filename.c_str(), field, sizeof(field));
    char hookStr[64];
    int ret = sscanf(field,
                     "hook=%s epoch=%ld step=%ld\n",
                     hookStr,
                     epochLast,
                     trainLast);
    visitor_hook_from_string(hookStr, *hook, *mode);
    // close our file
    closeread(fd, filename.c_str());
    if (ret != 3) {
      return false;
    }
    return true;
  }
  return false;
}

std::unique_ptr<callback_base>
build_checkpoint_callback_from_pbuf(const google::protobuf::Message& proto_msg)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackCheckpoint&>(proto_msg);
  return std::make_unique<checkpoint>(params.checkpoint_dir(),
                                      params.restart_dir(),
                                      params.checkpoint_epochs(),
                                      params.checkpoint_steps(),
                                      params.checkpoint_secs(),
                                      params.per_rank_dir(),
                                      params.ckpt_dist_epochs(),
                                      params.ckpt_dist_steps());
}

} // namespace callback
} // namespace lbann
