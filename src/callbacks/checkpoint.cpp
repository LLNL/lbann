////////////////////////////////////////////////////////////////////////////
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
//
// checkpoint .hpp .cpp - Callback hooks to checkpoint model
////////////////////////////////////////////////////////////////////////////////


#include "lbann/callbacks/checkpoint.hpp"

#include "lbann/models/model.hpp"

#include <callbacks.pb.h>

#include <memory>
#include <string>

namespace lbann {
namespace callback {
// Load from checkpoint occurs during setup callbacks
void checkpoint::setup(model *m) {
  p.set_cb_type(callback_type::invalid);
  reload_model(m);
}

// Restoring the execution context from checkpoint occurs during just
// before execution phase
void checkpoint::on_train_begin(model *m) {
  p.set_cb_type(callback_type::full_checkpoint);
  restart(m);
}

// Interval defined with checkpoint_epochs or ckpt_dist_epochs
void checkpoint::on_epoch_end(model *m) {
  p.set_cb_type(callback_type::full_checkpoint);
  if(need_checkpoint(m, callback_phase::epoch)){
    do_checkpoint(m);
  }
  p.set_cb_type(callback_type::invalid);
}
// Interval defined with checkpoint_epochs or ckpt_dist_epochs
void checkpoint::on_validation_end(model *m) {
  p.set_cb_type(callback_type::full_checkpoint);
  if(need_checkpoint(m, callback_phase::validation)){
    do_checkpoint(m);
  }
  p.set_cb_type(callback_type::invalid);
}
 // Interval defined with checkpoint_steps or ckpt_dist_steps
void checkpoint::on_batch_end(model *m) {
  p.set_cb_type(callback_type::full_checkpoint);
  if(need_checkpoint(m, callback_phase::batch)){
    do_checkpoint(m);
  }
  p.set_cb_type(callback_type::invalid);
}

// Decide if we need to trigger a checkpoint for either mode, based on prototext defined intervals
bool checkpoint::need_checkpoint(model *m, callback_phase phase) {
  const auto& c = static_cast<sgd_execution_context&>(m->get_execution_context());
  /* TODO: since we're using clocks, this requires a bcast for each call,
   * we could use number of samples processed to make a local decision */
  // if none of our checkpoint conditions are set, assume we're not checkpointing
  if (m_checkpoint_epochs == 0 &&
      m_checkpoint_steps  == 0 &&
      m_checkpoint_secs   == 0.0 &&
      m_ckpt_dist_epochs == 0 &&
      m_ckpt_dist_steps== 0) {
    return false;
  }
  // assume that we won't checkpoint
  m_checkpoint_shared = false;
  m_checkpoint_dist = false;
  lbann_comm *comm = m->get_comm();
  int cur_epoch = c.get_epoch();
  // If we are at the end of a training epoch and the training epoch lands on defined interval, ckpt
  if (!m_checkpoint_shared && m_checkpoint_epochs > 0 && (phase == callback_phase::epoch || phase == callback_phase::validation)){
      m_checkpoint_shared = (cur_epoch > 0) && (cur_epoch % m_checkpoint_epochs == 0);
    }

  if(!m_checkpoint_dist && m_ckpt_dist_epochs > 0 && (phase == callback_phase::epoch || phase == callback_phase::validation)){
      m_checkpoint_dist = (cur_epoch > 0) && (cur_epoch % m_ckpt_dist_epochs == 0);
  }

  // If we are at the end of a training mb step and the training mb step lands on defined interval, trigger checkpoint
  if (!m_checkpoint_shared && m_checkpoint_steps > 0) {
    m_checkpoint_shared = (c.get_step() > 0) && (c.get_step() % m_checkpoint_steps == 0);
  }

  if(!m_checkpoint_dist && m_ckpt_dist_steps > 0){
      m_checkpoint_dist = (c.get_step() > 0) && (c.get_step() % m_ckpt_dist_steps == 0);
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
bool checkpoint::do_checkpoint(model *m) {
  auto& c = static_cast<sgd_execution_context&>(m->get_execution_context());
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0 && m_per_rank_dir.length() == 0) {
    return false;
  }
  // time how long this takes
  // read current epoch and step counters from model
  El::Timer timer;
  char dir[1024];
  std::string epochdir;
  std::string latest_file;
  size_t epoch = std::numeric_limits<size_t>::max();
  size_t step = std::numeric_limits<size_t>::max();
  lbann_comm *comm = m->get_comm();
  // TODO: we would want to prepend dir with the model name and model rank:
  // m->get_name() + '.' + std::to_string(comm->get_trainer_rank()) + '.'
  // However, rng state is not part of model state but that of the world.
  // So, it needs to be in the root folder.
  comm->trainer_barrier();
  // let user know we're saving a checkpoint
  if (comm->am_trainer_master()) {
    epoch = c.get_epoch();
    step = c.get_step();
    timer.Start();
    std::cout << "Checkpoint [" << to_string(c.get_execution_mode())
              << "]: epoch " << epoch << " step " << step << " ..." << std::endl;
    fflush(stdout);
  }
  comm->trainer_broadcast(0, epoch);
  comm->trainer_broadcast(0, step);

  // Distributed ckpt
  if(m_checkpoint_dist){
    // prepend per rank directory with shared checkpoint dir name
    // Per rank directory typically a cache location like node local SSDs
    if(m_per_rank_dir.length() != 0){
      snprintf(dir, sizeof(dir), "%s/%s", m_per_rank_dir.c_str(), m_checkpoint_dir.c_str());
    } else {
      strcpy(dir, m_checkpoint_dir.c_str());
    }
    makedir(dir);
    // create directories per ranks
    epochdir = get_distributed_checkpoint_dirname(m, dir, c.get_execution_mode(), epoch, step);
    /** @todo BVE FIXME this should be refactored to only open the
        checkpoints files that we care about */
    p.open_checkpoint(epochdir.c_str());
    // Call top level save to checkpoint function in model, in turn calls save to checkpoint functions for other model classes (weights, layers)
    if(p.get_cb_type() == callback_type::model_only || p.get_cb_type() == callback_type::full_checkpoint) {
      m->save_to_checkpoint_distributed(p);
    }
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint) {
      auto save_checkpoint = [this](observer_ptr<execution_context> ctx)
        ->void { ctx->save_to_checkpoint_distributed(this->p); };
      c.get_trainer().for_each_execution_context(save_checkpoint);
    }
    p.close_checkpoint();
    // Print latest checkpoint to file
    if (comm->am_trainer_master()) {
      latest_file = get_last_distributed_checkpoint_filename(m, dir);
      write_latest(latest_file, c.get_execution_mode(), epoch, step);
    }
  }
  // Shared checkpoint, logic identical to Distributed.i
  if(m_checkpoint_shared){
    strcpy(dir, m_checkpoint_dir.c_str());
    makedir(dir);
    epochdir = get_shared_checkpoint_dirname(m, dir, c.get_execution_mode(), epoch, step);
    if (comm->am_trainer_master()) {
      p.open_checkpoint(epochdir.c_str());
    }
    // Need to give other ranks knowledge of checkpoint dir for writing of rank specific rng state
    comm->trainer_broadcast(0, &(p.m_checkpoint_dir[0]), sizeof(p.m_checkpoint_dir));
    if(p.get_cb_type() == callback_type::model_only || p.get_cb_type() == callback_type::full_checkpoint) {
      m->save_to_checkpoint_shared(p);
    }
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint) {
      auto save_checkpoint = [this](observer_ptr<execution_context> ctx)
         ->void { ctx->save_to_checkpoint_shared(this->p); };
      c.get_trainer().for_each_execution_context(save_checkpoint);
    }
    // close our checkpoint
    p.close_checkpoint();
    if (comm->am_trainer_master()) {
      latest_file = get_last_shared_checkpoint_filename(m, dir);
      write_latest(latest_file, c.get_execution_mode(), epoch, step);
    }
  }

  uint64_t bytes_count = p.get_bytes();

  if (comm->am_trainer_master()) {
    EvalType secs = timer.Stop();
    EvalType bw = 0;
    if (secs > 0.0) {
      bw = EvalType(bytes_count) / (secs * 1024.0 * 1024.0);
    }
    std::cout << "[" << m->get_name()
              << "." << comm->get_trainer_rank()
              << "] Checkpoint [" << to_string(c.get_execution_mode())
              << "] complete: Epoch=" << epoch
              << " Step=" << step
              << " (" << secs << " secs, " << bytes_count << " bytes, "
              << bw << " MB/sec)" << std::endl;
    fflush(stdout);
  }
  // record last checkpoint time in case checkpoint_secs interval defined.
  m_checkpoint_last = MPI_Wtime();
  p.reset_bytes();
  return true;
}

std::string checkpoint::find_latest_checkpoint(model *m, std::string& latest_file, execution_mode& mode, size_t &epoch, size_t& step, int& shared) {
  constexpr unsigned int max_len_dirname = 1024;
  char dir[max_len_dirname];
  size_t epoch_dist = 0;
  size_t step_dist = 0;
  lbann_comm *comm = m->get_comm();
  // Grab latest checkpoint information, checks for latest in dist and shared, restarts from most recent between the two.
  if (comm->am_trainer_master()) {
    if(m_per_rank_dir.length()){
      snprintf(dir, sizeof(dir), "%s/%s", m_per_rank_dir.c_str(), m_checkpoint_dir.c_str());
      latest_file = get_last_distributed_checkpoint_filename(m, dir);
      read_latest(latest_file, &mode, &epoch_dist, &step_dist);
    }
    if(m_checkpoint_dir.length()){
      strcpy(dir, m_checkpoint_dir.c_str());
      latest_file = get_last_shared_checkpoint_filename(m, dir);
      read_latest(latest_file, &mode, &epoch, &step);
    }

    if(epoch > epoch_dist){
      strcpy(dir, m_checkpoint_dir.c_str());
      shared = 1;
    }
    else if(epoch == epoch_dist && step > step_dist){
      strcpy(dir, m_checkpoint_dir.c_str());
      shared = 1;
    }
    else {
      snprintf(dir, sizeof(dir), "%s/%s", m_per_rank_dir.c_str(), m_checkpoint_dir.c_str());
      step = step_dist;
      epoch = epoch_dist;
      shared = 0;
    }
  }
  // Update other ranks on where we are loading from.
  // TODO: we would want to prepend dir with the model name and model rank:
  // m->get_name() + '.' + std::to_string(comm->get_trainer_rank()) + '.'
#if 1
  header_t<max_len_dirname> header;

  if (comm->am_trainer_master()) {
    header.mode = mode;
    header.epoch = epoch;
    header.step = step;
    header.shared = shared;
    memcpy(header.dirname, dir, sizeof(dir));
  }

  comm->trainer_broadcast(0, header);

  if (!comm->am_trainer_master()) {
    mode = header.mode;
    epoch = header.epoch;
    step = header.step;
    shared = header.shared;
    memcpy(dir, header.dirname, sizeof(dir));
  }
#else
  comm->trainer_broadcast(0, epoch);
  comm->trainer_broadcast(0, step);
  comm->trainer_broadcast(0, shared);
  comm->trainer_broadcast(0, &(dir[0]), sizeof(dir));
#endif
  return dir;
}

// Open latest Shared/Distributed checkpoint
bool checkpoint::open_latest_checkpoint(
  model *m,
  const std::string& task_label,
  std::function<void(/*const */persist&)> reload_shared_ckpt,
  std::function<void(/*const */persist&)> reload_distributed_ckpt) {
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0 &&  m_per_rank_dir.length() == 0) {
    return false;
  }

  // constexpr unsigned int max_len_dirname = 1024;
  // get top level directory
  // char dir[max_len_dirname];
  std::string latest_file;
  size_t epoch = std::numeric_limits<size_t>::max();
  size_t step = std::numeric_limits<size_t>::max();
  int shared = 1;
  execution_mode mode;
  lbann_comm *comm = m->get_comm();

  std::string dir = find_latest_checkpoint(m, latest_file, mode, epoch, step, shared);

  // if we couldn't find the latest epoch, just return
  if (epoch == std::numeric_limits<size_t>::max()) {
    return false;
  }
  // time how long this takes
  El::Timer timer;
  // let user know we're restarting from a checkpoint
  if (comm->am_trainer_master()) {
    timer.Start();
    std::cout << task_label << "ing: epoch " << epoch << " ..." << std::endl;
  }

  std::string epochdir;
  // Create dir to restart from based off last recorded checkpoint (or overriden values in last.shared[distributed].checkpoint
  if(!shared){
    epochdir = get_distributed_checkpoint_dirname(m, dir, mode, epoch, step);
    p.open_restart(epochdir.c_str());
    reload_distributed_ckpt(p);
    p.close_restart();
  }
  else {
    epochdir = get_shared_checkpoint_dirname(m, dir, mode, epoch, step);
    //    if (comm->am_trainer_master()) {
    /// @todo For the moment let all ranks open the checkpoint files
    p.open_restart(epochdir.c_str());
      //    }
    // Ensure all ranks have access to checkpoint dir, needed for loading rank specific rng state
    comm->trainer_broadcast(0, &(p.m_checkpoint_dir[0]), sizeof(p.m_checkpoint_dir));
    reload_shared_ckpt(p);
    //    if(comm->am_trainer_master())
    /// @todo For the moment let all ranks open the checkpoint files
    p.close_restart();
  }

  // close our checkpoint
  uint64_t bytes_count = p.get_bytes();
  // let user know we've completed reading our restart
  if (comm->am_trainer_master()) {
    EvalType secs = timer.Stop();
    EvalType bw = 0.0;
    if (secs > 0.0) {
      bw = EvalType(bytes_count) / (secs * 1024.0 * 1024.0);
    }
    std::cout << "[" << m->get_name()
              << "." << comm->get_trainer_rank()
              << "] " << task_label
              << " complete: Epoch=" << epoch
              << " Step=" << step
              << " (" << secs << " secs, " << bytes_count << " bytes, "
              << bw << " MB/sec)" << std::endl;
    fflush(stdout);
  }
  p.reset_bytes();
  return true;
}

// Reload a model from a Shared/Distributed checkpoint
bool checkpoint::reload_model(model *m) {
  auto reload_shared_model = std::function<void(/*const */persist&)>
    ([m](/*const */persist& p_ref)
     ->void {
      m->load_from_checkpoint_shared(p_ref);
      return;
    });

  auto reload_distributed_model = std::function<void(/*const */persist&)>
    ([m](/*const */persist& p_ref)
     ->void {
      m->load_from_checkpoint_distributed(p_ref);
      return;
    });


  open_latest_checkpoint(m, "Reload", reload_shared_model, reload_distributed_model);

  return true;
}


// Restart previously saved Shared/Distributed execution contexts
bool checkpoint::restart(model *m) {
  // This function needs to read the checkpoint to see what execution
  // contexts exists and create a valid execution context for each
  // one.
  // Then setup the model with the proper one
  sgd_execution_context& c = static_cast<sgd_execution_context&>(m->get_execution_context());

  auto restart_shared_model = std::function<void(/*const */persist&)>
    ([&m, &c](/*const */persist& p_ref)
     ->void {
      execution_mode current_mode = c.get_execution_mode();

      for(execution_mode mode : execution_mode_iterator()) {
        /// Restart should optionally load any other valid contexts
        if(mode == execution_mode::invalid) { continue; }
        trainer::execution_context_key_pair_t key;
        try {
          if(current_mode == mode) {
            /// Restart has to be able to load the currently running  execution context
            c.load_from_checkpoint_shared(p_ref);
          }else {
            key = c.get_trainer().check_and_build_execution_context(c, *m, mode);
            auto& evaluation_context = static_cast<sgd_execution_context&>(c.get_trainer().get_execution_context(key));
            evaluation_context.load_from_checkpoint_shared(p_ref);
          }
        }catch (NonexistentArchiveFile const&) {
          // Ignore the exception if the file is not for the current execution mode
          if(current_mode == mode) {
            LBANN_ERROR("Failed to restart model, invalid execution mode: " + to_string(current_mode));
          }else {
            c.get_trainer().delete_execution_context(key);
          }
        }
      }
      return;
    });

  auto restart_distributed_model = std::function<void(/*const */persist&)>
    ([&m, &c](/*const */persist& p_ref)
     ->void {
      execution_mode current_mode = c.get_execution_mode();

      for(execution_mode mode : execution_mode_iterator()) {
        /// Restart should optionally load any other valid contexts
        if(mode == execution_mode::invalid) { continue; }
        trainer::execution_context_key_pair_t key;
        try {
          if(current_mode == mode) {
            /// Restart has to be able to load the currently running  execution context
            c.load_from_checkpoint_distributed(p_ref);
          }else {
            key = c.get_trainer().check_and_build_execution_context(c, *m, mode);
            auto& evaluation_context = static_cast<sgd_execution_context&>(c.get_trainer().get_execution_context(key));
            evaluation_context.load_from_checkpoint_distributed(p_ref);
          }
        }catch (NonexistentArchiveFile const&) {
          // Ignore the exception if the file is not for the current execution mode
          if(current_mode == mode) {
            LBANN_ERROR("Failed to restart model, invalid execution mode: " + to_string(current_mode));
          }else {
            c.get_trainer().delete_execution_context(key);
          }
        }

      }
      return;
    });


  open_latest_checkpoint(m, "Restart", restart_shared_model, restart_distributed_model);

  return true;
}

std::unique_ptr<callback_base>
build_checkpoint_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackCheckpoint&>(proto_msg);
  return make_unique<checkpoint>(params.checkpoint_dir(),
                                                params.checkpoint_epochs(),
                                                params.checkpoint_steps(),
                                                params.checkpoint_secs(),
                                                params.per_rank_dir(),
                                                params.ckpt_dist_epochs(),
                                                params.ckpt_dist_steps());
}

} // namespace callback
} // namespace lbann
