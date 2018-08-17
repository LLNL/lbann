////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
// lbann_callback_checkpoint .hpp .cpp - Callback hooks to checkpoint model
////////////////////////////////////////////////////////////////////////////////


#include "lbann/callbacks/callback_checkpoint.hpp"

namespace lbann {
// Load from checkpoint occurs during setup callbacks
void lbann_callback_checkpoint::setup(model *m) {
  p.set_cb_type(callback_type::invalid);
  restart(m);
}
// Interval defined with checkpoint_epochs or ckpt_dist_epochs
void lbann_callback_checkpoint::on_epoch_end(model *m) {
  p.set_cb_type(callback_type::epoch);
  if(need_checkpoint(m)){
    checkpoint(m);
  }
  p.set_cb_type(callback_type::invalid);
}
 // Interval defined with checkpoint_epochs or ckpt_dist_epochs
void lbann_callback_checkpoint::on_validation_end(model *m) {
  p.set_cb_type(callback_type::validation);
  if(need_checkpoint(m)){
    checkpoint(m);
  }
  p.set_cb_type(callback_type::invalid);
}
 // Interval defined with checkpoint_steps or ckpt_dist_steps
void lbann_callback_checkpoint::on_batch_end(model *m) {
  p.set_cb_type(callback_type::batch);
  if(need_checkpoint(m)){
    checkpoint(m);
  }
  p.set_cb_type(callback_type::invalid);
}

// Decide if we need to trigger a checkpoint for either mode, based on prototext defined intervals
bool lbann_callback_checkpoint::need_checkpoint(model *m) {
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
  int cur_epoch = m->get_cur_epoch();
  // If we are at the end of a training epoch and the training epoch lands on defined interval, ckpt
  if (!m_checkpoint_shared && m_checkpoint_epochs > 0 && (p.get_cb_type() == callback_type::epoch || p.get_cb_type() == callback_type::validation)){
      m_checkpoint_shared = (cur_epoch > 0) && (cur_epoch % m_checkpoint_epochs == 0);
    }
    
  if(!m_checkpoint_dist && m_ckpt_dist_epochs > 0 && (p.get_cb_type() == callback_type::epoch || p.get_cb_type() == callback_type::validation)){
      m_checkpoint_dist = (cur_epoch > 0) && (cur_epoch % m_ckpt_dist_epochs == 0);
  }
  
  // If we are at the end of a training mb step and the training mb step lands on defined interval, trigger checkpoint
  if (!m_checkpoint_shared && m_checkpoint_steps > 0) {
    m_checkpoint_shared = (m->get_cur_step() > 0) && (m->get_cur_step() % m_checkpoint_steps == 0);
  }
  
  if(!m_checkpoint_dist && m_ckpt_dist_steps > 0){
      m_checkpoint_dist = (m->get_cur_step() > 0) && (m->get_cur_step() % m_ckpt_dist_steps == 0);
  }
  
  // check the clock if time-based checkpoint is enabled
  if (!m_checkpoint_shared && m_checkpoint_secs != 0.0) {
    // have rank 0 determine whether we should checkpoint
    // to avoid issues with clock skew, we rely on rank 0 to make decision
    if (comm->am_model_master()) {
      // get the current time
      EvalType current = MPI_Wtime();
      // compute time next checkpoint is due
      EvalType next = m_checkpoint_last + m_checkpoint_secs;
      // determine whether it's time for a checkpoint
      m_checkpoint_shared = (current >= next);
    }
    comm->model_broadcast(0, m_checkpoint_shared);
  } 
  // If either checkpoint version is triggered, return true, otherwise false.   
  return (m_checkpoint_shared || m_checkpoint_dist);
}

// Print last checkpoint to file, used to determine which checkpoint to load from. 
static bool write_latest(const char *dir, const char *name, int epoch, int train) {
  // define filename
  char filename[1024];
  sprintf(filename, "%s/%s", dir, name);
  // open the file for writing
  int fd = openwrite(filename);
  if (fd != -1) {
    char field[256];
    sprintf(field, "epoch=%d step=%d\n", epoch, train);
    write_string(fd, "last_checkpoint", field, strlen(field));
    // close our file
    closewrite(fd, filename);
  }
  return true;
}
/** \brief Reads the "latest" file and returns the epoch number and sample offset for most recent checkpoint */
static bool read_latest(const char *dir, const char *name, int *epochLast, int *trainLast) {
  // assume we don't have a file, we'll return -1 in that case
  *epochLast = -1;
  *trainLast = -1;
  // define filename
  char filename[1024];
  sprintf(filename, "%s/%s", dir, name);
  // open the file for reading
  int fd = openread(filename);
  if (fd != -1) {
    // read epoch from file
    char field[256];
    read_string(fd, "last_checkpoint", field, sizeof(field));
    int ret = sscanf(field, "epoch=%d step=%d\n", epochLast, trainLast);
    // close our file
    closeread(fd, filename);
    if(ret != 2) { return false; }
  }
  return true;
}
// Checkpoint Shared/Distributed
bool lbann_callback_checkpoint::checkpoint(model *m) {
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0 && m_per_rank_dir.length() == 0) {
    return false;
  }
  // time how long this takes
  // read current epoch and step counters from model
  El::Timer timer;
  char epochdir[1024];
  char dir[1024];
  int epoch = -1;
  int step = -1 ;
  lbann_comm *comm = m->get_comm();
  // TODO: we would want to prepend dir with the model name and model rank:
  // m->get_name() + '.' + std::to_string(comm->get_model_rank()) + '.'
  // However, rng state is not part of model state but that of the world.
  // So, it needs to be in the root folder.
  comm->model_barrier();
  // let user know we're saving a checkpoint
  if (comm->am_model_master()) {
    epoch = m->get_cur_epoch();
    step = m->get_cur_step();
    timer.Start();
    printf("Checkpoint: epoch %d step %d ...\n", epoch, step);
    fflush(stdout);
  }
  comm->model_broadcast(0, epoch);
  comm->model_broadcast(0, step);

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
    snprintf(epochdir, sizeof(epochdir), "%s/rank.%d.epoch.%d.step.%d", dir, comm->get_rank_in_model(), epoch, step);
    p.open_checkpoint(epochdir);
    // Call top level save to checkpoint function in model, in turn calls save to checkpoint functions for other model classes (weights, layers)
    m->save_to_checkpoint_distributed(p);
    p.close_checkpoint();
    // Print latest checkpoint to file
    if (comm->am_model_master()) {
      write_latest(dir, "last.distributed.checkpoint", epoch, step);
    } 
  }
  // Shared checkpoint, logic identical to Distributed.
  if(m_checkpoint_shared){
    strcpy(dir, m_checkpoint_dir.c_str());
    makedir(dir);
    snprintf(epochdir, sizeof(epochdir), "%s/shared.model.%d.epoch.%d.step.%d", dir, comm->get_model_rank(), epoch, step);
    if (comm->am_model_master()) {
      p.open_checkpoint(epochdir);
    }
      // Need to give other ranks knowledge of checkpoint dir for writing of rank specific rng state
    comm->model_broadcast(0, &(p.m_checkpoint_dir[0]), sizeof(p.m_checkpoint_dir));
    m->save_to_checkpoint_shared(p);
    // close our checkpoint
    if (comm->am_model_master()) {
      p.close_checkpoint();  
      write_latest(dir, "last.shared.checkpoint", epoch, step);
    }
  }

  uint64_t bytes_count = p.get_bytes();

  if (comm->am_model_master()) {
    EvalType secs = timer.Stop();
    EvalType bw = 0;
    if (secs > 0.0) {
      bw = EvalType(bytes_count) / (secs * 1024.0 * 1024.0);
    }
    printf("Checkpoint complete: Epoch=%d Step=%d (%f secs, %llu bytes, %f MB/sec)\n",
           epoch, step, secs, (unsigned long long) bytes_count, bw);
    fflush(stdout);
  }
  // record last checkpoint time in case checkpoint_secs interval defined.
  m_checkpoint_last = MPI_Wtime();
  p.reset_bytes();
  return true;
}

// Restart Shared/Distributed
bool lbann_callback_checkpoint::restart(model *m) {
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0 &&  m_per_rank_dir.length() == 0) {
    return false;
  }
  constexpr unsigned int max_len_dirname = 1024;
  // get top level directory
  char dir[max_len_dirname];
  int epoch = -1;
  int step = -1;
  int epoch_dist = -1;
  int step_dist = -1;
  lbann_comm *comm = m->get_comm();
  int shared = 1;
  // Grab latest checkpoint information, checks for latest in dist and shared, restarts from most recent between the two. 
  if (comm->am_model_master()) {
    if(m_per_rank_dir.length()){
      snprintf(dir, sizeof(dir), "%s/%s", m_per_rank_dir.c_str(), m_checkpoint_dir.c_str());
      read_latest(dir, "last.distributed.checkpoint", &epoch_dist, &step_dist);
    } 
    if(m_checkpoint_dir.length()){
      strcpy(dir, m_checkpoint_dir.c_str());
      read_latest(dir, "last.shared.checkpoint", &epoch, &step);
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
  // m->get_name() + '.' + std::to_string(comm->get_model_rank()) + '.'
#if 1
  header_t<max_len_dirname> header;

  header.epoch = epoch;
  header.step = step;
  header.shared = shared;
  memcpy(header.dirname, dir, sizeof(dir));

  comm->model_broadcast(0, header);

  epoch = header.epoch;
  step = header.step;
  shared = header.shared;
  memcpy(dir, header.dirname, sizeof(dir));
#else
  comm->model_broadcast(0, epoch);
  comm->model_broadcast(0, step);
  comm->model_broadcast(0, shared);
  comm->model_broadcast(0, &(dir[0]), sizeof(dir));
#endif

  // if we couldn't find the latest epoch, just return
  if (epoch < 0) {
    return false;
  }
  // time how long this takes
  El::Timer timer;
  // let user know we're restarting from a checkpoint
  if (comm->am_model_master()) {
    timer.Start();
    printf("Restart: epoch %d ...\n", epoch);
    fflush(stdout);
  }
  
  char epochdir[1024];
  // Create dir to restart from based off last recorded checkpoint (or overriden values in last.shared[distributed].checkpoint
  if(!shared){
    snprintf(epochdir, sizeof(epochdir), "%s/rank.%d.epoch.%d.step.%d", dir, comm->get_rank_in_model(), epoch, step);
    p.open_restart(epochdir);
    m->load_from_checkpoint_distributed(p);
    p.close_restart();
  }
  else {
    sprintf(epochdir, "%s/shared.model.%d.epoch.%d.step.%d", dir, comm->get_model_rank(), epoch, step);
    if (comm->am_model_master()) {
      p.open_restart(epochdir);
    }
    // Ensure all ranks have access to checkpoint dir, needed for loading rank specific rng state
    comm->model_broadcast(0, &(p.m_checkpoint_dir[0]), sizeof(p.m_checkpoint_dir));
    m->load_from_checkpoint_shared(p);
    if(comm->am_model_master())
      p.close_restart();
  }
  
  // close our checkpoint
  uint64_t bytes_count = p.get_bytes();
  // let user know we've completed reading our restart
  if (comm->am_model_master()) {
    EvalType secs = timer.Stop();
    EvalType bw = 0.0;
    if (secs > 0.0) {
      bw = EvalType(bytes_count) / (secs * 1024.0 * 1024.0);
    }
    printf("Restart complete: Epoch=%d Step=%d (%f secs, %llu bytes, %f MB/sec)\n",
           epoch, step, secs, (unsigned long long) bytes_count, bw
          );
    fflush(stdout);
  }
  p.reset_bytes();
  return true;
}

}
