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

void lbann_callback_checkpoint::setup(model *m) {
  m_epoch_end = false;
  m_val_end = false;
  restart(m);
}

void lbann_callback_checkpoint::on_epoch_end(model *m) {
  m_epoch_end = true;
  if(need_checkpoint(m)){
    checkpoint(m);
  }
  m_epoch_end = false;
  
}

void lbann_callback_checkpoint::on_validation_end(model *m) {
  m_val_end = true;
  if(need_checkpoint(m)){
    checkpoint(m);
  }
  m_val_end = false;
}

void lbann_callback_checkpoint::on_batch_end(model *m) {
  if(need_checkpoint(m)){
    checkpoint(m);
  }
}
bool lbann_callback_checkpoint::need_checkpoint(model *m) {
  /* TODO: since we're using clocks, this requires a bcast for each call,
   * we could use number of samples processed to make a local decision */
  // if none of our checkpoint conditions are set, assume we're not checkpointing
  if (m_checkpoint_epochs == 0 &&
      m_checkpoint_steps  == 0 &&
      m_checkpoint_secs   == 0.0 &&
      m_ckpt_dist_epochs == 0) {
    return false;
  }
  // assume that we won't checkpoint
  bool checkpoint_now = false;
  m_checkpoint_shared = false;
  m_checkpoint_dist = false;
  lbann_comm *comm = m->get_comm();
  int cur_epoch = m->get_cur_epoch();
  // if at start of epoch and evenly divide
  if (!checkpoint_now && m_checkpoint_epochs > 0 ) {
    if (m_epoch_end || m_val_end) {
      checkpoint_now = (cur_epoch > 0) && (cur_epoch % m_checkpoint_epochs == 0);
      m_checkpoint_shared = checkpoint_now;  
    }
  }
    
  if(m_ckpt_dist_epochs > 0 && (m_epoch_end || m_val_end)){
      if(cur_epoch % m_ckpt_dist_epochs == 0 && m_checkpoint_shared){
        m_checkpoint_dist = true;
      } else if (cur_epoch % m_ckpt_dist_epochs == 0){
        checkpoint_now = true; 
        m_checkpoint_dist = true; 
      }
  }
  // if our current step is evenly divisable by checkpoint steps,
  // take a checkpoint
  if (!checkpoint_now && m_checkpoint_steps > 0) {
    checkpoint_now = (m->get_cur_step() > 0) && (m->get_cur_step() % m_checkpoint_steps == 0);
    m_checkpoint_shared = checkpoint_now;
  }
  // check the clock if time-based checkpoint is enabled
  if (!checkpoint_now && m_checkpoint_secs != 0.0) {
    // have rank 0 determine whether we should checkpoint
    // to avoid issues with clock skew, we rely on rank 0 to make decision
    if (comm->am_world_master()) {
      // get the current time
      EvalType current = MPI_Wtime();
      // compute time next checkpoint is due
      EvalType next = m_checkpoint_last + m_checkpoint_secs;
      // determine whether it's time for a checkpoint
      checkpoint_now = (current >= next);
      m_checkpoint_shared = checkpoint_now;
    }
    // get checkpoint_now from rank 0
    int flag = checkpoint_now ? 1 : 0;
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    checkpoint_now = (bool) flag;
  }
  return checkpoint_now;
}

static bool write_latest(const char *dir, const char *name, int epoch, int train) {
  // define filename
  char filename[1024];
  sprintf(filename, "%s/%s", dir, name);
  // open the file for writing
  int fd = openwrite(filename);
  if (fd != -1) {
    char field[256];
    sprintf(field, "epoch=%d step=%d\n", epoch, train);
    write_string(fd, "shared.last", field, strlen(field));
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
    read_string(fd, "shared.last", field, sizeof(field));
    int ret = sscanf(field, "epoch=%d step=%d\n", epochLast, trainLast);
    // close our file
    closeread(fd, filename);
    if(ret != 2) { return false; }
  }
  return true;
}

bool lbann_callback_checkpoint::checkpoint(model *m) {
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0 && m_per_rank_dir.length() == 0) {
    return false;
  }
  // time how long this takes
  // read current epoch and step counters from model
  El::Timer timer;
  char epochdir[1024];
  persist p;
  char dir[1024];
  int epoch;
  int step;
  // let user know we're saving a checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  if (p.get_rank() == 0) {
    epoch = m->get_cur_epoch();
    step  = m->get_cur_step();
    timer.Start();
    printf("Checkpoint: epoch %d step %d ...\n", epoch, step);
    fflush(stdout);
  }
  MPI_Bcast(&epoch, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&step,  1, MPI_INT, 0, MPI_COMM_WORLD);
  // if this is a distributed ckpt
  if(m_checkpoint_dist){
    //prepend per rank directory with shared checkpoint dir name
    if(m_per_rank_dir.length() != 0){
      snprintf(dir, sizeof(dir), "%s/%s", m_per_rank_dir.c_str(), m_checkpoint_dir.c_str());
    } else {
      strcpy(dir, m_checkpoint_dir.c_str());
    }
    makedir(dir);
    // create directories per ranks
    snprintf(epochdir, sizeof(epochdir), "%s/rank.%d.epoch.%d.step.%d", dir, p.get_rank(), epoch, step);
    p.open_checkpoint(epochdir,1,m_val_end);
    m->save_to_checkpoint_distributed(p, m_val_end);
    m_checkpoint_dist = false;
    p.close_checkpoint();
    if (p.get_rank() == 0) {
      write_latest(dir, "last.distributed.checkpoint", epoch, step);
    } 
    m_checkpoint_dist = false;  
  }
  // shared checkpoint by default
  if(m_checkpoint_shared){
    strcpy(dir, m_checkpoint_dir.c_str());
     makedir(dir);
     snprintf(epochdir, sizeof(epochdir), "%s/shared.epoch.%d.step.%d", dir, epoch, step);
     p.open_checkpoint(epochdir,0, m_val_end);
     m->save_to_checkpoint_shared(p,m_val_end);
     // close our checkpoint
     p.close_checkpoint();
     if (p.get_rank() == 0) {
       write_latest(dir, "last.shared.checkpoint", epoch, step);
     }
     m_checkpoint_shared = false;
   }
   uint64_t bytes_count = p.get_bytes();
    
   if(p.get_rank() == 0) {
     EvalType secs = timer.Stop();
     EvalType bw = 0;
     if (secs > 0.0) {
       bw = EvalType(bytes_count) / (secs * 1024.0 * 1024.0);
     }
     printf("Checkpoint complete: Epoch=%d Step=%d (%f secs, %llu bytes, %f MB/sec)\n",
            epoch, step, secs, (unsigned long long) bytes_count, bw);
    fflush(stdout);
  }
  m_checkpoint_last = MPI_Wtime();
  return true;
}
bool lbann_callback_checkpoint::restart(model *m) {
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0 &&  m_per_rank_dir.length() == 0) {
    return false;
  }
  // get top level directory
  //const char *dir = m_checkpoint_dir.c_str();
  // read epoch number from current file
  char dir[1024];
  int epoch, step;
  lbann_comm *comm = m->get_comm();

  if(m_per_rank_dir.length() && comm->am_world_master()){
    snprintf(dir, sizeof(dir), "%s/%s", m_per_rank_dir.c_str(), m_checkpoint_dir.c_str());
   read_latest(dir, "last.distributed.checkpoint", &epoch, &step);
  } 
  else if(!m_per_rank_dir.length() && comm->am_world_master()){
    strcpy(dir, m_checkpoint_dir.c_str());
    read_latest(dir, "last.shared.checkpoint", &epoch, &step);
  }

  MPI_Bcast(&epoch, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&step,  1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dir,  sizeof(dir), MPI_CHAR, 0, MPI_COMM_WORLD);
  // if we couldn't find the latest epoch, just return
  if (epoch < 0) {
    return false;
  }
  // time how long this takes
  El::Timer timer;
  // let user know we're restarting from a checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  if (comm->am_world_master()) {
    timer.Start();
    printf("Restart: epoch %d ...\n", epoch);
    fflush(stdout);
  }
  
  char epochdir[1024];
  persist p;
  
  if(m_per_rank_dir.length() != 0){
    snprintf(epochdir, sizeof(epochdir), "%s/rank.%d.epoch.%d.step.%d", dir, p.get_rank(), epoch, step);
    p.open_restart(epochdir,1);
    m->load_from_checkpoint_distributed(p);
  }
  else{
    sprintf(epochdir, "%s/shared.epoch.%d.step.%d", dir, epoch, step);
    p.open_restart(epochdir,0);
    m->load_from_checkpoint_shared(p);
  }
  
  // close our checkpoint
  p.close_restart();
  uint64_t bytes_count = p.get_bytes();
  // let user know we've completed reading our restart
  MPI_Barrier(MPI_COMM_WORLD);
  if (comm->am_world_master()) {
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
  return true;
}


}
