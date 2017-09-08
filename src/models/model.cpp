////////////////////////////////////////////////////////////////////////////////
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
// lbann_model .hpp .cpp - Abstract class for neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/model.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/layers/io/input/input_layer.hpp"
#include <string>
#include <unistd.h>

#include "mpi.h"

namespace lbann {

model::model(lbann_comm *comm, int mini_batch_size,
             objective_functions::objective_function *obj_fn,
             optimizer_factory *optimizer_fac) :
  m_obj_fn(obj_fn),
  m_execution_mode(execution_mode::invalid),
  m_terminate_training(false),
  m_current_epoch(0), m_current_step(0),
  m_current_validation_step(0), m_current_testing_step(0),
  m_max_mini_batch_size(mini_batch_size),
  m_current_mini_batch_size(mini_batch_size),
  m_effective_mini_batch_size(mini_batch_size),
  m_current_phase(0),
  m_comm(comm),
  m_checkpoint_dir(""), m_checkpoint_epochs(0), m_checkpoint_steps(0),
  m_checkpoint_secs(0.0), m_checkpoint_last(MPI_Wtime()),
  m_optimizer_fac(optimizer_fac) {}

model::model(const model& other) :
  m_execution_mode(other.m_execution_mode),
  m_terminate_training(other.m_terminate_training),
  m_current_epoch(other.m_current_epoch),
  m_current_step(other.m_current_step),
  m_current_validation_step(other.m_current_validation_step),
  m_current_testing_step(other.m_current_testing_step),
  m_max_mini_batch_size(other.m_max_mini_batch_size),
  m_current_mini_batch_size(other.m_current_mini_batch_size),
  m_effective_mini_batch_size(other.m_effective_mini_batch_size),
  m_current_phase(other.m_current_phase),
  m_comm(other.m_comm),
  m_checkpoint_dir(other.m_checkpoint_dir),
  m_checkpoint_epochs(other.m_checkpoint_epochs),
  m_checkpoint_steps(other.m_checkpoint_steps),
  m_checkpoint_secs(other.m_checkpoint_secs),
  m_checkpoint_last(other.m_checkpoint_last),
// Don't need to deep-copy the factory.
  m_optimizer_fac(other.m_optimizer_fac) {
  for (const auto& metric : other.m_metrics) {
    metrics::metric* m_copy = metric->copy();
    m_copy->m_neural_network_model = this;
    m_metrics.push_back(m_copy);
  }
  m_obj_fn = other.m_obj_fn->copy();
  for (const auto& cb : other.m_callbacks) {
    m_callbacks.push_back(cb->copy());
  }
}

model& model::operator=(const model& other) {
  m_execution_mode = other.m_execution_mode;
  m_terminate_training = other.m_terminate_training;
  m_current_epoch = other.m_current_epoch;
  m_current_step = other.m_current_step;
  m_current_validation_step = other.m_current_validation_step;
  m_current_testing_step = other.m_current_testing_step;
  m_max_mini_batch_size = other.m_max_mini_batch_size;
  m_current_mini_batch_size = other.m_current_mini_batch_size;
  m_effective_mini_batch_size = other.m_effective_mini_batch_size;
  m_current_phase = other.m_current_phase;
  m_comm = other.m_comm;
  m_checkpoint_dir = other.m_checkpoint_dir;
  m_checkpoint_epochs = other.m_checkpoint_epochs;
  m_checkpoint_steps = other.m_checkpoint_steps;
  m_checkpoint_secs = other.m_checkpoint_secs;
  m_checkpoint_last = other.m_checkpoint_last;
  m_optimizer_fac = other.m_optimizer_fac;
  for (const auto& metric : other.m_metrics) {
    metrics::metric* m_copy = metric->copy();
    m_copy->m_neural_network_model = this;
    m_metrics.push_back(m_copy);
  }
  m_obj_fn = other.m_obj_fn->copy();
  for (const auto& cb : other.m_callbacks) {
    m_callbacks.push_back(cb->copy());
  }
  return *this;
}

model::~model() {
  if (m_obj_fn) delete m_obj_fn;
  // Free metrics.
  for (metrics::metric *m : get_metrics()) {
    if (m) delete m;
  }
  // Free callbacks.
  for (lbann_callback *c : m_callbacks) {
    if(c) delete c;
  }
}

void model::add_callback(lbann_callback *cb) {
  m_callbacks.push_back(cb);
}

void model::setup_callbacks() {
  for (auto&& cb : m_callbacks) {
    cb->setup(this);
  }
}

void model::add_metric(metrics::metric *m) {
  m_metrics.push_back(m);
}

void model::do_train_begin_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_train_begin(this);
  }
}

void model::do_train_end_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_train_end(this);
  }
}

void model::do_phase_end_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_phase_end(this);
  }
}

void model::do_epoch_begin_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_epoch_begin(this);
  }
}

void model::do_epoch_end_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_epoch_end(this);
  }
}

void model::do_batch_begin_cbs() {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_begin(this);
    }
  }
}

void model::do_batch_end_cbs() {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_end(this);
    }
  }
}

void model::do_test_begin_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_test_begin(this);
  }
}

void model::do_test_end_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_test_end(this);
  }
}

void model::do_validation_begin_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_validation_begin(this);
  }
}

void model::do_validation_end_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_validation_end(this);
  }
}

void model::do_model_forward_prop_begin_cbs() {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_forward_prop_begin(this);
    }
  }
}

void model::do_layer_forward_prop_begin_cbs(Layer *l) {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_forward_prop_begin(this, l);
    }
  }
}

void model::do_model_forward_prop_end_cbs() {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_forward_prop_end(this);
    }
  }
}

void model::do_layer_forward_prop_end_cbs(Layer *l) {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_forward_prop_end(this, l);
    }
  }
}

void model::do_model_backward_prop_begin_cbs() {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_begin(this);
    }
  }
}

void model::do_layer_backward_prop_begin_cbs(Layer *l) {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_begin(this, l);
    }
  }
}

void model::do_model_backward_prop_end_cbs() {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_end(this);
    }
  }
}

void model::do_layer_backward_prop_end_cbs(Layer *l) {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_backward_prop_end(this, l);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Evaluation callbacks
////////////////////////////////////////////////////////////////////////////////

void model::do_batch_evaluate_begin_cbs() {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_evaluate_begin(this);
    }
  }
}

void model::do_batch_evaluate_end_cbs() {
  for (auto&& cb : m_callbacks) {
    if (get_cur_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_evaluate_end(this);
    }
  }
}

void model::do_model_evaluate_forward_prop_begin_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_evaluate_forward_prop_begin(this);
  }
}

void model::do_layer_evaluate_forward_prop_begin_cbs(Layer *l) {
  for (auto&& cb : m_callbacks) {
    cb->on_evaluate_forward_prop_begin(this, l);
  }
}

void model::do_model_evaluate_forward_prop_end_cbs() {
  for (auto&& cb : m_callbacks) {
    cb->on_evaluate_forward_prop_end(this);
  }
}

void model::do_layer_evaluate_forward_prop_end_cbs(Layer *l) {
  for (auto&& cb : m_callbacks) {
    cb->on_evaluate_forward_prop_end(this, l);
  }
}

/** \brief Returns true if a checkpoint should be taken, false otherwise */
bool model::need_checkpoint() {
  /* TODO: since we're using clocks, this requires a bcast for each call,
   * we could use number of samples processed to make a local decision */

  // if none of our checkpoint conditions are set, assume we're not checkpointing
  if (m_checkpoint_epochs == 0 &&
      m_checkpoint_steps  == 0 &&
      m_checkpoint_secs   == 0.0) {
    return false;
  }

  // assume that we won't checkpoint
  int flag = 0;

  // if at start of epoch and evenly divide
  if (flag == 0 && m_checkpoint_epochs > 0) {
    if (at_epoch_start()) {
      flag = (int) (m_current_epoch % m_checkpoint_epochs == 0);
    }
  }

  // if our current step is evenly divisable by checkpoint steps,
  // take a checkpoint
  if (flag == 0 && m_checkpoint_steps > 0) {
    flag = (int) (m_current_step % m_checkpoint_steps == 0);
  }

  // check the clock if time-based checkpoint is enabled
  if (flag == 0 && m_checkpoint_secs != 0.0) {
    // have rank 0 determine whether we should checkpoint
    // to avoid issues with clock skew, we rely on rank 0 to make decision
    if (m_comm->am_world_master()) {
      // get the current time
      double current = MPI_Wtime();

      // compute time next checkpoint is due
      double next = m_checkpoint_last + m_checkpoint_secs;

      // determine whether it's time for a checkpoint
      flag = (current >= next);
    }

    // get flag from rank 0
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return (bool)flag;
}

/** \brief Writes a "latest" file which records epoch number and sample offset for the most recent checkpoint */
static bool write_latest(const char *dir, const char *name, int epoch, int train) {
  // define filename
  char filename[1024];
  sprintf(filename, "%s/%s", dir, name);

  // open the file for writing
  int fd = openwrite(filename);
  if (fd != -1) {
    write_uint32(fd, "epoch", (uint32_t)epoch);
    write_uint32(fd, "train", (uint32_t)train);

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
    uint32_t epoch;
    read_uint32(fd, "epoch", &epoch);
    *epochLast = (int) epoch;

    // read epoch from file
    uint32_t train;
    read_uint32(fd, "train", &train);
    *trainLast = train;

    // close our file
    closeread(fd, filename);
  }

  return true;
}

struct lbann_checkpoint {
  int epoch; // current epoch number
  int step;  // current offset into list of training example indices array
  float learning_rate; // current learning rate
};

//bool model::checkpointShared(TrainingParams& trainParams)
bool model::checkpointShared() {
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0) {
    return false;
  }

  // time how long this takes
  Timer timer;

  // get checkpoint directory
  const char *dir = m_checkpoint_dir.c_str();

  // read current epoch and step counters from model
  int epoch = m_current_epoch;
  int step  = m_current_step;

  // let user know we're saving a checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    timer.Start();
    printf("Checkpoint: epoch %d step %d ...\n", epoch, step);
    fflush(stdout);
  }

  // create top level directory
  //const char* dir = trainParams.ParameterDir.c_str();
  makedir(dir);

  // create subdirectory for this epoch
  char epochdir[1024];
  snprintf(epochdir, sizeof(epochdir), "%s/shared.epoch.%d.step.%d", dir, epoch, step);

  // start our checkpoint
  persist p;
  p.open_checkpoint(epochdir);

  // call virtual function to checkpoint model state
  this->save_to_checkpoint_shared(p);

  // close our checkpoint
  p.close_checkpoint();

  uint64_t bytes_count = p.get_bytes();

  // write epoch number to current file, we do this at the end so as to only update
  // this file when we know we have a new valid checkpoint
  if (m_comm->am_world_master()) {
    write_latest(dir, "shared.last", epoch, step);
  }

  // stop timer and report cost
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    double secs = timer.Stop();
    double bw = 0.0;
    if (secs > 0.0) {
      bw = ((double) bytes_count) / (secs * 1024.0 * 1024.0);
    }
    printf("Checkpoint complete: Epoch=%d Step=%d (%f secs, %llu bytes, %f MB/sec)\n",
           epoch, step, secs, (unsigned long long) bytes_count, bw
          );
    fflush(stdout);
  }

  // saved a checkpoint, update our last checkpoint time
  m_checkpoint_last = MPI_Wtime();

  return true;
}

bool model::restartShared() {
  // if the checkpoint directory is not defined, bail
  if (m_checkpoint_dir.length() == 0) {
    return false;
  }

  // get top level directory
  const char *dir = m_checkpoint_dir.c_str();

  // read epoch number from current file
  int epoch, step;
  if (m_comm->am_world_master()) {
    read_latest(dir, "shared.last", &epoch, &step);
  }
  MPI_Bcast(&epoch, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&step,  1, MPI_INT, 0, MPI_COMM_WORLD);

  // if we couldn't find the latest epoch, just return
  if (epoch < 0) {
    return false;
  }

  // time how long this takes
  Timer timer;

  // let user know we're restarting from a checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    timer.Start();
    printf("Restart: epoch %d ...\n", epoch);
    fflush(stdout);
  }

  // get subdirectory for this epoch
  char epochdir[1024];
  sprintf(epochdir, "%s/shared.epoch.%d.step.%d", dir, epoch, step);

  // open our checkpoint
  persist p;
  p.open_restart(epochdir);

  // call virtual function to restore model from checkpoint
  this->load_from_checkpoint_shared(p);

  // close our checkpoint
  p.close_restart();

  uint64_t bytes_count = p.get_bytes();

  // let user know we've completed reading our restart
  MPI_Barrier(MPI_COMM_WORLD);
  if (m_comm->am_world_master()) {
    double secs = timer.Stop();
    double bw = 0.0;
    if (secs > 0.0) {
      bw = ((double) bytes_count) / (secs * 1024.0 * 1024.0);
    }
    printf("Restart complete: Epoch=%d Step=%d (%f secs, %llu bytes, %f MB/sec)\n",
           epoch, step, secs, (unsigned long long) bytes_count, bw
          );
    fflush(stdout);
  }

  return true;
}

/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_model_header {
  uint32_t execution_mode;
  uint32_t terminate_training;
  uint64_t current_epoch;
  uint64_t current_step;
  uint32_t current_phase;
};

bool model::save_to_checkpoint_shared(persist& p) {
  // write out fields we need to save for model
  if (p.get_rank() == 0) {
    p.write_uint32(persist_type::train, "execution_mode",     (uint32_t) m_execution_mode);
    p.write_uint32(persist_type::train, "terminate_training", (uint32_t) m_terminate_training);
    p.write_uint64(persist_type::train, "current_epoch",      (uint64_t) m_current_epoch);
    p.write_uint64(persist_type::train, "current_step",       (uint64_t) m_current_step);
    p.write_uint32(persist_type::train, "current_phase",      (uint32_t) m_current_phase);
  }

  return true;
}

bool model::load_from_checkpoint_shared(persist& p) {
  // have rank 0 read the file
  // read state from file
  struct lbann_model_header header;
  if (p.get_rank() == 0) {
    p.read_uint32(persist_type::train, "execution_mode",     &header.execution_mode);
    p.read_uint32(persist_type::train, "terminate_training", &header.terminate_training);
    p.read_uint64(persist_type::train, "current_epoch",      &header.current_epoch);
    p.read_uint64(persist_type::train, "current_step",       &header.current_step);
    p.read_uint32(persist_type::train, "current_phase",      &header.current_phase);
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

  // set our member params from values read from disk
  m_execution_mode     = (execution_mode) header.execution_mode;
  m_terminate_training = (bool)           header.terminate_training;
  m_current_epoch      = (int)            header.current_epoch;
  m_current_step       = (int)            header.current_step;
  m_current_phase      =                  header.current_phase;

  return true;
}

}  // namespace lbann
