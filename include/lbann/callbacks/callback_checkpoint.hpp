//////////////////////////////////////////////////////////////////////////////
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
// lbann_callback_checkpoint .hpp .cpp - Callback hooks to checkpoint model
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/io/persist.hpp"

namespace lbann {

/** @brief Checkpoint at given interval in given directory */
class lbann_callback_checkpoint : public lbann_callback {
 public:

  /** @brief Construct the checkpoint callback
   *
   *  It may be beneficial to the distributed checkpoints at a higher
   *  tempo than the shared checkpoints because they are less
   *  expensive.
   *
   *  @param checkpoint_dir directory to save checkpoint files
   *  @param checkpoint_epochs interval to checkpoint
   *  @param checkpoint_steps interval to checkpoint
   *  @param checkpoint_secs interval to checkpoint
   *  @param per_rank_dir The directory into which to dump distributed checkpoints
   *  @param ckpt_dist_epochs The frequency of distributed checkpoints in epochs
   *  @param ckpt_dist_steps The frequence of distributed checkpoints in steps
   */
  lbann_callback_checkpoint(std::string checkpoint_dir,
                            int checkpoint_epochs,
                            int checkpoint_steps,
                            int checkpoint_secs,
                            std::string per_rank_dir,
                            int ckpt_dist_epochs,
                            int ckpt_dist_steps) :
    lbann_callback(),
    m_checkpoint_dir(checkpoint_dir),
    m_checkpoint_epochs(checkpoint_epochs),
    m_checkpoint_steps(checkpoint_steps),
    m_checkpoint_secs(checkpoint_secs),
    m_per_rank_dir(per_rank_dir),
    m_ckpt_dist_epochs(ckpt_dist_epochs),
    m_ckpt_dist_steps(ckpt_dist_steps) {}
  lbann_callback_checkpoint(const lbann_callback_checkpoint&) = default;
  lbann_callback_checkpoint& operator=(const lbann_callback_checkpoint&) = default;
  lbann_callback_checkpoint* copy() const override { return new lbann_callback_checkpoint(*this); }
  void setup(model *m) override;
  void on_epoch_end(model *m) override;
  void on_batch_end(model *m) override;
  void on_validation_end(model *m) override;

  inline void set_checkpoint_dir(std::string dir){
    m_checkpoint_dir= dir;
  }

  inline void set_checkpoint_epochs(int epochs){
    m_checkpoint_epochs= epochs;
  }

  inline void set_checkpoint_steps(int steps){
    m_checkpoint_steps= steps;
  }

  inline void set_checkpoint_secs(EvalType secs){
    m_checkpoint_secs= secs;
  }

  inline void set_per_rank_dir(std::string dir){
    m_per_rank_dir = dir;
  }

  inline void set_ckpt_dist_epochs(int ckpt_dist_epochs){
    m_ckpt_dist_epochs = ckpt_dist_epochs;
  }

  inline void set_ckpt_dist_steps(int ckpt_dist_steps){
    m_ckpt_dist_steps = ckpt_dist_steps;
  }

  bool need_checkpoint(model *m);
  bool checkpoint(model *m);
  bool restart(model *m);
  std::string name() const override { return "checkpoint"; }
 protected:
  std::string m_checkpoint_dir;
  int m_checkpoint_epochs;
  int m_checkpoint_steps;
  EvalType m_checkpoint_secs;
  std::string m_per_rank_dir;
  int m_ckpt_dist_epochs;
  int m_ckpt_dist_steps;
  EvalType m_checkpoint_last;
  persist p;
  bool m_checkpoint_dist;
  bool m_checkpoint_shared;

  template<size_t _max_dir_len>
  struct header_t {
    int epoch;
    int step;
    int shared;
    char dirname[_max_dir_len];
  };
};

static inline std::string get_last_shared_checkpoint_filename(model *m, std::string dir) {
  lbann_comm *comm = m->get_comm();
  std::stringstream ss;
  ss << dir << "/";
  ss << m->get_name().c_str() << ".";
  ss << comm->get_trainer_rank() << ".last.shared.checkpoint";
  return ss.str();
}

static inline std::string get_shared_checkpoint_dirname(model *m, std::string dir, int epoch, int step) {
  lbann_comm *comm = m->get_comm();
  std::stringstream ss;
  ss << dir << "/" << m->get_name().c_str();
  ss << "." << comm->get_trainer_rank();
  ss << ".shared.epoch." << epoch;
  ss << ".step."<< step << "/";
  return ss.str();
}

static inline std::string get_last_distributed_checkpoint_filename(model *m, std::string dir) {
  lbann_comm *comm = m->get_comm();
  std::stringstream ss;
  ss << dir << "/";
  ss << m->get_name().c_str() << ".";
  ss << comm->get_trainer_rank() << ".last.distributed.checkpoint";
  return ss.str();
}

static inline std::string get_distributed_checkpoint_dirname(model *m, std::string dir, int epoch, int step) {
  lbann_comm *comm = m->get_comm();
  std::stringstream ss;
  ss << dir << "/" << m->get_name().c_str();
  ss << "." << comm->get_trainer_rank();
  ss << ".rank." << comm->get_rank_in_trainer();
  ss << ".epoch." << epoch;
  ss << ".step."<< step << "/";
  return ss.str();
}

// Print last checkpoint to file, used to determine which checkpoint to load from.
static inline bool write_latest(std::string filename, int epoch, int train) {
  // open the file for writing
  int fd = openwrite(filename.c_str());
  if (fd != -1) {
    char field[256];
    sprintf(field, "epoch=%d step=%d\n", epoch, train);
    write_string(fd, filename.c_str(), field, strlen(field));
    // close our file
    closewrite(fd, filename.c_str());
  }
  return true;
}

/** \brief Reads the "latest" file and returns the epoch number and
 *        sample offset for most recent checkpoint
 */
static inline bool read_latest(std::string filename, int *epochLast, int *trainLast) {
  // assume we don't have a file, we'll return -1 in that case
  *epochLast = -1;
  *trainLast = -1;
  // open the file for reading
  int fd = openread(filename.c_str());
  if (fd != -1) {
    // read epoch from file
    char field[256];
    read_string(fd, filename.c_str(), field, sizeof(field));
    int ret = sscanf(field, "epoch=%d step=%d\n", epochLast, trainLast);
    // close our file
    closeread(fd, filename.c_str());
    if(ret != 2) { return false; }
  }
  return true;
}

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED
