//////////////////////////////////////////////////////////////////////////////
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
#ifndef LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/visitor_hooks.hpp"

namespace lbann {

// Forward-declarations
class TrainingAlgorithm;

namespace callback {

enum class callback_phase
{
  batch,
  epoch,
  validation,
  inference,
  invalid
};

/** @brief Checkpoint at given interval in given directory */
class checkpoint : public callback_base
{
public:
  /** @brief Construct the checkpoint callback
   *
   *  It may be beneficial to the distributed checkpoints at a higher
   *  tempo than the shared checkpoints because they are less
   *  expensive.
   *
   *  @param checkpoint_dir directory to save checkpoint files
   *  @param restart_dir directory to find checkpoint files
   *  @param checkpoint_epochs interval to checkpoint
   *  @param checkpoint_steps interval to checkpoint
   *  @param checkpoint_secs interval to checkpoint
   *  @param per_rank_dir The directory into which to dump distributed
   * checkpoints
   *  @param ckpt_dist_epochs The frequency of distributed checkpoints in epochs
   *  @param ckpt_dist_steps The frequence of distributed checkpoints in steps
   */
  checkpoint(std::string checkpoint_dir,
             std::string restart_dir,
             int checkpoint_epochs,
             int checkpoint_steps,
             int checkpoint_secs,
             std::string per_rank_dir,
             int ckpt_dist_epochs,
             int ckpt_dist_steps)
    : callback_base(),
      m_active_trainer(nullptr),
      m_active_training_algorithm(nullptr),
      m_checkpoint_dir(std::move(checkpoint_dir)),
      m_restart_dir(std::move(restart_dir)),
      m_checkpoint_epochs(checkpoint_epochs),
      m_checkpoint_steps(checkpoint_steps),
      m_checkpoint_secs(checkpoint_secs),
      m_per_rank_dir(per_rank_dir),
      m_ckpt_dist_epochs(ckpt_dist_epochs),
      m_ckpt_dist_steps(ckpt_dist_steps)
  {}
  checkpoint(const checkpoint&) = default;
  checkpoint& operator=(const checkpoint&) = default;
  checkpoint* copy() const override { return new checkpoint(*this); }
  void on_setup_begin(model* m) override;
  void setup(trainer* t) override;
  void on_train_begin(model* m) override;
  void on_train_end(model* m) override;
  void on_epoch_begin(model* m) override;
  void on_batch_begin(model* m) override;
  void on_validation_begin(model* m) override;

  inline void set_checkpoint_dir(const std::string& dir)
  {
    m_checkpoint_dir = dir;
  }

  inline const std::string& get_checkpoint_dir() { return m_checkpoint_dir; }

  inline void set_restart_dir(const std::string& dir) { m_restart_dir = dir; }

  inline const std::string& get_restart_dir()
  {
    // If the restart directory has been explicitly defined use that
    if (m_restart_dir.length() != 0) {
      return m_restart_dir;
    }
    else {
      return m_checkpoint_dir;
    }
  }

  inline void set_active_trainer(trainer* t) { m_active_trainer = t; }

  trainer& get_active_trainer();

  inline void set_active_training_algorithm(TrainingAlgorithm* t)
  {
    m_active_training_algorithm = t;
  }

  TrainingAlgorithm& get_active_training_algorithm();

  inline void set_checkpoint_epochs(int epochs)
  {
    m_checkpoint_epochs = epochs;
  }

  inline void set_checkpoint_steps(int steps) { m_checkpoint_steps = steps; }

  inline void set_checkpoint_secs(EvalType secs) { m_checkpoint_secs = secs; }

  inline void set_per_rank_dir(std::string dir) { m_per_rank_dir = dir; }

  inline const std::string& get_per_rank_dir() { return m_per_rank_dir; }

  inline void set_ckpt_dist_epochs(int ckpt_dist_epochs)
  {
    m_ckpt_dist_epochs = ckpt_dist_epochs;
  }

  inline void set_ckpt_dist_steps(int ckpt_dist_steps)
  {
    m_ckpt_dist_steps = ckpt_dist_steps;
  }

  inline std::string get_shared_checkpoint_rootdir()
  {
    return get_restart_dir();
  }

  /// @todo BVE FIMME this looks wrong  I think that the order
  /// should be reversed
  inline std::string get_distributed_checkpoint_rootdir()
  {
    if (m_per_rank_dir.length()) {
      return get_per_rank_dir() + "/" + get_restart_dir();
    }
    else {
      return get_restart_dir();
    }
  }

  bool need_checkpoint(model* m, callback_phase phase);
  std::string find_latest_checkpoint(lbann_comm& comm,
                                     const std::string& trainer_name,
                                     const std::string& alg_name,
                                     visitor_hook& hook,
                                     execution_mode& mode,
                                     size_t& epoch,
                                     size_t& step,
                                     bool& shared);
  bool open_latest_checkpoint(
    lbann_comm& comm,
    const std::string& task_label,
    const std::string& trainer_name,
    const std::string& alg_name,
    std::function<bool(/*const */ persist&)> reload_shared_ckpt,
    std::function<bool(/*const */ persist&)> reload_distributed_ckpt);
  bool reload_model(model* m);
  bool reload_trainer(trainer* t);
  bool restart(model* m);
  std::string name() const override { return "checkpoint"; }

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  bool do_checkpoint(model* m, visitor_hook hook);
  void do_distributed_checkpoint(lbann_comm& comm,
                                 trainer& t,
                                 model& m,
                                 visitor_hook hook,
                                 execution_mode mode,
                                 persist& p,
                                 size_t epoch,
                                 size_t step);
  void do_shared_checkpoint(lbann_comm& comm,
                            trainer& t,
                            model& m,
                            visitor_hook hook,
                            execution_mode mode,
                            persist& p,
                            size_t epoch,
                            size_t step);

private:
  trainer* m_active_trainer;
  TrainingAlgorithm* m_active_training_algorithm;
  std::string m_checkpoint_dir;
  // If the restart directory is not explicity set, default to the
  // checkpoint directory
  std::string m_restart_dir;
  int m_checkpoint_epochs;
  int m_checkpoint_steps;
  EvalType m_checkpoint_secs;
  std::string m_per_rank_dir;
  int m_ckpt_dist_epochs;
  int m_ckpt_dist_steps;
  EvalType m_checkpoint_last;
  bool m_checkpoint_dist;
  bool m_checkpoint_shared;

  template <size_t _max_dir_len>
  struct header_t
  {
    visitor_hook hook;
    execution_mode mode;
    int epoch;
    int step;
    int shared;
    char dirname[_max_dir_len];
  };
};

std::string get_trainer_checkpoint_dirname(const std::string& trainer_name,
                                           const std::string& dir);

std::string get_last_shared_checkpoint_filename(const std::string& alg_name,
                                                const std::string& dir);

std::string get_last_shared_checkpoint_filename(const std::string& trainer_name,
                                                const std::string& alg_name,
                                                const std::string& dir);

std::string get_shared_checkpoint_dirname(const std::string& alg_name,
                                          const std::string& dir,
                                          visitor_hook hook,
                                          execution_mode mode,
                                          size_t epoch,
                                          size_t step);

std::string get_shared_checkpoint_dirname(const std::string& trainer_name,
                                          const std::string& alg_name,
                                          const std::string& dir,
                                          visitor_hook hook,
                                          execution_mode mode,
                                          size_t epoch,
                                          size_t step);

std::string
get_last_distributed_checkpoint_filename(const std::string& alg_name,
                                         const std::string& dir);

std::string
get_last_distributed_checkpoint_filename(const std::string& trainer_name,
                                         const std::string& alg_name,
                                         const std::string& dir);

std::string get_distributed_checkpoint_dirname(const std::string& alg_name,
                                               const int rank_in_trainer,
                                               const std::string& dir,
                                               visitor_hook hook,
                                               execution_mode mode,
                                               size_t epoch,
                                               size_t step);

std::string get_distributed_checkpoint_dirname(const std::string& trainer_name,
                                               const std::string& alg_name,
                                               const int rank_in_trainer,
                                               const std::string& dir,
                                               visitor_hook hook,
                                               execution_mode mode,
                                               size_t epoch,
                                               size_t step);

// Print last checkpoint to file, used to determine which checkpoint to load
// from.
bool write_latest(std::string filename,
                  visitor_hook hook,
                  execution_mode mode,
                  size_t epoch,
                  size_t train);

/** \brief Reads the "latest" file and returns the epoch number and
 *        sample offset for most recent checkpoint
 */
bool read_latest(std::string filename,
                 visitor_hook* hook,
                 execution_mode* mode,
                 size_t* epochLast,
                 size_t* trainLast);

// Builder function
std::unique_ptr<callback_base>
build_checkpoint_callback_from_pbuf(const google::protobuf::Message&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED
