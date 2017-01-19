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
// lbann_model .hpp .cpp - Abstract class for neural network training models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_HPP
#define LBANN_MODEL_HPP

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/utils/lbann_summary.hpp"
#include "lbann/io/lbann_file_io.hpp"
#include "lbann/io/lbann_persist.hpp"
#include "lbann/objective_functions/lbann_objective_fn.hpp"
#include <vector>
#include <string>

namespace lbann {

// Forward-declare this.
class lbann_callback;

/**
 * Base class for LBANN models.
 */
class model {
public:
  model(lbann_comm* comm, objective_fn* obj_fn);
  virtual ~model() {}

  /** Initialize the model. */
  virtual void setup() {}

  /** Register a new callback for the model. */
  virtual void add_callback(lbann_callback* cb);

  /** Return the model's layers. */
  virtual std::vector<Layer*>& get_layers() = 0;

  /** Get the most recent training accuracy. */
  virtual DataType get_train_accuracy() const = 0;
  /** Get the most recent validation accuracy. */
  virtual DataType get_validate_accuracy() const = 0;
  /** Get the most recent test accuracy. */
  virtual DataType get_test_accuracy() const = 0;

  /** Get the model's comm. */
  inline lbann_comm* get_comm() const { return comm; }
  /** Get the current epoch for the model. */
  inline int64_t get_cur_epoch() const { return m_current_epoch; }
  /** Get the current step for the model. */
  inline int64_t get_cur_step() const { return m_current_step; }
  /** Get the current validation step for the model. */
  inline int64_t get_cur_validation_step() const { return m_current_validation_step; }
  /** Get the current testing step for the model. */
  inline int64_t get_cur_testing_step() const { return m_current_testing_step; }
  /** Get the model's execution mode. */
  inline execution_mode get_execution_mode() const { return m_execution_mode; }
  inline int64_t set_current_mini_batch_size(int64_t mini_batch_size)
  { m_current_mini_batch_size = mini_batch_size; return m_current_mini_batch_size; }
  inline int64_t get_current_mini_batch_size() { return m_current_mini_batch_size; }
  /** Get the current phase (multiple epochs) in layer-wise model training. */
  inline size_t get_current_phase() { return m_current_phase; }

  /** Produce summary information (if any). */
  virtual void summarize(lbann_summary& summarizer) {}

  /** Return true if the flag to stop training is set. */
  bool get_terminate_training() const { return m_terminate_training; }
  /** Set the terminate training flag (on or off). */
  void set_terminate_training(bool f) { m_terminate_training = f; }

  /** Return true if about to start a new training epoch */
  virtual bool at_epoch_start() = 0;

  objective_fn* obj_fn;

  /** Set checkpoint values */
  inline void set_checkpoint_dir(std::string dir)   { m_checkpoint_dir    = dir;    }
  inline void set_checkpoint_epochs(int64_t epochs) { m_checkpoint_epochs = epochs; }
  inline void set_checkpoint_steps(int64_t steps)   { m_checkpoint_steps  = steps;  }
  inline void set_checkpoint_secs(double secs)      { m_checkpoint_secs   = secs;   }

  /** Returns true if a checkpoint should be taken, false otherwise */
  bool need_checkpoint();

  /** Checkpoint model to given file descriptor, return number of bytes written */
  virtual bool save_to_checkpoint_shared(persist& p);
  /** Restore model by reading checkpoint from given file descriptor, return number of bytes read */
  virtual bool load_from_checkpoint_shared(persist& p);

  /*! Top-level call to start checkpoint.  This creates the persist object
   *  and then calls the model's save_to_checkpoint_shared() virtual function */ 
  bool checkpointShared();

  /*! Top-level call to restart.  This creates the persist object
   *  and then calls the model's load_from_checkpoint_shared() virtual function */ 
  bool restartShared();
    
protected:
  /** The model's current execution mode. */
  execution_mode m_execution_mode;
  /** Flag telling the model to terminate training. */
  bool m_terminate_training;
  /** Most recent/current epoch for the model. */
  int64_t m_current_epoch;
  /** Most recent/current training step for the model. */
  int64_t m_current_step;
  int64_t m_current_validation_step;
  int64_t m_current_testing_step;
  /** Size of the current mini-batch */
  int64_t m_current_mini_batch_size;
  /** current phase (multiple of epoch counts) in training a model */
  size_t m_current_phase;
  /** Communicator for the model. */
  lbann_comm* comm;
  /** Global rank of process in MPI_COMM_WORLD */
  int m_rank;
  /** Size of MPI_COMM_WORLD */
  int m_ranks;
  /** Current callbacks to process. */
  std::vector<lbann_callback*> callbacks;

  /** Directory where we should save checkpoints */
  std::string m_checkpoint_dir;
  /** Number of training steps to elapse between checkpoints */
  int64_t m_checkpoint_epochs;
  /** Number of training steps to elapse between checkpoints */
  int64_t m_checkpoint_steps;
  /** Number of seconds to elapse between checkpoints (checkpoint interval) */
  double m_checkpoint_secs;
  /** Timestamp of last checkpoint */
  double m_checkpoint_last;

  // Methods for calling every callback at different points.
  void setup_callbacks();
  void do_train_begin_cbs();
  void do_train_end_cbs();
  void do_phase_end_cbs();
  void do_epoch_begin_cbs();
  void do_epoch_end_cbs();
  void do_batch_begin_cbs();
  void do_batch_end_cbs();
  void do_test_begin_cbs();
  void do_test_end_cbs();
  void do_validation_begin_cbs();
  void do_validation_end_cbs();
  void do_model_forward_prop_begin_cbs();
  void do_layer_forward_prop_begin_cbs(Layer* l);
  void do_model_forward_prop_end_cbs();
  void do_layer_forward_prop_end_cbs(Layer* l);
  void do_model_backward_prop_begin_cbs();
  void do_layer_backward_prop_begin_cbs(Layer* l);
  void do_model_backward_prop_end_cbs();
  void do_layer_backward_prop_end_cbs(Layer* l);
  /// Evaluation phases (validation / testing)
  void do_batch_evaluate_begin_cbs();
  void do_batch_evaluate_end_cbs();
  void do_model_evaluate_forward_prop_begin_cbs();
  void do_layer_evaluate_forward_prop_begin_cbs(Layer* l);
  void do_model_evaluate_forward_prop_end_cbs();
  void do_layer_evaluate_forward_prop_end_cbs(Layer* l);
};

}  // namespace lbann

#endif  // LBANN_MODEL_HPP
