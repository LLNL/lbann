////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC. 
// Produced at the Lawrence Livermore National Laboratory. 
// Written by:
//         Brian Van Essen <vanessen1@llnl.gov>
//         Sam Jacobs <jacobs32@llnl.gov>
//         Hyojin Kim <kim63@llnl.gov>
//         Nikoli Dryden <dryden1@llnl.gov>
//         Tim Moon <moon13@llnl.gov>
//
// LLNL-CODE-XXXXXX.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network Toolkit, Version 0.9
//
// lbann_model .hpp .cpp - Abstract class for neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_HPP
#define LBANN_MODEL_HPP

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/utils/lbann_summary.hpp"
#include "lbann/io/lbann_file_io.hpp"
#include <vector>
#include <string>

namespace lbann {

// Forward-declare this.
class lbann_callback;

/**
 * Base class for LBANN models.
 */
class Model {
public:
  Model(lbann_comm* comm);
  virtual ~Model() {}

  /** Initialize the model. */
  virtual void setup() {}

  /** Register a new callback for the model. */
  virtual void add_callback(lbann_callback* cb);

  /** Return the layers this model has. */
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
  inline int64_t get_cur_epoch() const { return cur_epoch; }
  /** Get the current step for the model. */
  inline int64_t get_cur_step() const { return cur_step; }
  /** Get the model's execution mode. */
  inline execution_mode get_execution_mode() const { return m_execution_mode; }

  /** Produce summary information (if any). */
  virtual void summarize(lbann_summary& summarizer) {}

  /** Return true if the flag to stop training is set. */
  bool get_terminate_training() const { return terminate_training; }
  /** Set the terminate training flag (on or off). */
  void set_terminate_training(bool f) { terminate_training = f; }
    
protected:
  /** Communicator for the model. */
  lbann_comm* comm;
  /** Most recent/current epoch for the model. */
  int64_t cur_epoch;
  /** Most recent/current training step for the model. */
  int64_t cur_step;
  /** Current callbacks to process. */
  std::vector<lbann_callback*> callbacks;
  /** Flag telling the model to terminate training. */
  bool terminate_training;
  /** The model's current execution mode. */
  execution_mode m_execution_mode;

  // Methods for calling every callback at different points.
  void setup_callbacks();
  void do_train_begin_cbs();
  void do_train_end_cbs();
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
};

}  // namespace lbann

#endif  // LBANN_MODEL_HPP
