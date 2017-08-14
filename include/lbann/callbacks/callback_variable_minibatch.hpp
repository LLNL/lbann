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
// lbann_variable_minibatch .hpp .cpp - Callback for variable-size mini-batches
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_VARIABLE_MINIBATCH_HPP_INCLUDED
#define LBANN_CALLBACKS_VARIABLE_MINIBATCH_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Support changing the mini-batch size on different schedules.
 * Implementations should override implement the abstract methods to define
 * concrete schedules.
 */
class lbann_callback_variable_minibatch : public lbann_callback {
 public:
  lbann_callback_variable_minibatch(int starting_mbsize);
  lbann_callback_variable_minibatch(
    const lbann_callback_variable_minibatch&) = default;
  lbann_callback_variable_minibatch& operator=(
    const lbann_callback_variable_minibatch&) = default;
  /// Set the initial mini-batch size.
  void on_train_begin(model *m);
  /// Potentially change the mini-batch size.
  void on_epoch_end(model *m);
 protected:
  /** Return the new mini-batch size to use. */
  virtual int change_minibatch_size(model *m) = 0;
  /** Return a new learning rate to use. */
  virtual float change_learning_rate(model *m, float cur_lr, int old_mbsize,
                                     int new_mbsize) = 0;
  /// Initial mini-batch size.
  int m_starting_mbsize;
  /**
   * The current mini-batch size for this epoch.
   * This is kept separately from the model's get_current_mini_batch_size()
   * method, as calling that in on_epoch_end returns the size of the last mini-
   * batch, not the "base" mini-batch.
   */
  int m_current_mini_batch_size;
};

/**
 * Double the mini-batch size every set number of epochs.
 * Also doubles the learning rate.
 */
class lbann_callback_step_minibatch : public lbann_callback_variable_minibatch {
 public:
  lbann_callback_step_minibatch(int starting_mbsize, int step);
  lbann_callback_step_minibatch(const lbann_callback_step_minibatch&) = default;
  lbann_callback_step_minibatch& operator=(
    const lbann_callback_step_minibatch&) = default;
  lbann_callback_step_minibatch* copy() const {
    return new lbann_callback_step_minibatch(*this);
  }
  std::string name() const { return "step minibatch"; }
 protected:
  int change_minibatch_size(model *m);
  float change_learning_rate(model *m, float cur_lr, int old_mbsize,
                             int new_mbsize);
  /// Number of epochs between step changes.
  int m_step;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_VARIABLE_MINIBATCH_HPP_INCLUDED
