////////////////////////////////////////////////////////////////////////////////
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
  void on_train_begin(model *m) override;
  /// Potentially change the mini-batch size.
  void on_epoch_end(model *m) override;
 protected:
  /**
   * Implemented by child classes to provide the mini-batch/learning schedule.
   * This is called at the end of every training epoch. If it returns false,
   * no changes are made from the currently established schedule.
   * If this returns true, the mini-batch size will be changed accordingly.
   * If the mini-batch size is larger than the model's maximum mini-batch size,
   * a warning is printed and the maximum mini-batch size is used.
   * If new_lr also non-zero, the learning rate will be changed to new_lr,
   * with a linear ramp time. (If ramp_time is 0, it is changed immediately.)
   * Note changing the learning rate while in a ramp may lead to unexpected
   * behavior; also be aware of interactions with other learning rate
   * schedules.
   */
  virtual bool schedule(model *m, int& new_mbsize, float& new_lr,
                        int& ramp_time) = 0;
  /// Change the learning rate of every layer in m to new_lr.
  void change_learning_rate(model *m, float new_lr) const;
  /// Get the current learning rate (assumes every layer has the same one).
  float get_current_learning_rate(model *m) const;

  /// Initial mini-batch size.
  int m_starting_mbsize;
  /**
   * The current mini-batch size for this epoch.
   * This is kept separately from the model's get_current_mini_batch_size()
   * method, as calling that in on_epoch_end returns the size of the last mini-
   * batch, not the "base" mini-batch.
   */
  int m_current_mini_batch_size;
  /// Current number of epochs left to ramp the learning rate.
  int m_ramp_count = 0;
  /// Amount to increment the learning rate by when ramping.
  float m_lr_incr = 0.0f;
};

/**
 * Double the mini-batch size every set number of epochs.
 * Also doubles the learning rate.
 */
class lbann_callback_step_minibatch : public lbann_callback_variable_minibatch {
 public:
  lbann_callback_step_minibatch(int starting_mbsize, int step,
                                int ramp_time = 0);
  lbann_callback_step_minibatch(const lbann_callback_step_minibatch&) = default;
  lbann_callback_step_minibatch& operator=(
    const lbann_callback_step_minibatch&) = delete;
  lbann_callback_step_minibatch* copy() const override {
    return new lbann_callback_step_minibatch(*this);
  }
  std::string name() const override { return "step minibatch"; }
 protected:
  bool schedule(model *m, int& new_mbsize, float& new_lr, int& ramp_time) override;

 private:
  /// Number of epochs between mini-batch size increases.
  int m_step;
  /// Number of steps to ramp the learning rate over.
  int m_ramp_time;
};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_step_minibatch_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

class lbann_callback_minibatch_schedule : public lbann_callback_variable_minibatch {
 public:
  /// Represents a step in a schedule of mini-batch sizes.
  struct minibatch_step {
    /// Epoch for this schedule to start.
    int epoch;
    /// Mini-batch size to use.
    int mbsize;
    /// Learning rate to use.
    float lr;
    /// Number of epochs to ramp the learning rate over.
    int ramp_time;
    minibatch_step(int _epoch, int _mbsize, float _lr, int _ramp_time) :
      epoch(_epoch), mbsize(_mbsize), lr(_lr), ramp_time(_ramp_time) {}
  };

  lbann_callback_minibatch_schedule(
    int starting_mbsize, std::vector<minibatch_step> steps);
  lbann_callback_minibatch_schedule(
    const lbann_callback_minibatch_schedule&) = default;
  lbann_callback_minibatch_schedule& operator=(
    const lbann_callback_minibatch_schedule&) = delete;
  lbann_callback_minibatch_schedule* copy() const override {
    return new lbann_callback_minibatch_schedule(*this);
  }
  std::string name() const override { return "minibatch schedule"; }
 protected:
  bool schedule(model *m, int& new_mbsize, float& new_lr, int& ramp_time) override;
 private:
  /// Steps in the mini-batch schedule, stored in reverse sorted order.
  std::vector<minibatch_step> m_steps;
};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_minibatch_schedule_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

}  // namespace lbann

#endif  // LBANN_CALLBACKS_VARIABLE_MINIBATCH_HPP_INCLUDED
