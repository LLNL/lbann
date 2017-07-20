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
// lbann_learning_rate .hpp .cpp - Callback hooks for learning rate schedules
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_LEARNING_RATE_HPP_INCLUDED
#define LBANN_CALLBACKS_LEARNING_RATE_HPP_INCLUDED

#include <unordered_set>
#include <unordered_map>
#include "lbann/callbacks/callback.hpp"

namespace lbann {

// Different schedules should inherit from lbann_callback_learning_rate.

/**
 * Base class for learning rate schedules.
 * Child classes should implement the schedule method to make changes.
 */
class lbann_callback_learning_rate : public lbann_callback {
 public:
  lbann_callback_learning_rate();
  lbann_callback_learning_rate(const lbann_callback_learning_rate&) = default;
  lbann_callback_learning_rate& operator=(
    const lbann_callback_learning_rate&) = default;
  /** Only apply to specific layers. */
  lbann_callback_learning_rate(std::unordered_set<uint> layers);
  /** Do some initialization. */
  void setup(model *m);
  /** Apply the learning rate schedule. */
  void on_epoch_end(model *m);
 protected:
  /**
   * This is called every epoch to potentially update the learning rate.
   * @param m The model being trained.
   * @param l The layer in the model currently being potentially updated.
   * @return A new learning rate.
   */
  virtual float schedule(model *m, learning *l) = 0;
  /** Return true if l is the last layer to update this epoch. */
  bool is_last_layer(const Layer *l) const {
    return l->get_index() == (int) m_last_idx;
  }
 private:
  /** Indicies of layers to update. */
  std::unordered_set<uint> m_layer_indices;
  /** Record old learning rates to watch for changes. */
  std::unordered_map<uint, float> m_old_lrs;
  /** Index of the last layer to update. */
  uint m_last_idx;
};

/**
 * Decrease the learning rate by a fixed proportion every X epochs.
 */
class lbann_callback_step_learning_rate : public lbann_callback_learning_rate {
 public:
  /** Decrease the learning rate by amt every step epochs. */
  lbann_callback_step_learning_rate(int step, float amt);
  lbann_callback_step_learning_rate(int step, float amt,
                                    std::unordered_set<uint> layers);
  lbann_callback_step_learning_rate(
    const lbann_callback_step_learning_rate&) = default;
  lbann_callback_step_learning_rate& operator=(
    const lbann_callback_step_learning_rate&) = default;
  lbann_callback_step_learning_rate* copy() const {
    return new lbann_callback_step_learning_rate(*this);
  }
  std::string name() const { return "step learning rate"; }
 protected:
  float schedule(model *m, learning *l);
 private:
  /** Number of epochs between each learning rate decrease. */
  int m_step;
  /** Amount to decrease the learning rate by. */
  float m_amt;
};

/**
 * Decrease the learning rate by a fixed proportion when validation error stops
 * improving.
 */
class lbann_callback_adaptive_learning_rate : public lbann_callback_learning_rate {
 public:
  /**
   * Decrease the learning rate by amt if accuracy does not improve for patience
   * epochs.
   */
  lbann_callback_adaptive_learning_rate(int64_t patience, float amt);
  lbann_callback_adaptive_learning_rate(int64_t patience, float amt,
                                        std::unordered_set<uint> layers);
  lbann_callback_adaptive_learning_rate(
    const lbann_callback_adaptive_learning_rate&) = default;
  lbann_callback_adaptive_learning_rate& operator=(
    const lbann_callback_adaptive_learning_rate&) = default;
  lbann_callback_adaptive_learning_rate* copy() const {
    return new lbann_callback_adaptive_learning_rate(*this);
  }
  std::string name() const { return "adaptive learning rate"; }
 protected:
  float schedule(model *m, learning *l);
 private:
  /** Number of epochs to wait for improvements. */
  int64_t m_patience;
  /** Amount to decrease the learning rate by. */
  float m_amt;
  /** Last recorded score. */
  double m_last_score = std::numeric_limits<double>::max();
  /** Current number of epochs without improvement. */
  int64_t m_wait = 0;
};

/**
 * Use a custom user-provided schedule method to update the learning rate.
 */
class lbann_callback_custom_learning_rate : public lbann_callback_learning_rate {
 public:
  /** Use custom_schedule to change the learning rate. */
  lbann_callback_custom_learning_rate(
    std::function<float(model *, learning *)> custom_schedule);
  lbann_callback_custom_learning_rate(
    std::function<float(model *, learning *)> custom_schedule,
    std::unordered_set<uint> layers);
  /**
   * @todo Need to provide a way for model/layer to be updated after copy.
   */
  lbann_callback_custom_learning_rate(
    const lbann_callback_custom_learning_rate&) = default;
  lbann_callback_custom_learning_rate& operator=(
    const lbann_callback_custom_learning_rate&) = default;
  lbann_callback_custom_learning_rate* copy() const {
    return new lbann_callback_custom_learning_rate(*this);
  }
  std::string name() const { return "custom learning rate"; }
 protected:
  float schedule(model *m, learning *l);
 private:
  /** Custom update schedule. */
  std::function<float(model *, learning *)> m_custom_schedule;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_LEARNING_RATE_HPP_INCLUDED
