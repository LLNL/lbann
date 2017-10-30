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
#include "lbann/layers/optimizable_layer.hpp"

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
  lbann_callback_learning_rate(std::unordered_set<Layer *> layers);
  /** Do some initialization. */
  void setup(model *m);
  /** Apply global learning rate schedules. */
  void on_epoch_end(model *m);
  /** Apply local/per-layer learning rate schedules. */
  void on_backward_prop_end(model *m);
 protected:
  /**
   * This is called at the end of every epoch to update the learning rate
   * for every layer. Adjustments should be made based on the current global
   * learning rate.
   * The returned learning rate will be used to automatically update the current
   * global learning rate.
   */
  virtual float global_schedule(model *m) { return m_cur_global_lr; }
  /**
   * This is called at the end of every training mini-batch to update the
   * learning rate for layer l. The current global learning rate is *not*
   * updated automatically based on this method.
   */
  virtual float layer_schedule(model *m, Layer *l) {
    optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(l);
    return opt_layer->get_optimizer()->get_learning_rate();
  }

  /** Layers to update. */
  std::unordered_set<Layer *> m_layers;

  /**
   * This should be maintained by all learning rate schedule implementations
   * as the current global learning rate. This enables coordination among
   * different schedules, particularly ones that work on a per-layer basis.
   */
  static float m_cur_global_lr;
};

/**
 * Decrease the learning rate by a fixed proportion every X epochs.
 */
class lbann_callback_step_learning_rate : public lbann_callback_learning_rate {
 public:
  /** Decrease the learning rate by amt every step epochs. */
  lbann_callback_step_learning_rate(int step, float amt);
  lbann_callback_step_learning_rate(int step, float amt,
                                    std::unordered_set<Layer *> layers);
  lbann_callback_step_learning_rate(
    const lbann_callback_step_learning_rate&) = default;
  lbann_callback_step_learning_rate& operator=(
    const lbann_callback_step_learning_rate&) = default;
  lbann_callback_step_learning_rate* copy() const {
    return new lbann_callback_step_learning_rate(*this);
  }
  std::string name() const { return "step learning rate"; }
 protected:
  float global_schedule(model *m);
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
                                        std::unordered_set<Layer *> layers);
  lbann_callback_adaptive_learning_rate(
    const lbann_callback_adaptive_learning_rate&) = default;
  lbann_callback_adaptive_learning_rate& operator=(
    const lbann_callback_adaptive_learning_rate&) = default;
  lbann_callback_adaptive_learning_rate* copy() const {
    return new lbann_callback_adaptive_learning_rate(*this);
  }
  std::string name() const { return "adaptive learning rate"; }
 protected:
  float global_schedule(model *m);
 private:
  /** Number of epochs to wait for improvements. */
  int64_t m_patience;
  /** Amount to decrease the learning rate by. */
  float m_amt;
  /** Current epoch. */
  int m_cur_epoch = -1;
  /** Last recorded score. */
  double m_last_score = std::numeric_limits<double>::max();
  /** Current number of epochs without improvement. */
  int64_t m_wait = 0;
  /** Whether to adjust learning rate for current epoch. */
  bool m_adjust_learning_rate = false;
};

/**
 * Decrease learning rate by a fixed amount at fixed times.
 */
class lbann_callback_drop_fixed_learning_rate :
    public lbann_callback_learning_rate {
 public:
  /**
   * Decrease the learning rate by amt when each epoch in drop_epochs is
   * reached.
   */
  lbann_callback_drop_fixed_learning_rate(
    std::vector<int64_t> drop_epochs, float amt);
  lbann_callback_drop_fixed_learning_rate(
    std::vector<int64_t> drop_epochs, float amt,
    std::unordered_set<Layer *> layers);
  lbann_callback_drop_fixed_learning_rate(
    const lbann_callback_drop_fixed_learning_rate&) = default;
  lbann_callback_drop_fixed_learning_rate& operator=(
    const lbann_callback_drop_fixed_learning_rate&) = default;
  lbann_callback_drop_fixed_learning_rate* copy() const {
    return new lbann_callback_drop_fixed_learning_rate(*this);
  }
  std::string name() const { return "drop fixed learning rate"; }
 protected:
  float global_schedule(model *m);
 private:
  /// Amount to decrease the learning rate by.
  float m_amt;
  /**
   * Epochs to drop learning rate at. This is stored in reverse sorted order,
   * so that the end can be examined and then popped in constant time.
   */
  std::vector<int64_t> m_drop_epochs;
};

/**
 * Linearly increase the learning rate to reach a target value over a fixed
 * number of epochs.
 * @note This currently assumes every layer begins with the same learning rate.
 * This also *forces* its schedule and will stomp over other changes.
 */
class lbann_callback_linear_growth_learning_rate :
    public lbann_callback_learning_rate {
 public:
  /**
   * Linearly increase the learning rate to reach target after num_epochs.
   */
  lbann_callback_linear_growth_learning_rate(
    float target, int64_t num_epochs);
  lbann_callback_linear_growth_learning_rate(
    float target, int64_t num_epochs, int64_t delay);
  lbann_callback_linear_growth_learning_rate(
    float target, int64_t num_epochs, int64_t delay,
    std::unordered_set<Layer *> layers);
  lbann_callback_linear_growth_learning_rate(
    const lbann_callback_linear_growth_learning_rate&) = default;
  lbann_callback_linear_growth_learning_rate& operator=(
    const lbann_callback_linear_growth_learning_rate&) = default;
  lbann_callback_linear_growth_learning_rate* copy() const {
    return new lbann_callback_linear_growth_learning_rate(*this); }
  void setup(model *m);
  std::string name() const { return "linear growth learning rate"; }
 protected:
  float global_schedule(model *m);
 private:
  /// Initial learning rate.
  float m_base_lr;
  /// Target learning rate to reach.
  float m_target;
  /// Amount to increase each epoch.
  float m_inc;
  /// Number of epochs over which to scale the learning rate.
  int64_t m_num_epochs;
  /// Number of epochs to delay before starting growth.
  int64_t m_delay;
};

/**
 * This implements an adaptive scheme for adjust each layer's learning rate
 * based on the ratio of the norms of its weights and its gradients.
 * See: You et al. "Scaling SGD Batch Size to 32K for ImageNet Training", 2017.
 */
class lbann_callback_layerwise_adaptive_learning_rate : public lbann_callback_learning_rate {
 public:
  lbann_callback_layerwise_adaptive_learning_rate(float scale);
  lbann_callback_layerwise_adaptive_learning_rate(
    float scale, std::unordered_set<Layer*> layers);
  lbann_callback_layerwise_adaptive_learning_rate(
    const lbann_callback_layerwise_adaptive_learning_rate&) = default;
  lbann_callback_layerwise_adaptive_learning_rate& operator=(
    const lbann_callback_layerwise_adaptive_learning_rate&) = default;
  lbann_callback_layerwise_adaptive_learning_rate* copy() const {
    return new lbann_callback_layerwise_adaptive_learning_rate(*this); }
  std::string name() const { return "layerwise adaptive learning rate"; }
 protected:
  float layer_schedule(model *m, Layer *l);
 private:
  float m_scale;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_LEARNING_RATE_HPP_INCLUDED
