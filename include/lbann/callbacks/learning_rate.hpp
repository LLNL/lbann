////////////////////////////////////////////////////////////////////////////////
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
// lbann_learning_rate .hpp .cpp - Callback hooks for learning rate schedules
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_LEARNING_RATE_HPP_INCLUDED
#define LBANN_CALLBACKS_LEARNING_RATE_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <unordered_map>
#include <unordered_set>

namespace lbann {

// Forward declarations
class optimizer;

namespace callback {

// Different schedules should inherit from learning_rate.

/**
 * Base class for learning rate schedules.
 * Child classes should implement the schedule method to make changes.
 */
class learning_rate : public callback_base
{
public:
  learning_rate();
  learning_rate(const learning_rate&) = default;
  learning_rate& operator=(const learning_rate&) = default;
  /** Only apply to specific weights. */
  learning_rate(std::vector<std::string> weights_names);
  /** Do some initialization. */
  void setup(model* m) override;
  /** Apply global learning rate schedules. */
  void on_epoch_end(model* m) override;

  using callback_base::on_backward_prop_end;
  /** Apply local/per-optimizer learning rate schedules. */
  void on_backward_prop_end(model* m) override;

protected:
  std::vector<std::string> const& get_weights_names() const
  {
    return m_weights_names;
  }

protected:
  /**
   * This is called at the end of every epoch to update the learning
   * rate for every optimizer. Adjustments should be made based on the
   * current global learning rate.
   * The returned learning rate will be used to automatically update
   * the current global learning rate.
   */
  virtual float global_schedule(model* m)
  {
    return get_current_global_learning_rate();
  }

  /**
   * This is called at the end of every training mini-batch to update the
   * learning rate for optimizer opt. The current global learning rate is *not*
   * updated automatically based on this method.
   */
  virtual float optimizer_schedule(model* m, optimizer& opt);

  const std::unordered_set<weights*>& get_weights() const noexcept
  {
    return m_weights;
  }

  static float get_current_global_learning_rate() noexcept
  {
    return m_cur_global_lr;
  }

  static void update_global_learning_rate(float rate) noexcept
  {
    m_cur_global_lr = rate;
  }

private:
  /**
   * This should be maintained by all learning rate schedule
   * implementations as the current global learning rate. This enables
   * coordination among different schedules, particularly ones that
   * work on a per-optimizer basis.
   */
  static float m_cur_global_lr;

  /** Names of the weights being updated. */
  std::vector<std::string> m_weights_names;

  /** Weights to update. */
  std::unordered_set<weights*> m_weights;
};

/**
 * Decrease the learning rate by a fixed proportion every X epochs.
 */
class step_learning_rate : public learning_rate
{
public:
  /** Decrease the learning rate by amt every step epochs. */
  step_learning_rate(size_t step, float amt);
  step_learning_rate(size_t step,
                     float amt,
                     std::vector<std::string> weights_names);
  step_learning_rate(const step_learning_rate&) = default;
  step_learning_rate& operator=(const step_learning_rate&) = default;
  step_learning_rate* copy() const override
  {
    return new step_learning_rate(*this);
  }
  std::string name() const override { return "step learning rate"; }

protected:
  float global_schedule(model* m) override;

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /** Number of epochs between each learning rate decrease. */
  size_t m_step;
  /** Amount to decrease the learning rate by. */
  float m_amt;
};

// Builder function
std::unique_ptr<callback_base> build_step_learning_rate_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

/**
 * Set the learning rate to given value at given epoch.
 */
class set_learning_rate : public learning_rate
{
public:
  set_learning_rate(size_t step, float val);
  set_learning_rate(size_t step,
                    float val,
                    std::vector<std::string> weights_names);
  set_learning_rate(const set_learning_rate&) = default;
  set_learning_rate& operator=(const set_learning_rate&) = default;
  set_learning_rate* copy() const override
  {
    return new set_learning_rate(*this);
  }
  std::string name() const override { return "step learning rate"; }

protected:
  float global_schedule(model* m) override;

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;
  /** Number of epochs between each learning rate decrease. */
  size_t m_step;
  /** Amount to decrease the learning rate by. */
  float m_val;
};

// Builder function
std::unique_ptr<callback_base> build_set_learning_rate_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

/**
 * Decrease the learning rate by a fixed proportion when validation error stops
 * improving.
 */
class adaptive_learning_rate : public learning_rate
{
public:
  /**
   * Decrease the learning rate by amt if accuracy does not improve for patience
   * epochs.
   */
  adaptive_learning_rate(size_t patience, float amt);
  adaptive_learning_rate(size_t patience,
                         float amt,
                         std::vector<std::string> weights_names);
  adaptive_learning_rate(const adaptive_learning_rate&) = default;
  adaptive_learning_rate& operator=(const adaptive_learning_rate&) = default;
  adaptive_learning_rate* copy() const override
  {
    return new adaptive_learning_rate(*this);
  }
  std::string name() const override { return "adaptive learning rate"; }

protected:
  float global_schedule(model* m) override;

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /** Number of epochs to wait for improvements. */
  size_t m_patience;
  /** Amount to decrease the learning rate by. */
  float m_amt;
  /** Current epoch. */
  size_t m_cur_epoch = std::numeric_limits<size_t>::max();
  /** Last recorded score. */
  EvalType m_last_score = std::numeric_limits<EvalType>::max();
  /** Current number of epochs without improvement. */
  size_t m_wait = 0;
  /** Whether to adjust learning rate for current epoch. */
  bool m_adjust_learning_rate = false;
};

// Builder function
std::unique_ptr<callback_base> build_adaptive_learning_rate_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

/**
 * Decrease learning rate by a fixed amount at fixed times.
 */
class drop_fixed_learning_rate : public learning_rate
{
public:
  /**
   * Decrease the learning rate by amt when each epoch in drop_epochs is
   * reached.
   */
  drop_fixed_learning_rate(std::vector<size_t> drop_epochs, float amt);
  drop_fixed_learning_rate(std::vector<size_t> drop_epochs,
                           float amt,
                           std::vector<std::string> weights_names);
  drop_fixed_learning_rate(const drop_fixed_learning_rate&) = default;
  drop_fixed_learning_rate&
  operator=(const drop_fixed_learning_rate&) = default;
  drop_fixed_learning_rate* copy() const override
  {
    return new drop_fixed_learning_rate(*this);
  }
  std::string name() const override { return "drop fixed learning rate"; }

protected:
  float global_schedule(model* m) override;

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /// Amount to decrease the learning rate by.
  float m_amt;
  /**
   * Epochs to drop learning rate at. This is stored in reverse sorted order,
   * so that the end can be examined and then popped in constant time.
   */
  std::vector<size_t> m_drop_epochs;
};

// Builder function
std::unique_ptr<callback_base>
build_drop_fixed_learning_rate_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

/**
 * Linearly increase the learning rate to reach a target value over a
 * fixed number of epochs.
 * @note This currently assumes every optimizer begins with the same
 * learning rate.  This also *forces* its schedule and will stomp over
 * other changes.
 */
class linear_growth_learning_rate : public learning_rate
{
public:
  /**
   * Linearly increase the learning rate to reach target after num_epochs.
   */
  linear_growth_learning_rate(float target, size_t num_epochs);
  linear_growth_learning_rate(float target, size_t num_epochs, size_t delay);
  linear_growth_learning_rate(float target,
                              size_t num_epochs,
                              size_t delay,
                              std::vector<std::string> weights_names);
  linear_growth_learning_rate(const linear_growth_learning_rate&) = default;
  linear_growth_learning_rate&
  operator=(const linear_growth_learning_rate&) = default;
  linear_growth_learning_rate* copy() const override
  {
    return new linear_growth_learning_rate(*this);
  }
  void setup(model* m) override;
  std::string name() const override { return "linear growth learning rate"; }

protected:
  float global_schedule(model* m) override;

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /// Initial learning rate.
  float m_base_lr;
  /// Target learning rate to reach.
  float m_target;
  /// Amount to increase each epoch.
  float m_inc;
  /// Number of epochs over which to scale the learning rate.
  size_t m_num_epochs;
  /// Number of epochs to delay before starting growth.
  size_t m_delay;
};

// Builder function
std::unique_ptr<callback_base>
build_linear_growth_learning_rate_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

/**
 * Decrease the learning rate by polynomial policy
 * base_lr*(1 - i_cur/i_max)^p, where
 * base_lr is the initial learning rate, i_cur is the current iteration,
 * i_max is the maximum iteration, and p is a parameter.
 */
class poly_learning_rate : public learning_rate
{
public:
  poly_learning_rate(double p, size_t n_epochs, size_t max_iter);
  poly_learning_rate(double p,
                     size_t n_epochs,
                     size_t max_iter,
                     double endl_r,
                     std::vector<std::string> weights_names);
  poly_learning_rate(const poly_learning_rate&) = default;
  poly_learning_rate& operator=(const poly_learning_rate&) = default;
  poly_learning_rate* copy() const override
  {
    return new poly_learning_rate(*this);
  }
  void setup(model* m) override;
  std::string name() const override { return "poly learning rate"; }

protected:
  float global_schedule(model* m) override;
  float optimizer_schedule(model* m, optimizer& opt) override;

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /// The exponent to compute new learning rate in poly policy
  double m_p;
  /// The number of epochs for training
  size_t m_num_epochs;
  /// The maximum number of iterations until which the learning rate changes
  size_t m_max_iter;
  /// The initial learning rate
  float m_start_lr;
  /// The final learning rate
  float m_end_lr;
};

// Builder function
std::unique_ptr<callback_base> build_poly_learning_rate_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

/**
 * This implements an adaptive scheme for adjust each optimizer's
 * learning rate based on the ratio of the norms of its weights and
 * its gradients.
 * See: You et al. "Scaling SGD Batch Size to 32K for ImageNet
 * Training", 2017.
 */
class optimizerwise_adaptive_learning_rate : public learning_rate
{
public:
  optimizerwise_adaptive_learning_rate(float scale);
  optimizerwise_adaptive_learning_rate(float scale,
                                       std::vector<std::string> weights_names);
  optimizerwise_adaptive_learning_rate(
    const optimizerwise_adaptive_learning_rate&) = default;
  optimizerwise_adaptive_learning_rate&
  operator=(const optimizerwise_adaptive_learning_rate&) = default;
  optimizerwise_adaptive_learning_rate* copy() const override
  {
    return new optimizerwise_adaptive_learning_rate(*this);
  }
  std::string name() const override
  {
    return "optimizerwise adaptive learning rate";
  }

protected:
  float optimizer_schedule(model* m, optimizer& opt) override;

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  float m_scale;
};

// Builder function
std::unique_ptr<callback_base>
build_optimizerwise_adaptive_learning_rate_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

/**
 * Decrease the learning rate with a cosine policy and a potential stepwise
 * warmup, originally presented in:
 * Ilya Loshchilov, Frank Hutter, "SGDR: Stochastic Gradient Descent with Warm
 * Restarts." ICLR 2017.
 *
 * The formula is as follows:
 *
 * lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t_cur / t_total))
 *
 * where t_cur is the current iteration and t_total is the total number of
 * iterations. If warmup steps are given, the schedule starts with
 * ``initial_learning_rate`` and gradually raises the learning rate to
 * ``lr_max``, after which cosine decay will occur for the specified number of
 * steps until ``lr_min`` is reached.
 */
class cosine_decay_learning_rate : public learning_rate
{
public:
  cosine_decay_learning_rate(double lr_max,
                             double lr_min,
                             size_t decay_steps,
                             double initial_learning_rate = 0.0,
                             size_t warmup_steps = 0);
  cosine_decay_learning_rate(double lr_max,
                             double lr_min,
                             size_t decay_steps,
                             double initial_learning_rate,
                             size_t warmup_steps,
                             std::vector<std::string> weight_names);
  cosine_decay_learning_rate(const cosine_decay_learning_rate&) = default;
  cosine_decay_learning_rate&
  operator=(const cosine_decay_learning_rate&) = default;
  cosine_decay_learning_rate* copy() const override
  {
    return new cosine_decay_learning_rate(*this);
  }
  void setup(model* m) override;
  std::string name() const override { return "cosine decay learning rate"; }

protected:
  float global_schedule(model* m) override;
  float optimizer_schedule(model* m, optimizer& opt) override;

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /// The starting learning rate before decay
  float m_lr_max;
  /// The learning rate after cosine decay
  float m_lr_min;
  /// The number of steps for decay
  size_t m_decay_steps;
  /// The initial learning rate for warmup. Relevant only if m_warmup_steps > 0
  float m_initial_lr;
  /// Number of warmup steps
  size_t m_warmup_steps;
};

// Builder function
std::unique_ptr<callback_base>
build_cosine_decay_learning_rate_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_LEARNING_RATE_HPP_INCLUDED
