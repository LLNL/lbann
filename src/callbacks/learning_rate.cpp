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

#include "lbann/callbacks/learning_rate.hpp"
#include "lbann/data_ingestion/coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include "callback_helpers.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <algorithm>
#include <cmath> // std::pow
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lbann {
namespace callback {

float learning_rate::m_cur_global_lr = 0.0f;

learning_rate::learning_rate() {}

learning_rate::learning_rate(std::vector<std::string> weights_names)
  : m_weights_names(std::move(weights_names))
{}

void learning_rate::setup(model* m)
{

  // Add all weights if list of weights is not initialized
  std::vector<weights*> weights_list =
    select_things_by_name(m->get_weights(), m_weights_names);
  if (weights_list.empty()) {
    weights_list = m->get_weights();
  }

  // Remove weights that are not being optimized
  std::unordered_set<weights*>().swap(m_weights);
  for (weights* w : weights_list) {
    if (w->has_optimizer()) {
      m_weights.insert(w);
      // Initialize the global learning rate, exactly once.
      if (m_cur_global_lr == 0.0f) {
        m_cur_global_lr = w->get_optimizer()->get_learning_rate();
      }
    }
  }
}

void learning_rate::on_epoch_end(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  const float new_lr = global_schedule(m);
  const float old_global_lr = m_cur_global_lr;
  m_cur_global_lr = new_lr;
  lbann_comm* comm = m->get_comm();
  if (comm->am_trainer_master() && new_lr != old_global_lr) {
    std::cout << "Model " << comm->get_trainer_rank() << ": "
              << "changing global learning rate to " << new_lr << " at epoch "
              << c.get_epoch() << std::endl;
  }
  for (weights* w : this->get_weights()) {
    auto* opt = w->get_optimizer();
    const float old_lr = opt->get_learning_rate();
    if (old_lr != new_lr) {
      opt->set_learning_rate(new_lr);
    }
  }
}

void learning_rate::on_backward_prop_end(model* m)
{
  for (weights* w : this->get_weights()) {
    auto& opt = *w->get_optimizer();
    const float old_lr = opt.get_learning_rate();
    const float new_lr = optimizer_schedule(m, opt);
    if (old_lr != new_lr) {
      opt.set_learning_rate(new_lr);
    }
  }
}

float learning_rate::optimizer_schedule(model* m, optimizer& opt)
{
  return opt.get_learning_rate();
}

step_learning_rate::step_learning_rate(size_t step, float amt)
  : learning_rate(), m_step(step), m_amt(amt)
{}

step_learning_rate::step_learning_rate(size_t step,
                                       float amt,
                                       std::vector<std::string> weights_names)
  : learning_rate(std::move(weights_names)), m_step(step), m_amt(amt)
{}

float step_learning_rate::global_schedule(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  if (c.get_epoch() % m_step == 0) {
    return step_learning_rate::get_current_global_learning_rate() * m_amt;
  }
  else {
    return step_learning_rate::get_current_global_learning_rate();
  }
}

void step_learning_rate::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_step_learning_rate();
  msg->set_weights(protobuf::to_space_sep_string(this->get_weights_names()));
  msg->set_step(m_step);
  msg->set_amt(m_amt);
}

set_learning_rate::set_learning_rate(size_t step, float val)
  : learning_rate(), m_step(step), m_val(val)
{}

set_learning_rate::set_learning_rate(size_t step,
                                     float val,
                                     std::vector<std::string> weights_names)
  : learning_rate(std::move(weights_names)), m_step(step), m_val(val)
{}

float set_learning_rate::global_schedule(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  if (c.get_epoch() == m_step) {
    return m_val;
  }
  else {
    return set_learning_rate::get_current_global_learning_rate();
  }
}

void set_learning_rate::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_set_learning_rate();
  msg->set_weights(protobuf::to_space_sep_string(this->get_weights_names()));
  msg->set_step(m_step);
  msg->set_val(m_val);
}

adaptive_learning_rate::adaptive_learning_rate(size_t patience, float amt)
  : adaptive_learning_rate(patience, amt, std::vector<std::string>())
{}

adaptive_learning_rate::adaptive_learning_rate(
  size_t patience,
  float amt,
  std::vector<std::string> weights_list)
  : learning_rate(std::move(weights_list)), m_patience(patience), m_amt(amt)
{}

float adaptive_learning_rate::global_schedule(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  // Determine behavior the first time this is called in an epoch
  if (m_cur_epoch != c.get_epoch()) {
    m_cur_epoch = c.get_epoch();
    const auto mode = c.get_execution_mode();
    const EvalType score = m->get_objective_function()->get_mean_value(mode);
    if (score < m_last_score) {
      // Reset wait counter if score has decreased
      m_last_score = score;
      m_wait = 0;
      m_adjust_learning_rate = false;
    }
    else if (m_wait >= m_patience) {
      // Adjust learning rate if patience has been exceeded
      m_last_score = score;
      m_wait = 0;
      m_adjust_learning_rate = true;
    }
    else {
      // Otherwise increment wait counter
      m_wait++;
      m_adjust_learning_rate = false;
    }
  }

  // Adjust learning rate if needed
  if (m_adjust_learning_rate) {
    return adaptive_learning_rate::get_current_global_learning_rate() * m_amt;
  }
  else {
    return adaptive_learning_rate::get_current_global_learning_rate();
  }
}

void adaptive_learning_rate::write_specific_proto(
  lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_adaptive_learning_rate();
  msg->set_weights(protobuf::to_space_sep_string(this->get_weights_names()));
  msg->set_patience(m_patience);
  msg->set_amt(m_amt);
}

drop_fixed_learning_rate::drop_fixed_learning_rate(
  std::vector<size_t> drop_epochs,
  float amt)
  : drop_fixed_learning_rate(std::move(drop_epochs),
                             amt,
                             std::vector<std::string>())
{}

drop_fixed_learning_rate::drop_fixed_learning_rate(
  std::vector<size_t> drop_epochs,
  float amt,
  std::vector<std::string> weights_names)
  : learning_rate(std::move(weights_names)),
    m_amt(amt),
    m_drop_epochs(std::move(drop_epochs))
{
  // Sort in reverse order.
  std::sort(m_drop_epochs.rbegin(), m_drop_epochs.rend());
}

void drop_fixed_learning_rate::write_specific_proto(
  lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_drop_fixed_learning_rate();
  msg->set_weights(protobuf::to_space_sep_string(this->get_weights_names()));
  protobuf::assign_to_repeated(*msg->mutable_drop_epoch(), m_drop_epochs);
  msg->set_amt(m_amt);
}

float drop_fixed_learning_rate::global_schedule(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  // Delete last drop epoch if we have already passed it
  while (!m_drop_epochs.empty() && c.get_epoch() > m_drop_epochs.back()) {
    m_drop_epochs.pop_back();
  }

  // Adjust learning rate if at a drop epoch
  if (!m_drop_epochs.empty() && c.get_epoch() == m_drop_epochs.back()) {
    return drop_fixed_learning_rate::get_current_global_learning_rate() * m_amt;
  }
  else {
    return drop_fixed_learning_rate::get_current_global_learning_rate();
  }
}

linear_growth_learning_rate::linear_growth_learning_rate(float target,
                                                         size_t num_epochs)
  : linear_growth_learning_rate(target,
                                num_epochs,
                                0,
                                std::vector<std::string>())
{}

linear_growth_learning_rate::linear_growth_learning_rate(float target,
                                                         size_t num_epochs,
                                                         size_t delay)
  : linear_growth_learning_rate(target,
                                num_epochs,
                                delay,
                                std::vector<std::string>())
{}

linear_growth_learning_rate::linear_growth_learning_rate(
  float target,
  size_t num_epochs,
  size_t delay,
  std::vector<std::string> weights_names)
  : learning_rate(std::move(weights_names)),
    m_target(target),
    m_inc(0),
    m_num_epochs(num_epochs),
    m_delay(delay)
{}

void linear_growth_learning_rate::setup(model* m)
{
  learning_rate::setup(m);
  // Compute the learning rate increase.
  if (!this->get_weights().empty()) {
    // Assumes all optimizers have the same initial learning rate.
    m_base_lr = linear_growth_learning_rate::get_current_global_learning_rate();
    m_inc = (m_target - m_base_lr) / m_num_epochs;
  }
}

float linear_growth_learning_rate::global_schedule(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  if (c.get_epoch() < m_delay) {
    return linear_growth_learning_rate::get_current_global_learning_rate();
  }
  else if (c.get_epoch() <= m_num_epochs + m_delay) {
    int num_left = m_num_epochs + m_delay - c.get_epoch();
    return m_base_lr + m_inc * (m_num_epochs - num_left);
  }
  else {
    return linear_growth_learning_rate::get_current_global_learning_rate();
  }
}

void linear_growth_learning_rate::write_specific_proto(
  lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_linear_growth_learning_rate();
  msg->set_weights(protobuf::to_space_sep_string(this->get_weights_names()));
  msg->set_target(m_target);
  msg->set_num_epochs(m_num_epochs);
  msg->set_delay(m_delay);
}

/**
 * This constructor takes the policy specific parameters, the exponent (p)
 * and the maximum number of iterations (max_iter).
 * In case that max_iter is set to 0, it is calculated from the number of
 * epochs (n_epochs). n_epochs is not used otherwise.
 */
poly_learning_rate::poly_learning_rate(double p,
                                       size_t n_epochs,
                                       size_t max_iter)
  : learning_rate(std::vector<std::string>()),
    m_p(p),
    m_num_epochs(n_epochs),
    m_max_iter(max_iter),
    m_start_lr(0.0f),
    m_end_lr(0.0f)
{}

poly_learning_rate::poly_learning_rate(double p,
                                       size_t n_epochs,
                                       size_t max_iter,
                                       double end_lr,
                                       std::vector<std::string> weights_names)
  : learning_rate(std::move(weights_names)),
    m_p(p),
    m_num_epochs(n_epochs),
    m_max_iter(max_iter),
    m_start_lr(0.0f),
    m_end_lr(end_lr)
{}

/**
 * Check if the maximum number of iterations is set. If not, compute it by the
 * number of epochs and the number of iterations per epoch.
 */
void poly_learning_rate::setup(model* m)
{
  learning_rate::setup(m);
  m_start_lr = get_current_global_learning_rate();
  if (m_max_iter == 0ull) {
    data_coordinator& dc = get_trainer().get_data_coordinator();
    m_max_iter =
      m_num_epochs * dc.get_num_iterations_per_epoch(execution_mode::training);
  }
}

/**
 * Keep the record of the learning rate at the end of the current epoch.
 */
float poly_learning_rate::global_schedule(model* m)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  const size_t iter = std::min(c.get_step(), m_max_iter);
  const float scale = static_cast<float>(
    std::pow(static_cast<double>(m_max_iter - iter) / m_max_iter, m_p));
  return (m_start_lr - m_end_lr) * scale + m_end_lr;
}

/**
 * Compute the learning rate for the next iteration.
 */
float poly_learning_rate::optimizer_schedule(model* m, optimizer& opt)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  const size_t iter = std::min(c.get_step(), m_max_iter);
  const float scale = static_cast<float>(
    std::pow(static_cast<double>(m_max_iter - iter) / m_max_iter, m_p));
  return (m_start_lr - m_end_lr) * scale + m_end_lr;
}

void poly_learning_rate::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_poly_learning_rate();
  msg->set_weights(protobuf::to_space_sep_string(this->get_weights_names()));
  msg->set_power(m_p);
  msg->set_num_epochs(m_num_epochs);
  msg->set_max_iter(m_max_iter);
  msg->set_end_lr(m_end_lr);
}

optimizerwise_adaptive_learning_rate::optimizerwise_adaptive_learning_rate(
  float scale)
  : optimizerwise_adaptive_learning_rate(scale, std::vector<std::string>())
{}

optimizerwise_adaptive_learning_rate::optimizerwise_adaptive_learning_rate(
  float scale,
  std::vector<std::string> weights_names)
  : learning_rate(std::move(weights_names)), m_scale(scale)
{}

float optimizerwise_adaptive_learning_rate::optimizer_schedule(model* m,
                                                               optimizer& opt)
{
  auto& dto = dynamic_cast<data_type_optimizer<DataType>&>(opt);
  DataType param_norm = El::Nrm2(dto.get_weights().get_values());
  DataType param_grad_norm = El::Nrm2(dto.get_gradient_sharded());
  if (param_norm > DataType(0) && param_grad_norm > DataType(0)) {
    // TODO: Should incorporate weight decay, etc. here.
    return optimizerwise_adaptive_learning_rate::
             get_current_global_learning_rate() *
           m_scale * param_norm / param_grad_norm;
  }
  else {
    return dto.get_learning_rate();
  }
}

cosine_decay_learning_rate::cosine_decay_learning_rate(
  double lr_max,
  double lr_min,
  size_t decay_steps,
  double initial_learning_rate,
  size_t warmup_steps)
  : learning_rate(std::vector<std::string>()),
    m_lr_max(lr_max),
    m_lr_min(lr_min),
    m_decay_steps(decay_steps),
    m_initial_lr(initial_learning_rate),
    m_warmup_steps(warmup_steps)
{}

cosine_decay_learning_rate::cosine_decay_learning_rate(
  double lr_max,
  double lr_min,
  size_t decay_steps,
  double initial_learning_rate,
  size_t warmup_steps,
  std::vector<std::string> weights_names)
  : learning_rate(std::move(weights_names)),
    m_lr_max(lr_max),
    m_lr_min(lr_min),
    m_decay_steps(decay_steps),
    m_initial_lr(initial_learning_rate),
    m_warmup_steps(warmup_steps)
{}

void cosine_decay_learning_rate::setup(model* m) { learning_rate::setup(m); }

/**
 * Keep the record of the learning rate at the end of the current epoch.
 * The function computes a step-wise learning rate and is also called from
 * ``optimizer_schedule``.
 */
float cosine_decay_learning_rate::global_schedule(model* m)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  auto step = c.get_step();
  size_t max_steps = m_warmup_steps + m_decay_steps;
  // Piecewise evaluation based on step
  if (step >= max_steps) { // Post-decay
    return m_lr_min;
  }
  else if (step >= m_warmup_steps) { // Post-warmup, in decay
    step -= m_warmup_steps;
    float ratio = static_cast<float>(step) / m_decay_steps;
    float cosine = 0.5f * (1.0f + static_cast<float>(std::cos(M_PI * ratio)));
    return m_lr_min + (m_lr_max - m_lr_min) * cosine;
  }
  else { // Warmup (step < m_warmup_steps)
    float ratio = static_cast<float>(step) / m_warmup_steps;
    float delta = m_lr_max - m_initial_lr;
    return m_initial_lr + ratio * delta;
  }
}

/**
 * Compute the learning rate for the next iteration.
 */
float cosine_decay_learning_rate::optimizer_schedule(model* m, optimizer& opt)
{
  return this->global_schedule(m);
}

void cosine_decay_learning_rate::write_specific_proto(
  lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_cosine_decay_learning_rate();
  msg->set_weights(protobuf::to_space_sep_string(this->get_weights_names()));
  msg->set_lr_max(m_lr_max);
  msg->set_lr_min(m_lr_min);
  msg->set_decay_steps(m_decay_steps);
  msg->set_initial_warmup_learning_rate(m_initial_lr);
  msg->set_warmup_steps(m_warmup_steps);
}

void optimizerwise_adaptive_learning_rate::write_specific_proto(
  lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_optimizerwise_adaptive_learning_rate();
  msg->set_weights(protobuf::to_space_sep_string(this->get_weights_names()));
  msg->set_scale(m_scale);
}

std::unique_ptr<callback_base> build_step_learning_rate_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackStepLearningRate&>(
      proto_msg);
  return std::make_unique<step_learning_rate>(
    params.step(),
    params.amt(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<callback_base> build_set_learning_rate_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSetLearningRate&>(
      proto_msg);
  return std::make_unique<set_learning_rate>(
    params.step(),
    params.val(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<callback_base> build_adaptive_learning_rate_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackAdaptiveLearningRate&>(
      proto_msg);
  return std::make_unique<adaptive_learning_rate>(
    params.patience(),
    params.amt(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<callback_base>
build_drop_fixed_learning_rate_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDropFixedLearningRate&>(
      proto_msg);
  std::vector<size_t> drop_epochs;
  for (int i = 0; i < params.drop_epoch_size(); ++i) {
    drop_epochs.push_back(params.drop_epoch(i));
  }
  return std::make_unique<drop_fixed_learning_rate>(
    std::move(drop_epochs),
    params.amt(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<callback_base>
build_linear_growth_learning_rate_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  using MsgType = lbann_data::Callback::CallbackLinearGrowthLearningRate;
  using CallbackType = linear_growth_learning_rate;
  const auto& params = dynamic_cast<const MsgType&>(proto_msg);
  return std::make_unique<CallbackType>(
    params.target(),
    params.num_epochs(),
    params.delay(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<callback_base>
build_optimizerwise_adaptive_learning_rate_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  using MsgType =
    lbann_data::Callback::CallbackOptimizerwiseAdaptiveLearningRate;
  using CallbackType = optimizerwise_adaptive_learning_rate;
  const auto& params = dynamic_cast<const MsgType&>(proto_msg);
  return std::make_unique<CallbackType>(
    params.scale(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<callback_base> build_poly_learning_rate_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackPolyLearningRate&>(
      proto_msg);
  return std::make_unique<poly_learning_rate>(
    params.power(),
    params.num_epochs(),
    params.max_iter(),
    params.end_lr(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<callback_base>
build_cosine_decay_learning_rate_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackCosineDecayLearningRate&>(
      proto_msg);
  return std::make_unique<cosine_decay_learning_rate>(
    params.lr_max(),
    params.lr_min(),
    params.decay_steps(),
    params.initial_warmup_learning_rate(),
    params.warmup_steps(),
    parse_list<std::string>(params.weights()));
}

} // namespace callback
} // namespace lbann
