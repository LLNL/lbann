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
// lbann_learning_rate .hpp .cpp - Callback hooks for learning rate schedules
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_learning_rate.hpp"
#include "lbann/proto/proto_common.hpp"

#include "callback_helpers.hpp"

#include <algorithm>
#include <cmath> // std::pow
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lbann {

float lbann_callback_learning_rate::m_cur_global_lr = 0.0f;

lbann_callback_learning_rate::lbann_callback_learning_rate() {}

lbann_callback_learning_rate::lbann_callback_learning_rate(
  std::vector<std::string> weights_names)
  : m_weights_names(std::move(weights_names)) {}

void lbann_callback_learning_rate::setup(model *m) {

  // Add all weights if list of weights is not initialized
  std::vector<weights *> weights_list =
    select_things_by_name(m->get_weights(), m_weights_names);
  if (weights_list.empty()) {
    weights_list = m->get_weights();
  }

  // Remove weights that are not being optimized
  std::unordered_set<weights*>().swap(m_weights);
  for (weights *w : weights_list) {
    optimizer *opt = w->get_optimizer();
    if (opt != nullptr) {
      m_weights.insert(w);
      // Initialize the global learning rate, exactly once.
      if (m_cur_global_lr == 0.0f) {
        m_cur_global_lr = opt->get_learning_rate();
      }
    }
  }

}

void lbann_callback_learning_rate::on_epoch_end(model *m) {
  const float new_lr = global_schedule(m);
  const float old_global_lr = m_cur_global_lr;
  m_cur_global_lr = new_lr;
  lbann_comm *comm = m->get_comm();
  if (comm->am_trainer_master() && new_lr != old_global_lr) {
    std::cout << "Model " << comm->get_trainer_rank() << ": "
              << "changing global learning rate to " << new_lr
              << " at epoch " << m->get_epoch() << std::endl;
  }
  for (weights *w : this->get_weights()) {
    optimizer *opt = w->get_optimizer();
    const float old_lr = opt->get_learning_rate();
    if (old_lr != new_lr) {
      opt->set_learning_rate(new_lr);
    }
  }
}

void lbann_callback_learning_rate::on_backward_prop_end(model *m) {
  for (weights *w : this->get_weights()) {
    optimizer& opt = *w->get_optimizer();
    const float old_lr = opt.get_learning_rate();
    const float new_lr = optimizer_schedule(m, opt);
    if (old_lr != new_lr) {
      opt.set_learning_rate(new_lr);
    }
  }
}

lbann_callback_step_learning_rate::lbann_callback_step_learning_rate(
  int step, float amt) :
  lbann_callback_learning_rate(), m_step(step), m_amt(amt) {}

lbann_callback_step_learning_rate::lbann_callback_step_learning_rate(
  int step, float amt, std::vector<std::string> weights_names) :
  lbann_callback_learning_rate(std::move(weights_names)),
  m_step(step), m_amt(amt) {}

float lbann_callback_step_learning_rate::global_schedule(model *m) {
  if (m->get_epoch() % m_step == 0) {
    return get_current_global_learning_rate() * m_amt;
  } else {
    return get_current_global_learning_rate();
  }
}

lbann_callback_adaptive_learning_rate::lbann_callback_adaptive_learning_rate(
  int64_t patience, float amt) :
  lbann_callback_adaptive_learning_rate(patience, amt,
                                        std::vector<std::string>()) {}

lbann_callback_adaptive_learning_rate::lbann_callback_adaptive_learning_rate(
  int64_t patience, float amt, std::vector<std::string> weights_list) :
  lbann_callback_learning_rate(std::move(weights_list)),
  m_patience(patience), m_amt(amt) {}

float lbann_callback_adaptive_learning_rate::global_schedule(model *m) {
  // Determine behavior the first time this is called in an epoch
  if (m_cur_epoch != m->get_epoch()) {
    m_cur_epoch = m->get_epoch();
    const execution_mode mode = m->get_execution_mode();
    const EvalType score = m->get_objective_function()->get_mean_value(mode);
    if (score < m_last_score) {
      // Reset wait counter if score has decreased
      m_last_score = score;
      m_wait = 0;
      m_adjust_learning_rate = false;
    } else if (m_wait >= m_patience) {
      // Adjust learning rate if patience has been exceeded
      m_last_score = score;
      m_wait = 0;
      m_adjust_learning_rate = true;
    } else {
      // Otherwise increment wait counter
      m_wait++;
      m_adjust_learning_rate = false;
    }
  }

  // Adjust learning rate if needed
  if (m_adjust_learning_rate) {
    return get_current_global_learning_rate() * m_amt;
  } else {
    return get_current_global_learning_rate();
  }
}

lbann_callback_drop_fixed_learning_rate::lbann_callback_drop_fixed_learning_rate(
  std::vector<int64_t> drop_epochs, float amt) :
  lbann_callback_drop_fixed_learning_rate(std::move(drop_epochs), amt,
                                          std::vector<std::string>()) {}

lbann_callback_drop_fixed_learning_rate::lbann_callback_drop_fixed_learning_rate(
  std::vector<int64_t> drop_epochs, float amt, std::vector<std::string> weights_names) :
  lbann_callback_learning_rate(std::move(weights_names)),
  m_amt(amt), m_drop_epochs(std::move(drop_epochs)) {
  // Sort in reverse order.
  std::sort(m_drop_epochs.rbegin(), m_drop_epochs.rend());
}

float lbann_callback_drop_fixed_learning_rate::global_schedule(model* m) {
  // Delete last drop epoch if we have already passed it
  while (!m_drop_epochs.empty()
         && m->get_epoch() > m_drop_epochs.back()) {
    m_drop_epochs.pop_back();
  }

  // Adjust learning rate if at a drop epoch
  if (!m_drop_epochs.empty() && m->get_epoch() == m_drop_epochs.back()) {
    return get_current_global_learning_rate() * m_amt;
  } else {
    return get_current_global_learning_rate();
  }
}

lbann_callback_linear_growth_learning_rate::lbann_callback_linear_growth_learning_rate(
  float target, int64_t num_epochs) :
  lbann_callback_linear_growth_learning_rate(target, num_epochs, 0,
                                             std::vector<std::string>()) {}

lbann_callback_linear_growth_learning_rate::lbann_callback_linear_growth_learning_rate(
  float target, int64_t num_epochs, int64_t delay) :
  lbann_callback_linear_growth_learning_rate(target, num_epochs, delay,
                                             std::vector<std::string>()) {}

lbann_callback_linear_growth_learning_rate::lbann_callback_linear_growth_learning_rate(
  float target, int64_t num_epochs, int64_t delay,
  std::vector<std::string> weights_names) :
  lbann_callback_learning_rate(std::move(weights_names)),
  m_target(target), m_inc(0),
  m_num_epochs(num_epochs), m_delay(delay) {}

void lbann_callback_linear_growth_learning_rate::setup(model *m) {
  lbann_callback_learning_rate::setup(m);
  // Compute the learning rate increase.
  if (!this->get_weights().empty()) {
    // Assumes all optimizers have the same initial learning rate.
    m_base_lr = get_current_global_learning_rate();
    m_inc = (m_target - m_base_lr) / m_num_epochs;
  }
}

float lbann_callback_linear_growth_learning_rate::global_schedule(model *m) {
  if (m->get_epoch() < m_delay) {
    return get_current_global_learning_rate();
  } else if (m->get_epoch() <= m_num_epochs + m_delay) {
    int num_left = m_num_epochs + m_delay - m->get_epoch();
    return m_base_lr + m_inc*(m_num_epochs - num_left);
  } else {
    return get_current_global_learning_rate();
  }
}

/**
 * This constructor takes the policy specific parameters, the exponent (p)
 * and the maximum number of iterations (max_iter).
 * In case that max_iter is set to 0, it is calculated from the number of
 * epochs (n_epochs). n_epochs is not used otherwise.
 */
lbann_callback_poly_learning_rate::lbann_callback_poly_learning_rate(
  double p, uint64_t n_epochs, uint64_t max_iter)
  : lbann_callback_learning_rate(std::vector<std::string>()),
    m_p(p), m_num_epochs(n_epochs), m_max_iter(max_iter),
    m_end_lr(0.0f),
    m_lr(1.0f), m_last_epoch_lr(1.0f) {}

lbann_callback_poly_learning_rate::lbann_callback_poly_learning_rate(
  double p, uint64_t n_epochs, uint64_t max_iter, double end_lr,  std::vector<std::string> weights_names)
  : lbann_callback_learning_rate(std::move(weights_names)),
    m_p(p), m_num_epochs(n_epochs), m_max_iter(max_iter),
    m_end_lr(end_lr),
    m_lr(1.0f), m_last_epoch_lr(1.0f) {}

/**
 * Check if the maximum number of iterations is set. If not, compute it by the
 * number of epochs and the number of iterations per epoch.
 */
void lbann_callback_poly_learning_rate::setup(model *m) {
  lbann_callback_learning_rate::setup(m);
  if (m_max_iter == 0ull) {
    m_max_iter = m_num_epochs * m->get_num_iterations_per_epoch(execution_mode::training);
  }
}

/**
 * Keep the record of the learning rate at the end of the current epoch.
 */
float lbann_callback_poly_learning_rate::global_schedule(model *m) {
  const float scale = m_lr / m_last_epoch_lr;
  m_last_epoch_lr = m_lr;
  return (get_current_global_learning_rate() - m_end_lr) * scale + m_end_lr;
}

/**
 * Compute the learning rate for the next iteration.
 */
float lbann_callback_poly_learning_rate::optimizer_schedule(model *m, optimizer &opt) {
  const uint64_t cur_iter = static_cast<uint64_t>(m->get_step(execution_mode::training));
  if (m_max_iter > cur_iter) {
    m_lr = static_cast<float>(std::pow(static_cast<double>(m_max_iter - cur_iter)/m_max_iter, m_p));
  }
  const float scale = m_lr / m_last_epoch_lr;
  return (get_current_global_learning_rate() - m_end_lr) * scale + m_end_lr;
}

lbann_callback_optimizerwise_adaptive_learning_rate::
lbann_callback_optimizerwise_adaptive_learning_rate(
  float scale) :
  lbann_callback_optimizerwise_adaptive_learning_rate(
    scale,
    std::vector<std::string>()) {}

lbann_callback_optimizerwise_adaptive_learning_rate::
lbann_callback_optimizerwise_adaptive_learning_rate(
  float scale, std::vector<std::string> weights_names) :
  lbann_callback_learning_rate(std::move(weights_names)), m_scale(scale) {}

float lbann_callback_optimizerwise_adaptive_learning_rate::optimizer_schedule(
  model *m, optimizer &opt) {
  DataType param_norm = El::Nrm2(opt.get_weights().get_values());
  DataType param_grad_norm = El::Nrm2(opt.get_gradient());
  if (param_norm > DataType(0) && param_grad_norm > DataType(0)) {
    // TODO: Should incorporate weight decay, etc. here.
    return get_current_global_learning_rate() * m_scale * param_norm / param_grad_norm;
  } else {
    return opt.get_learning_rate();
  }
}

std::unique_ptr<lbann_callback>
build_callback_step_learning_rate_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary*) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackStepLearningRate&>(proto_msg);
  return make_unique<lbann_callback_step_learning_rate>(
    params.step(),
    params.amt(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<lbann_callback>
build_callback_adaptive_learning_rate_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary*) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackAdaptiveLearningRate&>(proto_msg);
  return make_unique<lbann_callback_adaptive_learning_rate>(
    params.patience(),
    params.amt(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<lbann_callback>
build_callback_drop_fixed_learning_rate_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary*) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDropFixedLearningRate&>(proto_msg);
  std::vector<int64_t> drop_epochs;
  for (int i = 0; i < params.drop_epoch_size(); ++i) {
    drop_epochs.push_back(params.drop_epoch(i));
  }
  return make_unique<lbann_callback_drop_fixed_learning_rate>(
    std::move(drop_epochs),
    params.amt(),
    parse_list<std::string>(params.weights()));
}

std::unique_ptr<lbann_callback>
build_callback_linear_growth_learning_rate_from_pbuf(
  const google::protobuf::Message& proto_msg,lbann_summary*) {
  using MsgType = lbann_data::Callback::CallbackLinearGrowthLearningRate;
  using CallbackType = lbann_callback_linear_growth_learning_rate;
  const auto& params =
    dynamic_cast<const MsgType&>(proto_msg);
  return make_unique<CallbackType>(params.target(),
                                   params.num_epochs(),
                                   params.delay(),
                                   parse_list<std::string>(params.weights()));
}

std::unique_ptr<lbann_callback>
build_callback_optimizerwise_adaptive_learning_rate_from_pbuf(
  const google::protobuf::Message& proto_msg,lbann_summary*) {
  using MsgType = lbann_data::Callback::CallbackOptimizerwiseAdaptiveLearningRate;
  using CallbackType = lbann_callback_optimizerwise_adaptive_learning_rate;
  const auto& params = dynamic_cast<const MsgType&>(proto_msg);
  return make_unique<CallbackType>(params.scale(),
                                   parse_list<std::string>(params.weights()));
}

std::unique_ptr<lbann_callback>
build_callback_poly_learning_rate_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary*) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackPolyLearningRate&>(proto_msg);
  return make_unique<lbann_callback_poly_learning_rate>(
    params.power(),
    params.num_epochs(),
    params.max_iter(),
    params.end_lr(),
    parse_list<std::string>(params.weights()));
}

}  // namespace lbann
