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

#include "lbann/callbacks/callback_learning_rate.hpp"
#include <limits>

namespace lbann {

lbann_callback_learning_rate::lbann_callback_learning_rate() {}

lbann_callback_learning_rate::lbann_callback_learning_rate(
  std::unordered_set<Layer *> layers) : m_layers(layers) {}

void lbann_callback_learning_rate::setup(model *m) {

  // Add all layers if list of layers is not initialized
  std::vector<Layer *> all_layers;
  if (m_layers.empty()) {
    all_layers = m->get_layers();
  } else {
    for (Layer *layer : m_layers) {
      all_layers.push_back(layer);
    }
  }

  // Remove layers that are not optimizable
  m_layers.clear();
  for (Layer *layer : all_layers) {
    optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(layer);
    if (opt_layer == nullptr) {
      continue;
    } else if (opt_layer->get_optimizer() == nullptr) {
      throw lbann_exception(
                            "callback_learning_rate: requested layer " + layer->get_name() +
                            " which has NULL optimizer");
    }
    m_layers.insert(layer);
  }

}

void lbann_callback_learning_rate::on_epoch_end(model *m) {
  for (Layer *layer : m_layers) {
    optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(layer);
    const float old_lr = opt_layer->get_optimizer()->get_learning_rate();
    const float new_lr = schedule(m, layer);
    if (old_lr != new_lr) {
      opt_layer->get_optimizer()->set_learning_rate(new_lr);
      lbann_comm *comm = m->get_comm();
      if (comm->am_model_master()) {
        std::cout << "Model " << comm->get_model_rank() << ": "
                  << "changing layer " << layer->get_name() << " "
                  << "learning rate to " << new_lr << " "
                  << "at epoch " << m->get_cur_epoch() << std::endl;
      }
    }
  }
}

lbann_callback_step_learning_rate::lbann_callback_step_learning_rate(
  int step, float amt) :
  lbann_callback_learning_rate(), m_step(step), m_amt(amt) {}

lbann_callback_step_learning_rate::lbann_callback_step_learning_rate(
  int step, float amt, std::unordered_set<Layer *> layers) :
  lbann_callback_learning_rate(layers), m_step(step), m_amt(amt) {}

float lbann_callback_step_learning_rate::schedule(model *m, Layer *l) {
  optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(l);
  float cur_lr = opt_layer->get_optimizer()->get_learning_rate();
  if (m->get_cur_epoch() % m_step == 0) {
    return cur_lr * m_amt;
  } else {
    return cur_lr;
  }
}

lbann_callback_adaptive_learning_rate::lbann_callback_adaptive_learning_rate(
  int64_t patience, float amt) :
  lbann_callback_adaptive_learning_rate(patience, amt,
                                        std::unordered_set<Layer *>()) {}

lbann_callback_adaptive_learning_rate::lbann_callback_adaptive_learning_rate(
  int64_t patience, float amt, std::unordered_set<Layer *> layers) :
  lbann_callback_learning_rate(layers), m_patience(patience), m_amt(amt) {}

float lbann_callback_adaptive_learning_rate::schedule(model *m, Layer *l) {
  optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(l);

  // Determine behavior the first time this is called in an epoch
  if (m_cur_epoch != m->get_cur_epoch()) {
    m_cur_epoch = m->get_cur_epoch();
    const double score = m->m_obj_fn->get_mean_value();
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
  const float cur_lr = opt_layer->get_optimizer()->get_learning_rate();
  if (m_adjust_learning_rate) {
    return cur_lr * m_amt;
  } else {
    return cur_lr;
  }

}

lbann_callback_drop_fixed_learning_rate::lbann_callback_drop_fixed_learning_rate(
  std::vector<int64_t> drop_epochs, float amt) :
  lbann_callback_drop_fixed_learning_rate(drop_epochs, amt,
                                          std::unordered_set<Layer *>()) {}

lbann_callback_drop_fixed_learning_rate::lbann_callback_drop_fixed_learning_rate(
  std::vector<int64_t> drop_epochs, float amt, std::unordered_set<Layer *> layers) :
  lbann_callback_learning_rate(layers), m_amt(amt), m_drop_epochs(drop_epochs) {
  // Sort in reverse order.
  std::sort(m_drop_epochs.rbegin(), m_drop_epochs.rend());
}

float lbann_callback_drop_fixed_learning_rate::schedule(model* m, Layer *l) {

  // Delete last drop epoch if we have already passed it
  while (!m_drop_epochs.empty()
         && m->get_cur_epoch() > m_drop_epochs.back()) {
    m_drop_epochs.pop_back();
  }

  // Adjust learning rate if at a drop epoch
  optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(l);
  float cur_lr = opt_layer->get_optimizer()->get_learning_rate();
  if (!m_drop_epochs.empty() && m->get_cur_epoch() == m_drop_epochs.back()) {
    return cur_lr * m_amt;
  } else {
    return cur_lr;
  }

}

lbann_callback_linear_growth_learning_rate::lbann_callback_linear_growth_learning_rate(
  float target, int64_t num_epochs) :
  lbann_callback_linear_growth_learning_rate(target, num_epochs, 0,
                                             std::unordered_set<Layer *>()) {}

lbann_callback_linear_growth_learning_rate::lbann_callback_linear_growth_learning_rate(
  float target, int64_t num_epochs, int64_t delay) :
  lbann_callback_linear_growth_learning_rate(target, num_epochs, delay,
                                             std::unordered_set<Layer *>()) {}

lbann_callback_linear_growth_learning_rate::lbann_callback_linear_growth_learning_rate(
  float target, int64_t num_epochs, int64_t delay,
  std::unordered_set<Layer *> layers) :
  lbann_callback_learning_rate(layers), m_target(target), m_inc(0),
  m_num_epochs(num_epochs), m_delay(delay) {}

void lbann_callback_linear_growth_learning_rate::setup(model *m) {
  lbann_callback_learning_rate::setup(m);
  // Compute the learning rate increase.
  if (!m_layers.empty()) {
    // Assumes every layer has the same learning rate.
    optimizable_layer *l = dynamic_cast<optimizable_layer *>(*m_layers.begin());
    float base_lr = l->get_optimizer()->get_learning_rate();
    m_inc = (m_target - base_lr) / m_num_epochs;
  }
}

float lbann_callback_linear_growth_learning_rate::schedule(model *m,
                                                           Layer *l) {
  optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(l);
  float cur_lr = opt_layer->get_optimizer()->get_learning_rate();
  if (m->get_cur_epoch() < m_delay) {
    return cur_lr;
  } else if (m->get_cur_epoch() <= m_num_epochs + m_delay) {
    return cur_lr + m_inc;
  } else {
    return cur_lr;
  }
}

lbann_callback_custom_learning_rate::lbann_callback_custom_learning_rate(
  std::function<float(model *, Layer *)> custom_schedule) :
  lbann_callback_learning_rate(), m_custom_schedule(custom_schedule) {}

lbann_callback_custom_learning_rate::lbann_callback_custom_learning_rate(
  std::function<float(model *, Layer *)> custom_schedule,
  std::unordered_set<Layer *> layers) :
  lbann_callback_learning_rate(layers), m_custom_schedule(custom_schedule) {}

float lbann_callback_custom_learning_rate::schedule(model *m, Layer *l) {
  return m_custom_schedule(m, l);
}

}  // namespace lbann
