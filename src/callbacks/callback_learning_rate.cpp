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
  std::unordered_set<uint> layers) : m_layer_indices(layers) {}

void lbann_callback_learning_rate::setup(model *m) {
  std::vector<Layer *>& layers = m->get_layers();
  std::unordered_set<uint> to_update = m_layer_indices;
  for (size_t l = 0; l < layers.size(); ++l) {
    Layer *layer = layers[l];
    uint idx = layer->get_index();
    // Skip layers without optimizers.
    optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(layer);
    if (!opt_layer) {
      continue;
    }
    bool has_optimizer = opt_layer->get_optimizer() != nullptr;
    if (m_layer_indices.empty() && has_optimizer) {
      to_update.insert(idx);
      m_last_idx = idx;
    } else if (m_layer_indices.find(idx) != m_layer_indices.end()) {
      if (!has_optimizer) {
        throw lbann_exception(
          "callback_learning_rate: requested layer " + std::to_string(idx) +
          " which has NULL optimizer");
      }
      m_last_idx = idx;
    }
  }
  m_layer_indices = to_update;
}

void lbann_callback_learning_rate::on_epoch_end(model *m) {
  std::vector<Layer *>& layers = m->get_layers();
  for (size_t l = 0; l < layers.size(); ++l) {
    Layer *layer = layers[l];
    uint idx = layer->get_index();
    // Skip layers without optimizers.
    optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(layer);
    if (opt_layer == NULL) {
      continue;
    }
    if (m_layer_indices.find(idx) != m_layer_indices.end()) {
      float old_lr = opt_layer->get_optimizer()->get_learning_rate();
      float new_lr = schedule(m, layer);
      if (old_lr != new_lr) {
        opt_layer->get_optimizer()->set_learning_rate(new_lr);
        lbann_comm *comm = m->get_comm();
        if (comm->am_model_master()) {
          std::cout << "Model " << comm->get_model_rank() <<
                    ": changing layer " << idx << " learning rate to " << new_lr <<
                    " at epoch " << m->get_cur_epoch() << std::endl;
        }
      }
    }
  }
}

lbann_callback_step_learning_rate::lbann_callback_step_learning_rate(
  int step, float amt) :
  lbann_callback_learning_rate(), m_step(step), m_amt(amt) {}

lbann_callback_step_learning_rate::lbann_callback_step_learning_rate(
  int step, float amt, std::unordered_set<uint> layers) :
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
                                        std::unordered_set<uint>()) {}

lbann_callback_adaptive_learning_rate::lbann_callback_adaptive_learning_rate(
  int64_t patience, float amt, std::unordered_set<uint> layers) :
  lbann_callback_learning_rate(layers), m_patience(patience), m_amt(amt) {}

float lbann_callback_adaptive_learning_rate::schedule(model *m, Layer *l) {
  optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(l);
  float cur_lr = opt_layer->get_optimizer()->get_learning_rate();
  double score = m->m_obj_fn->get_mean_value();
  if (score < m_last_score) {
    m_last_score = score;
    m_wait = 0;
  } else {
    if (m_wait >= m_patience) {
      if (is_last_layer(l)) {
        m_wait = 0;
        m_last_score = score;
      }
      return cur_lr * m_amt;
    } else {
      if (is_last_layer(l)) {
        ++m_wait;
      }
    }
  }
  return cur_lr;
}

lbann_callback_drop_fixed_learning_rate::lbann_callback_drop_fixed_learning_rate(
  std::vector<int64_t> drop_epochs, float amt) :
  lbann_callback_drop_fixed_learning_rate(drop_epochs, amt,
                                          std::unordered_set<uint>()) {}

lbann_callback_drop_fixed_learning_rate::lbann_callback_drop_fixed_learning_rate(
  std::vector<int64_t> drop_epochs, float amt, std::unordered_set<uint> layers) :
  lbann_callback_learning_rate(layers), m_amt(amt), m_drop_epochs(drop_epochs) {
  // Sort in reverse order.
  std::sort(m_drop_epochs.rbegin(), m_drop_epochs.rend());
}

float lbann_callback_drop_fixed_learning_rate::schedule(model* m, Layer *l) {
  optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(l);
  float cur_lr = opt_layer->get_optimizer()->get_learning_rate();
  if (!m_drop_epochs.empty() && m->get_cur_epoch() == m_drop_epochs.back()) {
    if (l->get_index() == m_last_idx) {
      m_drop_epochs.pop_back();
    }
    return cur_lr * m_amt;
  } else {
    return cur_lr;
  }
}

lbann_callback_linear_growth_learning_rate::lbann_callback_linear_growth_learning_rate(
  float target, int64_t num_epochs) :
  lbann_callback_linear_growth_learning_rate(target, num_epochs, 0,
                                             std::unordered_set<uint>()) {}

lbann_callback_linear_growth_learning_rate::lbann_callback_linear_growth_learning_rate(
  float target, int64_t num_epochs, int64_t delay) :
  lbann_callback_linear_growth_learning_rate(target, num_epochs, delay,
                                             std::unordered_set<uint>()) {}

lbann_callback_linear_growth_learning_rate::lbann_callback_linear_growth_learning_rate(
  float target, int64_t num_epochs, int64_t delay,
  std::unordered_set<uint> layers) :
  lbann_callback_learning_rate(layers), m_target(target), m_inc(0),
  m_num_epochs(num_epochs), m_delay(delay) {}

void lbann_callback_linear_growth_learning_rate::setup(model *m) {
  lbann_callback_learning_rate::setup(m);
  // Compute the learning rate increase.
  if (!m_layer_indices.empty()) {
    std::vector<Layer *>& layers = m->get_layers();
    // Assumes every layer has the same learning rate.
    uint idx = *m_layer_indices.begin();
    optimizable_layer *l = dynamic_cast<optimizable_layer *>(layers[idx]);
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
  std::unordered_set<uint> layers) :
  lbann_callback_learning_rate(layers), m_custom_schedule(custom_schedule) {}

float lbann_callback_custom_learning_rate::schedule(model *m, Layer *l) {
  return m_custom_schedule(m, l);
}

}  // namespace lbann
