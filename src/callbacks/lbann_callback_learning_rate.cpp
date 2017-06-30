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
  for (size_t l = 0; l < layers.size(); ++l) {
    Layer *layer = layers[l];
    uint idx = layer->get_index();
    // Skip non-learning layers.
    learning *learning_layer = (learning *) dynamic_cast<learning *> (layer);
    if(learning_layer == NULL) {
      continue;
    }
    if (m_layer_indices.size() == 0 ||
        m_layer_indices.find(idx) != m_layer_indices.end()) {
      if (learning_layer->get_optimizer() != NULL) {
        m_old_lrs[idx] = learning_layer->get_optimizer()->get_learning_rate();
        m_last_idx = idx;
      }
    }
  }
}

void lbann_callback_learning_rate::on_epoch_end(model *m) {
  std::vector<Layer *>& layers = m->get_layers();
  for (size_t l = 0; l < layers.size(); ++l) {
    Layer *layer = layers[l];
    uint idx = layer->get_index();
    // Skip non-learning layers.
    learning *learning_layer = (learning *) dynamic_cast<learning *> (layer);
    if(learning_layer == NULL) {
      continue;
    }
    if (m_old_lrs.find(idx) != m_old_lrs.end()) {
      float new_lr = schedule(m, layer);
      if (new_lr != m_old_lrs[idx]) {
        m_old_lrs[idx] = new_lr;
        learning_layer->get_optimizer()->set_learning_rate(new_lr);
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
  // Skip non-learning layers.
  learning *learning_layer = (learning *) dynamic_cast<learning *> (l);
  if(learning_layer == NULL) {
    return 0;
  }
  float cur_lr = learning_layer->get_optimizer()->get_learning_rate();
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

/// Monitor the objective function to see if the validation score
/// continues to improve
float lbann_callback_adaptive_learning_rate::schedule(model *m, Layer *l) {
  // Skip non-learning layers.
  learning *learning_layer = (learning *) dynamic_cast<learning *> (l);
  if(learning_layer == NULL) {
    return 0;
  }
  float cur_lr = learning_layer->get_optimizer()->get_learning_rate();
  double score = m->m_obj_fn->report_aggregate_avg_obj_fn(execution_mode::validation);
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
