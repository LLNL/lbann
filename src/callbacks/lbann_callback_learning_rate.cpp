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

#include "lbann/callbacks/lbann_callback_learning_rate.hpp"
#include <limits>

namespace lbann {

lbann_callback_learning_rate::lbann_callback_learning_rate() {}

lbann_callback_learning_rate::lbann_callback_learning_rate(
  std::unordered_set<uint> _layers) : layer_indices(_layers) {}

void lbann_callback_learning_rate::setup(model* m) {
  std::vector<Layer*>& layers = m->get_layers();
  for (size_t l = 0; l < layers.size(); ++l) {
    Layer* layer = layers[l];
    uint idx = layer->get_index();
    if (layer_indices.size() == 0 ||
        layer_indices.find(idx) != layer_indices.end()) {
      if (layer->get_optimizer() != NULL) {
        old_lrs[idx] = layer->get_optimizer()->get_learning_rate();
        last_idx = idx;
      }
    }
  }
}

void lbann_callback_learning_rate::on_epoch_begin(model* m) {
  std::vector<Layer*>& layers = m->get_layers();
  for (size_t l = 0; l < layers.size(); ++l) {
    Layer* layer = layers[l];
    uint idx = layer->get_index();
    if (old_lrs.find(idx) != old_lrs.end()) {
      float new_lr = schedule(m, layer);
      if (new_lr != old_lrs[idx]) {
        old_lrs[idx] = new_lr;
        layer->get_optimizer()->set_learning_rate(new_lr);
        lbann_comm* comm = m->get_comm();
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
  int step, float amt) : lbann_callback_learning_rate(), step(step), amt(amt) {}

lbann_callback_step_learning_rate::lbann_callback_step_learning_rate(
  int step, float amt, std::unordered_set<uint> _layers) :
  lbann_callback_learning_rate(_layers), step(step), amt(amt) {}

float lbann_callback_step_learning_rate::schedule(model* m, Layer* l) {
  float cur_lr = l->get_optimizer()->get_learning_rate();
  if (m->get_cur_epoch() % step == 0) {
    return cur_lr * amt;
  } else {
    return cur_lr;
  }
}

lbann_callback_acc_learning_rate::lbann_callback_acc_learning_rate(
  int64_t patience, float amt) :
  lbann_callback_acc_learning_rate(patience, amt,
                                   std::unordered_set<uint>()) {}

lbann_callback_acc_learning_rate::lbann_callback_acc_learning_rate(
  int64_t patience, float amt, std::unordered_set<uint> _layers) :
  lbann_callback_learning_rate(_layers), patience(patience), amt(amt),
  last_acc(std::numeric_limits<DataType>::lowest()), wait(0) {}

float lbann_callback_acc_learning_rate::schedule(model* m, Layer* l) {
  DataType cur_acc = m->get_test_accuracy();
  float cur_lr = l->get_optimizer()->get_learning_rate();
  if (cur_acc > last_acc) {
    last_acc = cur_acc;
    wait = 0;
  } else {
    if (wait >= patience) {
      if (is_last_layer(l)) {
        wait = 0;
        last_acc = cur_acc;
      }
      return cur_lr * amt;
    } else {
      if (is_last_layer(l)) {
        ++wait;
      }
    }
  }
  return cur_lr;
}

lbann_callback_custom_learning_rate::lbann_callback_custom_learning_rate(
  std::function<float(model*, Layer*)> custom_schedule) :
  lbann_callback_learning_rate(), custom_schedule(custom_schedule) {}

lbann_callback_custom_learning_rate::lbann_callback_custom_learning_rate(
  std::function<float(model*, Layer*)> custom_schedule,
  std::unordered_set<uint> _layers) :
  lbann_callback_learning_rate(_layers), custom_schedule(custom_schedule) {}

float lbann_callback_custom_learning_rate::schedule(model* m, Layer* l) {
  return custom_schedule(m, l);
}

}  // namespace lbann
