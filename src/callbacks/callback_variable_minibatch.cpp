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
// lbann_variable_minibatch .hpp .cpp - Callback for variable-size mini-batches
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_variable_minibatch.hpp"

namespace lbann {

lbann_callback_variable_minibatch::lbann_callback_variable_minibatch(
  int starting_mbsize) : m_starting_mbsize(starting_mbsize),
                         m_current_mini_batch_size(starting_mbsize) {}

void lbann_callback_variable_minibatch::on_train_begin(model *m) {
  // Avoid issues with the train method being called multiple times.
  if (m->get_cur_epoch() != 0) {
    return;
  }
  input_layer* input = dynamic_cast<input_layer*>(m->get_layers()[0]);
  if (!input) {
    throw lbann_exception("variable_minibatch: could not get input layer");
  }
  if (m_starting_mbsize > m->get_max_mini_batch_size()) {
    throw lbann_exception(
      "variable_minibatch: starting mini-batch size is larger than max");
  }
  if (m_starting_mbsize == m->get_max_mini_batch_size()) {
    if (m->get_comm()->am_world_master()) {
      std::cout << "WARNING: starting mini-batch size equals max mini-batch "
                << "size and using variable-sized mini-batches" << std::endl;
    }
  }
  input->calculate_num_iterations_per_epoch_training_spans_models(
    m_starting_mbsize);
}

void lbann_callback_variable_minibatch::on_epoch_end(model *m) {
  input_layer* input = dynamic_cast<input_layer*>(m->get_layers()[0]);
  lbann_comm *comm = m->get_comm();
  int new_mbsize = 0;
  float new_lr = 0.0f;
  int ramp_time = 0;
  if (schedule(m, new_mbsize, new_lr, ramp_time)) {
    if (new_mbsize > m->get_max_mini_batch_size()) {
      if (comm->am_model_master()) {
        std::cout << "Model " << comm->get_model_rank() << ": WARNING " <<
          "requested new mini-batch size " << new_mbsize <<
          " is greater than the model maximum mini-batch size " <<
          m->get_max_mini_batch_size() << std::endl;
      }
      new_mbsize = m->get_max_mini_batch_size();
    }
    input->calculate_num_iterations_per_epoch_training_spans_models(new_mbsize);
    m_current_mini_batch_size = new_mbsize;
    m_ramp_count = ramp_time;
    if (new_lr != 0.0f) {
      if (ramp_time == 0) {
        // Change learning rate immediately.
        change_learning_rate(m, new_lr);
      } else {
        // Compute the per-epoch learning rate increment.
        float cur_lr = get_current_learning_rate(m);
        m_lr_incr = (new_lr - cur_lr) / ramp_time;
      }
      if (comm->am_model_master()) {
        std::cout << "Model " << comm->get_model_rank() <<
          ": Changing mini-batch size to " << new_mbsize <<
          " and learning rate to " << new_lr << " at epoch " <<
          m->get_cur_epoch() << std::endl;
      }
    } else if (comm->am_model_master()) {
      std::cout << "Model " << comm->get_model_rank() <<
        ": Changing mini-batch size to " << new_mbsize <<
        " at epoch " << m->get_cur_epoch() << std::endl;
    }
  }
  // Ramp the learning rate, if needed.
  if (m_ramp_count > 0) {
    --m_ramp_count;
    float target_lr = get_current_learning_rate(m) + m_lr_incr;
    change_learning_rate(m, target_lr);
    if (comm->am_model_master()) {
      std::cout << "Model " << comm->get_model_rank() <<
        ": Variable-size mini-batch ramping learning rate to " <<
        target_lr << std::endl;
    }
  }
}

void lbann_callback_variable_minibatch::change_learning_rate(
  model *m, float new_lr) const {
  std::vector<Layer*>& layers = m->get_layers();
  for (size_t l = 0; l < layers.size(); ++l) {
    optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(layers[l]);
    if (opt_layer == nullptr) {
      continue;
    }
    optimizer *opt = opt_layer->get_optimizer();
    if (opt != nullptr) {
      opt->set_learning_rate(new_lr);
    }
  }
}

float lbann_callback_variable_minibatch::get_current_learning_rate(
  model *m) const {
  std::vector<Layer*>& layers = m->get_layers();
  for (size_t l = 0; l < layers.size(); ++l) {
    optimizable_layer *opt_layer = dynamic_cast<optimizable_layer*>(layers[l]);
    if (opt_layer == nullptr) {
      continue;
    }
    optimizer *opt = opt_layer->get_optimizer();
    if (opt != nullptr) {
      return opt->get_learning_rate();
    }
  }
  return 0.0f;
}

lbann_callback_step_minibatch::lbann_callback_step_minibatch(
  int starting_mbsize, int step, int ramp_time) :
  lbann_callback_variable_minibatch(starting_mbsize), m_step(step),
  m_ramp_time(ramp_time) {}

bool lbann_callback_step_minibatch::schedule(
  model *m, int& new_mbsize, float& new_lr, int& ramp_time) {
  if (m->get_cur_epoch() % m_step == 0) {
    new_mbsize = m_current_mini_batch_size * 2;
    new_lr = get_current_learning_rate(m) * 2;
    ramp_time = m_ramp_time;
    return true;
  } else {
    return false;
  }
}

lbann_callback_minibatch_schedule::lbann_callback_minibatch_schedule(
  int starting_mbsize, std::vector<minibatch_step> steps) :
  lbann_callback_variable_minibatch(starting_mbsize), m_steps(steps) {
  std::sort(m_steps.rbegin(), m_steps.rend(),
            [] (const minibatch_step& a, const minibatch_step& b) {
              return a.epoch < b.epoch;
            });
}

bool lbann_callback_minibatch_schedule::schedule(
  model *m, int& new_mbsize, float& new_lr, int& ramp_time) {
  if (!m_steps.empty() && m->get_cur_epoch() == m_steps.back().epoch) {
    new_mbsize = m_steps.back().mbsize;
    new_lr = m_steps.back().lr;
    ramp_time = m_steps.back().ramp_time;
    m_steps.pop_back();
    return true;
  }
  return false;
}

}  // namespace lbann
