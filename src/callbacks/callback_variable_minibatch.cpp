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
  input->calculate_num_iterations_per_epoch_training_spans_models(
    m_starting_mbsize);
}

void lbann_callback_variable_minibatch::on_epoch_end(model *m) {
  input_layer* input = dynamic_cast<input_layer*>(m->get_layers()[0]);
  int new_mbsize = change_minibatch_size(m);
  if (new_mbsize != m_current_mini_batch_size &&
      new_mbsize <= m->get_max_mini_batch_size()) {
    input->calculate_num_iterations_per_epoch_training_spans_models(new_mbsize);
    std::vector<Layer*>& layers = m->get_layers();
    // Determine the current learning rate.
    float cur_lr = 0;
    for (size_t l = 0; l < layers.size(); ++l) {
      // Skip non-learning layers.
      learning *learning_layer = dynamic_cast<learning*>(layers[l]);
      if (learning_layer == NULL) {
        continue;
      }
      optimizer *opt = learning_layer->get_optimizer();
      if (opt) {
        cur_lr = opt->get_learning_rate();
        break;
      }
    }
    // Potentially update the layers' learning rate.
    float new_lr = change_learning_rate(m, cur_lr, m_current_mini_batch_size,
                                        new_mbsize);
    for (size_t l = 0; l < layers.size(); ++l) {
      // Skip non-learning layers.
      learning *learning_layer = dynamic_cast<learning*>(layers[l]);
      if (learning_layer == NULL) {
        continue;
      }
      optimizer *opt = learning_layer->get_optimizer();
      if (opt) {
        opt->set_learning_rate(new_lr);
      }
    }
    lbann_comm *comm = m->get_comm();
    if (comm->am_model_master()) {
      std::cout << "Model " << comm->get_model_rank() <<
        ": changing mini-batch size to " << new_mbsize <<
        " and learning rate to " << new_lr << " at epoch " <<
        m->get_cur_epoch() << std::endl;
    }
    // Update our tracking of the mini-batch size.
    m_current_mini_batch_size = new_mbsize;
  }
}

lbann_callback_step_minibatch::lbann_callback_step_minibatch(
  int starting_mbsize, int step) :
  lbann_callback_variable_minibatch(starting_mbsize), m_step(step) {}

int lbann_callback_step_minibatch::change_minibatch_size(model *m) {
  if (m->get_cur_epoch() % m_step == 0) {
    return m_current_mini_batch_size * 2;
  } else {
    return m_current_mini_batch_size;
  }
}

float lbann_callback_step_minibatch::change_learning_rate(
  model *m, float cur_lr, int old_mbsize, int new_mbsize) {
  return cur_lr * ((float) new_mbsize / (float) old_mbsize);
}

}  // namespace lbann
