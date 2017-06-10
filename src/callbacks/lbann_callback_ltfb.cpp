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
// lbann_callback_ltfb .hpp .cpp - Manage LTFB training for a model
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/lbann_callback_ltfb.hpp"
#include "lbann/models/lbann_model_dnn.hpp"

namespace lbann {

lbann_callback_ltfb::lbann_callback_ltfb(
  uint round_size, model *remote_model, lbann_summary *summarizer) :
  lbann_callback(1, summarizer), m_round_size(round_size),
  m_remote_model(remote_model) {
}

lbann_callback_ltfb::~lbann_callback_ltfb() {

}

void lbann_callback_ltfb::setup(model *m) {
  m_comm = m->get_comm();
  // Validate that the round size divides the number of minibatches.
  // Duplicate model.
}

void lbann_callback_ltfb::on_batch_end(model *m) {
  if (m->get_cur_step() % m_round_size != 0) {
    return;  // Not the end of a round.
  }

  int partner = select_partner();
  if (partner == m_comm->get_model_rank()) {
    // No partner this round, skip.
    return;
  }
  // Transfer data to secondary model.
  exchange(m, partner);
  // Evaluate on tournament data.
  // Have to cast, assumes deep_neural_network.
  deep_neural_network *dnn = static_cast<deep_neural_network *>(m);
  deep_neural_network *remote_dnn =
    static_cast<deep_neural_network *>(m_remote_model);
  dnn->evaluate(execution_mode::validation);
  remote_dnn->evaluate(execution_mode::validation);
  // Reset the execution mode.
  m->set_execution_mode(execution_mode::training);
  m_remote_model->set_execution_mode(execution_mode::training);
  // Assume there is a categorical accuracy metric.
  double local_acc = 0;
  double remote_acc = 0;
  for (auto&& metric : m->metrics) {
    if (metric->type == metrics::metric_type::categorical_accuracy) {
      local_acc = metric->report_metric(execution_mode::validation);
      break;
    }
  }
  for (auto&& metric : m_remote_model->metrics) {
    if (metric->type == metrics::metric_type::categorical_accuracy) {
      remote_acc = metric->report_metric(execution_mode::validation);
      break;
    }
  }
  // If the remote is better, keep it.
  if (remote_acc > local_acc) {
    if (m_comm->am_model_master()) {
      std::cout << m_comm->get_model_rank() << ": (step " << m->get_cur_step() << ") Replacing local model (" << local_acc << ") with " <<
                "model " << partner << " (" << remote_acc << ")" << std::endl;
    }
    replace_with_remote(m);
  }
}

int lbann_callback_ltfb::select_partner() {
  int my_partner = 0;
  // Master generates partners for everyone.
  if (m_comm->am_world_master()) {
    std::vector<int> ranks(m_comm->get_num_models());
    std::iota(ranks.begin(), ranks.end(), 0);
    std::shuffle(ranks.begin(), ranks.end(), get_fast_generator());
    // Adjacent pairs become partners.
    // Dilate so that we can do one scatter to every process.
    std::vector<int> partners(
      m_comm->get_num_models() * m_comm->get_procs_per_model());
    for (size_t i = 0; i < (ranks.size() & ~1); i += 2) {
      int rank1 = ranks[i];
      int rank2 = ranks[i+1];
      std::fill_n(partners.begin() + rank1*m_comm->get_procs_per_model(),
                  m_comm->get_procs_per_model(), rank2);
      std::fill_n(partners.begin() + rank2*m_comm->get_procs_per_model(),
                  m_comm->get_procs_per_model(), rank1);
    }
    // Handle the last rank if needed.
    if (partners.size() % 2 != 0) {
      int last_rank = ranks[ranks.size() - 1];
      std::fill_n(partners.begin() + last_rank*m_comm->get_procs_per_model(),
                  m_comm->get_procs_per_model(), last_rank);
    }
    El::mpi::Scatter(partners.data(), 1, &my_partner, 1, 0,
                     El::mpi::COMM_WORLD);
  } else {
    El::mpi::Scatter((int *) nullptr, 0, &my_partner, 1, 0, El::mpi::COMM_WORLD);
  }
  return my_partner;
}

void lbann_callback_ltfb::exchange(model *m, int partner) {
  std::vector<Layer *>& layers = m->get_layers();
  std::vector<Layer *>& remote_layers = m_remote_model->get_layers();
  // Skip input/target layers.
  for (size_t i = 1; i < layers.size() - 1; ++i) {
    Layer *layer = layers[i];
    Layer *remote_layer = remote_layers[i];
    // TODO: Support sending optimizer state.
    ElMat& weights = layer->get_weights_biases();
    ElMat& remote_weights = remote_layer->get_weights_biases();
    if (weights.Height() > 0) {
      m_comm->sendrecv(weights.LockedBuffer(),
                       weights.LocalHeight()*weights.LocalWidth(),
                       partner,
                       remote_weights.Buffer(),
                       weights.LocalHeight()*weights.LocalWidth(),
                       partner);
    }
  }
}

void lbann_callback_ltfb::replace_with_remote(model *m) {
  std::vector<Layer *>& layers = m->get_layers();
  std::vector<Layer *>& remote_layers = m_remote_model->get_layers();
  // Skip input/target layers.
  for (size_t i = 1; i < layers.size() - 1; ++i) {
    Layer *layer = layers[i];
    Layer *remote_layer = remote_layers[i];
    // TODO: Update optimizers.
    layer->get_weights_biases().Matrix() =
      remote_layer->get_weights_biases().Matrix();
  }
}

}  // namespace lbann
