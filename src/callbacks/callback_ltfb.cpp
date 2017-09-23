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

#include "lbann/callbacks/callback_ltfb.hpp"
#include "lbann/metrics/metric_categorical_accuracy.hpp"
#include <typeinfo>
#include <typeindex>

namespace lbann {

lbann_callback_ltfb::lbann_callback_ltfb(
  uint round_size, lbann_summary *summarizer) :
  lbann_callback(1, summarizer), m_round_size(round_size) {}

lbann_callback_ltfb::lbann_callback_ltfb(const lbann_callback_ltfb& other) :
  lbann_callback(other),
  m_comm(other.m_comm),
  m_round_size(other.m_round_size) {
  if (other.m_remote_model) {
    m_remote_model = other.m_remote_model->copy();
  }
}

lbann_callback_ltfb& lbann_callback_ltfb::operator=(
  const lbann_callback_ltfb& other) {
  m_comm = other.m_comm;
  m_round_size = other.m_round_size;
  if (m_remote_model) {
    delete m_remote_model;
    m_remote_model = nullptr;
  }
  if (other.m_remote_model) {
    m_remote_model = other.m_remote_model->copy();
  }
  return *this;
}

lbann_callback_ltfb::~lbann_callback_ltfb() {

}

void lbann_callback_ltfb::setup(model *m) {
  m_comm = m->get_comm();
  // TODO: Validate that the round size divides the number of minibatches.
  // Duplicate the model so we have a replica for storing the remote model in.
  m_remote_model = m->copy();
}

void lbann_callback_ltfb::on_batch_end(model *m) {
  if (m->get_cur_step() % m_round_size != 0 ||
      m->get_cur_step() == 0) {
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
  deep_neural_network *dnn = dynamic_cast<deep_neural_network *>(m);
  deep_neural_network *remote_dnn =
    dynamic_cast<deep_neural_network *>(m_remote_model);
  double local_acc = evaluate(dnn);
  double remote_acc = evaluate(remote_dnn);
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
  for (size_t i = 0; i < layers.size(); ++i) {
    // Only exchange learning layers.
    learning *layer = dynamic_cast<learning*>(layers[i]);
    if (layer) {
      learning *remote_layer = dynamic_cast<learning*>(remote_layers[i]);
      // TODO: Support sending optimizer state.
      AbsDistMat& weights = layer->get_weights();
      AbsDistMat& remote_weights = remote_layer->get_weights();
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
}

double lbann_callback_ltfb::evaluate(deep_neural_network *m) {
  m->evaluate(execution_mode::validation);
  m->set_execution_mode(execution_mode::training);  // Reset execution mode.
  double acc = 0;
  const std::type_info& ca_model_type = typeid(
    metrics::categorical_accuracy<data_layout::MODEL_PARALLEL>);
  const std::type_info& ca_data_type = typeid(
    metrics::categorical_accuracy<data_layout::DATA_PARALLEL>);
  for (auto&& metric : m->get_metrics()) {
    const std::type_info& m_type = typeid(*metric);
    if (std::type_index(ca_model_type) == std::type_index(m_type) ||
        std::type_index(ca_data_type) == std::type_index(m_type)) {
      acc = metric->report_metric(execution_mode::validation);
      break;
    }
  }
  return acc;
}

void lbann_callback_ltfb::replace_with_remote(model *m) {
  std::vector<Layer *>& layers = m->get_layers();
  std::vector<Layer *>& remote_layers = m_remote_model->get_layers();
  for (size_t i = 0; i < layers.size(); ++i) {
    learning *layer = dynamic_cast<learning*>(layers[i]);
    if (layer) {
      learning *remote_layer = dynamic_cast<learning*>(remote_layers[i]);
      // TODO: Update optimizers.
      layer->get_weights().Matrix() =
        remote_layer->get_weights().Matrix();
    }
  }
}

}  // namespace lbann
