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
// lbann_callback_summary .hpp .cpp - Callback hooks to summarize to Tensorboard
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_summary.hpp"

namespace lbann {

lbann_callback_summary::lbann_callback_summary(lbann_summary *summarizer,
                                               int batch_interval,
                                               int mat_interval) :
  lbann_callback(batch_interval, summarizer),
  m_mat_interval(mat_interval) {}

lbann_callback_summary::~lbann_callback_summary() {
  delete m_summarizer;
}

void lbann_callback_summary::on_train_begin(model *m) {
  save_histograms(m);
}

void lbann_callback_summary::on_batch_end(model *m) {
  m->summarize_stats(*m_summarizer);
  if (m->get_cur_step() % m_mat_interval == 0) {
    m->summarize_matrices(*m_summarizer);
  }
  lbann_comm *comm = m->get_comm();
  size_t bytes_sent = comm->get_bytes_sent();
  size_t bytes_received = comm->get_bytes_received();
  size_t model_barriers = comm->get_num_model_barriers();
  size_t intermodel_barriers = comm->get_num_intermodel_barriers();
  size_t global_barriers = comm->get_num_global_barriers();
  comm->reset_stats_counters();
  m_summarizer->sum_reduce_scalar("bytes_sent", bytes_sent, m->get_cur_step());
  m_summarizer->sum_reduce_scalar("bytes_received", bytes_received,
                                  m->get_cur_step());
  m_summarizer->reduce_scalar("model_barriers", model_barriers,
                              m->get_cur_step());
  m_summarizer->reduce_scalar("intermodel_barriers", intermodel_barriers,
                              m->get_cur_step());
  m_summarizer->reduce_scalar("global_barriers", global_barriers,
                              m->get_cur_step());
}

void lbann_callback_summary::on_epoch_end(model *m) {
  for (auto&& metric : m->get_metrics()) {
    double train_score = metric->report_metric(execution_mode::training);
    // Replace spaces with _ for consistency.
    std::string metric_name = metric->name();
    std::transform(metric_name.begin(), metric_name.end(), metric_name.begin(),
                   [] (char c) { return c == ' ' ? '_' : c; });
    std::string phase = "train_" + metric_name;
    m_summarizer->reduce_scalar(phase, train_score, m->get_cur_step());
  }
  save_histograms(m);
  m_summarizer->flush();
}

void lbann_callback_summary::on_test_end(model *m) {
  lbann_comm *comm = m->get_comm();
  for (auto&& metric : m->get_metrics()) {
    double test_score = metric->report_metric(execution_mode::testing);
    // Replace spaces with _ for consistency.
    std::string metric_name = metric->name();
    std::transform(metric_name.begin(), metric_name.end(), metric_name.begin(),
                   [] (char c) { return c == ' ' ? '_' : c; });
    std::string phase = "test_" + metric_name;
    m_summarizer->reduce_scalar(phase, test_score, m->get_cur_step());
  }
  // Reset counters incremented during test phase.
  comm->reset_stats_counters();
  for (auto&& layer : m->get_layers()) {
    layer->reset_counters();
  }
}

void lbann_callback_summary::save_histograms(model *m) {
  for (const auto& layer : m->get_layers()) {
    std::string prefix = "layer" + std::to_string(layer->get_index()) +
      "/";
    m_summarizer->reduce_histogram(prefix + "activations",
                                   layer->get_activations(),
                                   m->get_cur_step());
    learning *learning_layer = (learning *) dynamic_cast<learning *> (layer);
    if(learning_layer != nullptr) {
      m_summarizer->reduce_histogram(prefix + "weights",
                                     learning_layer->get_weights(),
                                     m->get_cur_step());
      m_summarizer->reduce_histogram(
        prefix + "weights_gradient",
        learning_layer->get_weights_gradient(),
        m->get_cur_step());
    }
  }
}

}  // namespace lbann
