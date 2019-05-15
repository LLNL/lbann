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
// lbann_callback_summary .hpp .cpp - Callback hooks to summarize to Tensorboard
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_summary.hpp"
#include "lbann/utils/profiling.hpp"

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
  prof_region_begin("summary-batch", prof_colors[0], false);
  m->summarize_stats(*m_summarizer);
  if (m_mat_interval > 0 && m->get_step(execution_mode::training) % m_mat_interval == 0) {
    m->summarize_matrices(*m_summarizer);
  }
  lbann_comm *comm = m->get_comm();
  size_t bytes_sent = comm->get_bytes_sent();
  size_t bytes_received = comm->get_bytes_received();
  size_t trainer_barriers = comm->get_num_trainer_barriers();
  size_t intertrainer_barriers = comm->get_num_intertrainer_barriers();
  size_t global_barriers = comm->get_num_global_barriers();
  comm->reset_stats_counters();
  m_summarizer->sum_reduce_scalar("bytes_sent", bytes_sent, m->get_step(execution_mode::training));
  m_summarizer->sum_reduce_scalar("bytes_received", bytes_received,
                                  m->get_step(execution_mode::training));
  m_summarizer->reduce_scalar("trainer_barriers", trainer_barriers,
                              m->get_step(execution_mode::training));
  m_summarizer->reduce_scalar("intertrainer_barriers", intertrainer_barriers,
                              m->get_step(execution_mode::training));
  m_summarizer->reduce_scalar("global_barriers", global_barriers,
                              m->get_step(execution_mode::training));
  prof_region_end("summary-batch", false);
}

void lbann_callback_summary::on_epoch_end(model *m) {
  prof_region_begin("summary-epoch", prof_colors[0], false);
  for (const auto& met : m->get_metrics()) {
    EvalType train_score = met->get_mean_value(m->get_execution_mode());
    // Replace spaces with _ for consistency.
    std::string metric_name = met->name();
    std::transform(metric_name.begin(), metric_name.end(), metric_name.begin(),
                   [] (char c) { return c == ' ' ? '_' : c; });
    std::string phase = "train_" + metric_name;
    m_summarizer->reduce_scalar(phase, train_score, m->get_step(execution_mode::training));
  }
  save_histograms(m);
  m_summarizer->flush();
  prof_region_end("summary-epoch", false);
}

void lbann_callback_summary::on_test_end(model *m) {
  prof_region_begin("summary-test", prof_colors[0], false);
  lbann_comm *comm = m->get_comm();
  for (auto&& met : m->get_metrics()) {
    EvalType test_score = met->get_mean_value(m->get_execution_mode());
    // Replace spaces with _ for consistency.
    std::string metric_name = met->name();
    std::transform(metric_name.begin(), metric_name.end(), metric_name.begin(),
                   [] (char c) { return c == ' ' ? '_' : c; });
    std::string phase = "test_" + metric_name;
    m_summarizer->reduce_scalar(phase, test_score, m->get_step(execution_mode::training));
  }
  // Reset counters incremented during test phase.
  comm->reset_stats_counters();
  for (auto&& layer : m->get_layers()) {
    layer->reset_counters();
  }
  prof_region_end("summary-test", false);
}

void lbann_callback_summary::save_histograms(model *m) {
  for (const auto& layer : m->get_layers()) {
    const std::string prefix = layer->get_name() + "/";
    for (int i = 0; i < layer->get_num_children(); ++i) {
      AbsDistMatReadProxy<El::Device::CPU> acts(layer->get_activations(i));
      m_summarizer->reduce_histogram(prefix + "activations" + std::to_string(i),
                                     acts.GetLocked(),
                                     m->get_step(execution_mode::training));
    }
  }
  for (const auto& w : m->get_weights()) {
    const std::string prefix = w->get_name() + "/";
    AbsDistMatReadProxy<El::Device::CPU> weights(w->get_values());
    m_summarizer->reduce_histogram(prefix + "weights",
                                   weights.GetLocked(),
                                   m->get_step(execution_mode::training));
    optimizer *opt = w->get_optimizer();
    if (opt != nullptr) {
      AbsDistMatReadProxy<El::Device::CPU> gradients(opt->get_gradient());
      m_summarizer->reduce_histogram(prefix + "weights_gradient",
                                     gradients.GetLocked(),
                                     m->get_step(execution_mode::training));
    }
  }
}

}  // namespace lbann
