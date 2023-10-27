////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
// summary .hpp .cpp - Callback hooks to summarize to Tensorboard
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/summary.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include "lbann/utils/memory.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/summary_impl.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <algorithm>
#include <string>

namespace lbann {
namespace callback {

summary::summary(const std::shared_ptr<lbann_summary>& summarizer,
                 int batch_interval,
                 int mat_interval)
  : callback_base(batch_interval),
    m_summarizer(summarizer),
    m_mat_interval(mat_interval)
{}

void summary::on_train_begin(model* m) { save_histograms(m); }

void summary::on_batch_end(model* m)
{
  if (!m_summarizer) {
    LBANN_ERROR("Summary callback failed: m_summarizer does not exist.");
  }

  const auto& c = m->get_execution_context();

  prof_region_begin("summary-batch", prof_colors[0], false);
  m->summarize_stats(*m_summarizer);
  if (m_mat_interval > 0 && c.get_step() % m_mat_interval == 0) {
    m->summarize_matrices(*m_summarizer);
  }
  lbann_comm* comm = m->get_comm();
  size_t bytes_sent = comm->get_bytes_sent();
  size_t bytes_received = comm->get_bytes_received();
  size_t trainer_barriers = comm->get_num_trainer_barriers();
  size_t intertrainer_barriers = comm->get_num_intertrainer_barriers();
  size_t global_barriers = comm->get_num_global_barriers();
  comm->reset_stats_counters();
  m_summarizer->sum_reduce_scalar("bytes_sent", bytes_sent, c.get_step());
  m_summarizer->sum_reduce_scalar("bytes_received",
                                  bytes_received,
                                  c.get_step());
  m_summarizer->reduce_scalar("trainer_barriers",
                              trainer_barriers,
                              c.get_step());
  m_summarizer->reduce_scalar("intertrainer_barriers",
                              intertrainer_barriers,
                              c.get_step());
  m_summarizer->reduce_scalar("global_barriers", global_barriers, c.get_step());
  prof_region_end("summary-batch", false);
}

void summary::on_epoch_end(model* m)
{
  if (!m_summarizer) {
    LBANN_ERROR("Summary callback failed: m_summarizer does not exist.");
  }

  const auto& c = m->get_execution_context();
  prof_region_begin("summary-epoch", prof_colors[0], false);
  for (const auto& met : m->get_metrics()) {
    EvalType train_score = met->get_mean_value(c.get_execution_mode());
    // Replace spaces with _ for consistency.
    std::string metric_name = met->name();
    std::transform(metric_name.begin(),
                   metric_name.end(),
                   metric_name.begin(),
                   [](char c_) { return c_ == ' ' ? '_' : c_; });
    std::string phase = "train_" + metric_name;
    m_summarizer->reduce_scalar(phase, train_score, c.get_step());
  }
  save_histograms(m);
  m_summarizer->flush();
  prof_region_end("summary-epoch", false);
}

void summary::on_test_end(model* m)
{

  if (!m_summarizer) {
    LBANN_ERROR("Summary callback failed: m_summarizer does not exist.");
  }
  const auto& c = m->get_execution_context();
  prof_region_begin("summary-test", prof_colors[0], false);
  lbann_comm* comm = m->get_comm();
  for (auto&& met : m->get_metrics()) {
    EvalType test_score = met->get_mean_value(c.get_execution_mode());
    // Replace spaces with _ for consistency.
    std::string metric_name = met->name();
    std::transform(metric_name.begin(),
                   metric_name.end(),
                   metric_name.begin(),
                   [](char c_) { return c_ == ' ' ? '_' : c_; });
    std::string phase = "test_" + metric_name;
    m_summarizer->reduce_scalar(phase, test_score, c.get_step());
  }
  // Reset counters incremented during test phase.
  comm->reset_stats_counters();
  for (auto&& layer : m->get_layers()) {
    layer->reset_counters();
  }
  prof_region_end("summary-test", false);
}

void summary::save_histograms(model* m)
{
  using LayerType = data_type_layer<DataType>;
  using OptimizerType = data_type_optimizer<DataType>;
  using WeightsType = data_type_weights<DataType>;

  if (!m_summarizer) {
    LBANN_ERROR("Summary callback failed: m_summarizer does not exist.");
  }
  const auto& c = m->get_execution_context();
  for (const auto& layer : m->get_layers()) {
    const std::string prefix = layer->get_name() + "/";
    for (int i = 0; i < layer->get_num_children(); ++i) {
      auto* dtl = dynamic_cast<LayerType*>(layer);
      AbsDistMatReadProxy<El::Device::CPU> acts(dtl->get_activations(i));
      m_summarizer->reduce_histogram(prefix + "activations" + std::to_string(i),
                                     acts.GetLocked(),
                                     c.get_step());
    }
  }
  for (const auto& w : m->get_weights()) {
    const std::string prefix = w->get_name() + "/";
    auto* dtw = dynamic_cast<WeightsType*>(w);
    AbsDistMatReadProxy<El::Device::CPU> weights(dtw->get_values());
    m_summarizer->reduce_histogram(prefix + "weights",
                                   weights.GetLocked(),
                                   c.get_step());
    optimizer* opt = w->get_optimizer();
    if (opt != nullptr) {
      auto* dt_opt = dynamic_cast<OptimizerType*>(opt);
      auto grad = dt_opt->get_gradient();
      AbsDistMatReadProxy<El::Device::CPU> gradients(*grad);
      m_summarizer->reduce_histogram(prefix + "weights_gradient",
                                     gradients.GetLocked(),
                                     c.get_step());
    }
  }
}

void summary::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_summary();
  msg->set_batch_interval(m_batch_interval);
  msg->set_mat_interval(m_mat_interval);
}

std::unique_ptr<callback_base> build_summary_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>& summarizer)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSummary&>(proto_msg);
  return std::make_unique<summary>(summarizer,
                                   params.batch_interval(),
                                   params.mat_interval());
}

} // namespace callback
} // namespace lbann
