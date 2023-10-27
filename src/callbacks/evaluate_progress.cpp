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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/evaluate_progress.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/trainers/trainer.hpp"

#include "lbann/proto/callbacks.pb.h"

namespace lbann {
namespace callback {

namespace {

/** Get mean metric value with validation set. */
EvalType evaluate(model& m, const std::string& metric_name, execution_mode mode)
{
  auto& c = m.get_execution_context();
  // Make sure data readers finish asynchronous work
  const auto original_mode = c.get_execution_mode();
  data_coordinator& dc = get_trainer().get_data_coordinator();
  dc.collect_background_data_fetch(original_mode);

  // Mark the data store as loading - Note that this is a temporary fix
  // for the current use of the tournament
  dc.mark_data_store_explicitly_loading(mode);

  // Evaluate model on validation set
  get_trainer().evaluate(&m, mode);

  // Get metric value
  bool found_metric = false;
  EvalType metric_value = 0;
  for (const auto& met : m.get_metrics()) {
    if (met->name() == metric_name) {
      found_metric = true;
      metric_value = met->get_mean_value(mode);
      break;
    }
  }
  if (!found_metric) {
    LBANN_ERROR("could not find metric \"",
                metric_name,
                "\" ",
                "in model \"",
                m.get_name(),
                "\"");
  }

  // Mark the data store as loaded - Note that this is a temporary fix
  // for the current use of the evaluation
  dc.make_data_store_preloaded(mode);

  // Clean up and return metric value
  m.reset_mode(c, original_mode);
  dc.reset_mode(c);
  return metric_value;
}

} // namespace

evaluate_progress::evaluate_progress(El::Int batch_interval,
                                     std::string metric_name)
  : callback_base(batch_interval), m_metric_name{std::move(metric_name)}
{}

evaluate_progress::evaluate_progress(const evaluate_progress& other)
  : callback_base(other), m_metric_name(other.m_metric_name)
{}

evaluate_progress& evaluate_progress::operator=(const evaluate_progress& other)
{
  callback_base::operator=(other);
  m_metric_name = other.m_metric_name;
  return *this;
}

void evaluate_progress::on_batch_begin(model* m)
{
  auto& local_model = *m;
  auto& context = local_model.get_execution_context();
  auto&& comm = *local_model.get_comm();

  // Check whether to start EVALUATE_PROGRESS round
  const auto mode = context.get_execution_mode();
  const auto step = context.get_step();
  if (mode != execution_mode::training || step == 0) {
    return;
  }

  data_coordinator& dc = get_trainer().get_data_coordinator();
  auto evaluation_mode = execution_mode::invalid;
  auto modes = {execution_mode::tournament,
                execution_mode::validation,
                execution_mode::testing};
  for (auto em : modes) {
    if (dc.is_execution_mode_valid(em)) {
      evaluation_mode = em;
      break;
    }
  }
  if (evaluation_mode == execution_mode::invalid) {
    LBANN_WARNING("evaluate_progress requires ",
                  to_string(execution_mode::tournament),
                  " or ",
                  to_string(execution_mode::validation),
                  " or ",
                  to_string(execution_mode::testing),
                  " execution modes");
    return;
  }

  // Print message
  const auto message_prefix =
    (std::string{} + "evaluate progress using (" + to_string(evaluation_mode) +
     " data set) on " + "model \"" + local_model.get_name() + "\", " + "step " +
     std::to_string(step) + "): ");

  if (comm.am_trainer_master()) {
    std::ostringstream msg;
    msg << "Starting to " << message_prefix << " while "
        << "(" << to_string(mode) << ") "
        << "\n";
    std::cout << msg.str() << std::flush;
  }
  auto local_score = evaluate(local_model, m_metric_name, evaluation_mode);

  // Report evaluation results
  if (comm.am_trainer_master()) {
    std::ostringstream msg;
    msg << "Finished " << message_prefix << "during "
        << "(" << to_string(mode) << ") "
        << "= " << local_score << "\n";
    std::cout << msg.str() << std::flush;
  }
}

void evaluate_progress::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_evaluate_progress();
  msg->set_batch_interval(m_batch_interval);
  msg->set_metric(m_metric_name);
}
std::unique_ptr<callback_base> build_evaluate_progress_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackEvaluateProgress&>(
      proto_msg);
  return std::make_unique<evaluate_progress>(params.batch_interval(),
                                             params.metric());
}

} // namespace callback
} // namespace lbann
