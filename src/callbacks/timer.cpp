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
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/timer.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/summary_impl.hpp"
#include "lbann/utils/timer.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <algorithm>

namespace lbann {
namespace callback {

template <class Archive>
void timer::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_start_times),
     CEREAL_NVP(m_batch_start_times),
     CEREAL_NVP(m_batch_times));
  /// @todo Consider what to do with m_summarizer (preferably remove)
}

void timer::batch_timing_begin(const model& m)
{
  const auto& c = m.get_execution_context();
  auto const mode = c.get_execution_mode();
  if (mode == execution_mode::training && c.get_step() < m_skip_steps)
    return;
  m_batch_start_times[mode] = get_time();
}

void timer::batch_timing_end(const model& m)
{
  const auto& c = m.get_execution_context();
  const auto& mode = c.get_execution_mode();
  if (mode == execution_mode::training && c.get_step() - 1 < m_skip_steps)
    return;
  const auto& batch_time = get_time() - m_batch_start_times[mode];
  m_batch_times[mode].push_back(batch_time);
  if (m_summarizer != nullptr) {
    m_summarizer->reduce_scalar("minibatch_time", batch_time, c.get_step() - 1);
    m_summarizer->reduce_scalar_all("minibatch_time",
                                    batch_time,
                                    c.get_step() - 1);
  }
}

void timer::timing_begin(const model& m)
{
  const auto& c = m.get_execution_context();
  const auto& mode = c.get_execution_mode();
  m_start_times[mode] = get_time();
  m_batch_times[mode].clear();
}

void timer::timing_end(model& m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m.get_execution_context());
  constexpr EvalType zero = 0;

  // Get run time
  const auto& mode = c.get_execution_mode();
  const auto& run_time = get_time() - m_start_times[mode];

  // Compute minibatch statistics
  const auto& batch_times = m_batch_times[mode];
  const auto& num_batches = batch_times.size();
  EvalType batch_time_mean = std::nan("");
  EvalType batch_time_min = std::nan("");
  EvalType batch_time_max = std::nan("");
  EvalType batch_time_median = std::nan("");
  EvalType batch_time_stdev = std::nan("");
  if (num_batches > 0) {
    batch_time_mean =
      std::accumulate(batch_times.begin(), batch_times.end(), zero) /
      num_batches;
    batch_time_min = *std::min_element(batch_times.begin(), batch_times.end());
    batch_time_max = *std::max_element(batch_times.begin(), batch_times.end());
    std::vector<EvalType> sorted_times(batch_times.begin(), batch_times.end());
    std::sort(sorted_times.begin(), sorted_times.end());
    if (num_batches % 2 == 0) {
      batch_time_median = sorted_times[num_batches / 2];
    }
    else {
      batch_time_median =
        (sorted_times[(num_batches - 1) / 2] + sorted_times[num_batches / 2]) /
        2;
    }
  }
  if (num_batches > 1) {
    batch_time_stdev = zero;
    for (const auto& bt : batch_times) {
      const auto& diff = bt - batch_time_mean;
      batch_time_stdev += diff * diff;
    }
    batch_time_stdev /= num_batches - 1;
    batch_time_stdev = El::Sqrt(std::max(batch_time_stdev, zero));
  }

  // Get string for execution mode
  std::string mode_string;
  switch (mode) {
  case execution_mode::training:
    mode_string = "training epoch " + std::to_string(c.get_epoch() - 1);
    break;
  case execution_mode::validation:
    mode_string = "validation";
    break;
  case execution_mode::tournament:
    mode_string = "tournament";
    break;
  case execution_mode::testing:
    mode_string = "test";
    break;
  default:
    LBANN_ERROR("invalid execution mode");
  }

  // Report timing results
  auto& comm = *m.get_comm();
  const El::Int num_trainers = comm.get_num_trainers();
  if (comm.am_trainer_master()) {

    auto& arg_parser = global_argument_parser();
    bool allow_global_statistics =
      arg_parser.get<bool>(LBANN_OPTION_LTFB_ALLOW_GLOBAL_STATISTICS);
    std::stringstream report;

    if (allow_global_statistics) {
      // Gather timing results in world master
      std::vector<EvalType> run_time_list(num_trainers);
      std::vector<EvalType> mean_list(num_trainers);
      std::vector<EvalType> min_list(num_trainers);
      std::vector<EvalType> max_list(num_trainers);
      std::vector<EvalType> stdev_list(num_trainers);
      if (comm.am_world_master()) {
        comm.intertrainer_gather(run_time, run_time_list);
        comm.intertrainer_gather(batch_time_mean, mean_list);
        comm.intertrainer_gather(batch_time_min, min_list);
        comm.intertrainer_gather(batch_time_max, max_list);
        comm.intertrainer_gather(batch_time_stdev, stdev_list);
      }
      else {
        const auto& world_master = comm.get_intertrainer_master();
        comm.intertrainer_gather(run_time, world_master);
        comm.intertrainer_gather(batch_time_mean, world_master);
        comm.intertrainer_gather(batch_time_min, world_master);
        comm.intertrainer_gather(batch_time_max, world_master);
        comm.intertrainer_gather(batch_time_stdev, world_master);
      }

      // Print results
      if (comm.am_world_master()) {
        for (El::Int i = 0; i < num_trainers; ++i) {
          std::cout << m.get_name() << " (instance " << i << ") " << mode_string
                    << " "
                    << "run time : " << run_time_list[i] << "s" << std::endl;
        }
        for (El::Int i = 0; i < num_trainers; ++i) {
          std::cout << m.get_name() << " (instance " << i << ") " << mode_string
                    << " "
                    << "mini-batch time statistics : ";
          if (std::isnan(mean_list[i])) {
            std::cout << "N/A";
          }
          else {
            std::cout << mean_list[i] << "s";
          }
          std::cout << " mean, ";
          std::cout << batch_time_median << "s median, ";
          if (std::isnan(max_list[i])) {
            std::cout << "N/A";
          }
          else {
            std::cout << max_list[i] << "s";
          }
          std::cout << " max, ";
          if (std::isnan(min_list[i])) {
            std::cout << "N/A";
          }
          else {
            std::cout << min_list[i] << "s";
          }
          std::cout << " min, ";
          if (std::isnan(stdev_list[i])) {
            std::cout << "N/A";
          }
          else {
            std::cout << stdev_list[i] << "s";
          }
          std::cout << " stdev" << std::endl;
        }
      }
    }
    else {
      // Print results for each trainer
      report << m.get_name() << " (instance " << comm.get_trainer_rank() << ") "
             << mode_string << " "
             << "run time : " << run_time << "s" << std::endl;
      report << m.get_name() << " (instance " << comm.get_trainer_rank() << ") "
             << mode_string << " "
             << "mini-batch time statistics : ";
      if (std::isnan(batch_time_mean)) {
        report << "N/A";
      }
      else {
        report << batch_time_mean << "s";
      }
      report << " mean, ";
      report << batch_time_median << "s median, ";
      if (std::isnan(batch_time_max)) {
        report << "N/A";
      }
      else {
        report << batch_time_max << "s";
      }
      report << " max, ";
      if (std::isnan(batch_time_min)) {
        report << "N/A";
      }
      else {
        report << batch_time_min << "s";
      }
      report << " min, ";
      if (std::isnan(batch_time_stdev)) {
        report << "N/A";
      }
      else {
        report << batch_time_stdev << "s";
      }
      report << " stdev" << std::endl;

      std::cout << report.str() << std::flush;
    }
  }
}

void timer::write_specific_proto(lbann_data::Callback& proto) const
{
  proto.mutable_timer();
}

std::unique_ptr<callback_base>
build_timer_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                               std::shared_ptr<lbann_summary> const& summarizer)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackTimer&>(proto_msg);
  return std::make_unique<timer>(summarizer, params.skip_steps());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::timer
#define LBANN_CLASS_LIBNAME callback_timer
#include <lbann/macros/register_class_with_cereal.hpp>
