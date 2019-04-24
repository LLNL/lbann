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
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_timer.hpp"
#include "lbann/utils/timer.hpp"
#include <algorithm>

namespace lbann {

void lbann_callback_timer::batch_timing_begin(const model& m) {
  const auto& mode = m.get_execution_mode();
  m_batch_start_times[mode] = get_time();
}

void lbann_callback_timer::batch_timing_end(const model& m) {
  const auto& mode = m.get_execution_mode();
  const auto& batch_time = get_time() - m_batch_start_times[mode];
  m_batch_times[mode].push_back(batch_time);
  if (m_summarizer != nullptr) {
    m_summarizer->reduce_scalar("minibatch_time", batch_time, m.get_step(execution_mode::training)-1);
    m_summarizer->reduce_scalar_all("minibatch_time", batch_time, m.get_step(execution_mode::training)-1);
  }
}

void lbann_callback_timer::timing_begin(const model& m) {
  const auto& mode = m.get_execution_mode();
  m_start_times[mode] = get_time();
  m_batch_times[mode].clear();
}

void lbann_callback_timer::timing_end(model& m) {
  constexpr EvalType zero = 0;

  // Get run time
  const auto& mode = m.get_execution_mode();
  const auto& run_time = get_time() - m_start_times[mode];

  // Compute minibatch statistics
  const auto& batch_times = m_batch_times[mode];
  const auto& num_batches = batch_times.size();
  EvalType batch_time_mean = std::nan("");
  EvalType batch_time_min = std::nan("");
  EvalType batch_time_max = std::nan("");
  EvalType batch_time_stdev = std::nan("");
  if (num_batches > 0) {
    batch_time_mean = std::accumulate(batch_times.begin(),
                                      batch_times.end(),
                                      zero) / num_batches;
    batch_time_min = *std::min_element(batch_times.begin(),
                                       batch_times.end());
    batch_time_max = *std::max_element(batch_times.begin(),
                                       batch_times.end());
  }
  if (num_batches > 1) {
    batch_time_stdev = zero;
    for (const auto& t : batch_times) {
      const auto& diff = t - batch_time_mean;
      batch_time_stdev += diff * diff;
    }
    batch_time_stdev /= num_batches - 1;
    batch_time_stdev = std::sqrt(std::max(batch_time_stdev, zero));
  }

  // Get string for execution mode
  std::string mode_string;
  switch(mode) {
  case execution_mode::training:
    mode_string = "training epoch " + std::to_string(m.get_epoch()-1);
    break;
  case execution_mode::validation:
    mode_string = "validation";
    break;
  case execution_mode::testing:
    mode_string = "test";
    break;
  default:
    LBANN_ERROR("invalid execution mode");
  }

  // Report timing results
  auto& comm = *m.get_comm();
  const El::Int num_models = comm.get_num_trainers();
  if (comm.am_trainer_master()) {

    // Gather timing results in world master
    std::vector<EvalType> run_time_list(num_models);
    std::vector<EvalType> mean_list(num_models);
    std::vector<EvalType> min_list(num_models);
    std::vector<EvalType> max_list(num_models);
    std::vector<EvalType> stdev_list(num_models);
    if (comm.am_world_master()) {
      comm.intertrainer_gather(run_time, run_time_list);
      comm.intertrainer_gather(batch_time_mean, mean_list);
      comm.intertrainer_gather(batch_time_min, min_list);
      comm.intertrainer_gather(batch_time_max, max_list);
      comm.intertrainer_gather(batch_time_stdev, stdev_list);
    } else {
      const auto& world_master = comm.get_intertrainer_master();
      comm.intertrainer_gather(run_time, world_master);
      comm.intertrainer_gather(batch_time_mean, world_master);
      comm.intertrainer_gather(batch_time_min, world_master);
      comm.intertrainer_gather(batch_time_max, world_master);
      comm.intertrainer_gather(batch_time_stdev, world_master);
    }

    // Print results
    if (comm.am_world_master()) {
      for (El::Int i = 0; i < num_models; ++i) {
        std::cout << m.get_name() << " (instance "<< i << ") " << mode_string << " "
                  << "run time : " << run_time_list[i] << "s"
                  << std::endl;
      }
      for (El::Int i = 0; i < num_models; ++i) {
        std::cout << m.get_name() << " (instance " << i << ") " << mode_string << " "
                  << "mini-batch time statistics : ";
        if (std::isnan(mean_list[i])) {
          std::cout << "N/A";
        } else {
          std::cout << mean_list[i] << "s";
        }
        std::cout << " mean, ";
        if (std::isnan(max_list[i])) {
          std::cout << "N/A";
        } else {
          std::cout << max_list[i] << "s";
        }
        std::cout << " max, ";
        if (std::isnan(min_list[i])) {
          std::cout << "N/A";
        } else {
          std::cout << min_list[i] << "s";
        }
        std::cout << " min, ";
        if (std::isnan(stdev_list[i])) {
          std::cout << "N/A";
        } else {
          std::cout << stdev_list[i] << "s";
        }
        std::cout << " stdev" << std::endl;
      }

    }
  }

}

}  // namespace lbann
