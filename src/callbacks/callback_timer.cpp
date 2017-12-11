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
// lbann_callback_timer .hpp .cpp - Callback hooks to time training
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include "lbann/utils/timer.hpp"
#include "lbann/callbacks/callback_timer.hpp"

namespace lbann {

void lbann_callback_timer::batch_timing_begin(model *m) {
  m_batch_start_time = get_time();
}

void lbann_callback_timer::batch_timing_end(model *m) {
  const double mb_time = get_time() - m_batch_start_time;
  m_batch_times.push_back(mb_time);
  if (m_summarizer != nullptr) {
    m_summarizer->reduce_scalar("minibatch_time", mb_time, m->get_cur_step());
  }
}

void lbann_callback_timer::timing_begin(model *m) {
  m_batch_times.clear();
  m_start_time = get_time();
}

void lbann_callback_timer::timing_end(model *m) {
  lbann_comm *comm = m->get_comm();

  // Get run time
  const double run_time = get_time() - m_start_time;

  // Compute minibatch statistics
  const int num_batches = m_batch_times.size();
  const double mean = (std::accumulate(m_batch_times.begin(),
                                       m_batch_times.end(),
                                       0.0)
                       / num_batches);
  const auto& minmax = std::minmax_element(m_batch_times.begin(), m_batch_times.end());
  const double sqmean = (std::inner_product(m_batch_times.begin(),
                                            m_batch_times.end(),
                                            m_batch_times.begin(),
                                            0.0)
                         / num_batches);
  const double var = std::max(sqmean - mean * mean, 0.0);
  const double stdev = std::sqrt(var * num_batches / (num_batches - 1));

  // Get string for execution mode
  std::string mode_string;
  switch(m->get_execution_mode()) {
  case execution_mode::training:
    mode_string = "training epoch " + std::to_string(m->get_cur_epoch());
    break;
  case execution_mode::validation:
    mode_string = "validation";
    break;
  case execution_mode::testing:
    mode_string = "test";
    break;
  default:
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid execution mode for reporting results";
    throw lbann_exception(err.str());
  }

  // Report timing results
  const int num_models = comm->get_num_models();
  if (comm->am_model_master()) {

    // Gather timing results in world master
    std::vector<double> run_time_list(num_models);
    std::vector<double> mean_list(num_models);
    std::vector<double> min_list(num_models);
    std::vector<double> max_list(num_models);
    std::vector<double> stdev_list(num_models);
    if (comm->am_world_master()) {
      comm->intermodel_gather(run_time, run_time_list);
      comm->intermodel_gather(mean, mean_list);
      comm->intermodel_gather(*(minmax.first), min_list);
      comm->intermodel_gather(*(minmax.second), max_list);
      comm->intermodel_gather(stdev, stdev_list);
    } else {
      comm->intermodel_gather(run_time, comm->get_intermodel_master());
      comm->intermodel_gather(mean, comm->get_intermodel_master());
      comm->intermodel_gather(*(minmax.first), comm->get_intermodel_master());
      comm->intermodel_gather(*(minmax.second), comm->get_intermodel_master());
      comm->intermodel_gather(stdev, comm->get_intermodel_master());
    }

    // Print results
    if (comm->am_world_master()) {
      for (int i = 0; i < num_models; ++i) {
        std::cout << "Model " << i << " " << mode_string << " "
                  << "run time : " << run_time_list[i] << "s"
                  << std::endl;
      }
      for (int i = 0; i < num_models; ++i) {
        std::cout << "Model " << i << " " << mode_string << " "
                  << "mini-batch time statistics : "
                  << mean_list[i] << "s mean, "
                  << min_list[i] << "s min, "
                  << max_list[i] << "s max, "
                  << stdev_list[i] << "s stdev" << std::endl;
      }

    }
  }
  
}

}  // namespace lbann
