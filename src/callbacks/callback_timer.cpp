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
  const EvalType mb_time = get_time() - m_batch_start_time;
  m_batch_times.push_back(mb_time);
  if (m_summarizer != nullptr) {
    m_summarizer->reduce_scalar("minibatch_time", mb_time, m->get_cur_step()-1);
  }
}

void lbann_callback_timer::timing_begin(model *m) {
  m_batch_times.clear();
  m_start_time = get_time();
}

void lbann_callback_timer::timing_end(model *m) {
  lbann_comm *comm = m->get_comm();

  // Get run time
  const EvalType run_time = get_time() - m_start_time;

  // Compute minibatch statistics
  const int num_batches = m_batch_times.size();
  EvalType batch_time_mean = std::nan("");
  EvalType batch_time_min = std::nan("");
  EvalType batch_time_max = std::nan("");
  EvalType batch_time_stdev = std::nan("");
  if (num_batches > 0) {
    batch_time_mean = std::accumulate(m_batch_times.begin(),
                                      m_batch_times.end(),
                                      0.0);
    batch_time_mean /= num_batches;
    batch_time_min = *std::min_element(m_batch_times.begin(),
                                       m_batch_times.end());
    batch_time_max = *std::max_element(m_batch_times.begin(),
                                       m_batch_times.end());
  }
  if (num_batches > 1) {
    const EvalType sqsum = std::inner_product(m_batch_times.begin(),
                                            m_batch_times.end(),
                                            m_batch_times.begin(),
                                            0.0);
    EvalType var = sqsum / num_batches - batch_time_mean * batch_time_mean;
    var = num_batches * var / (num_batches - 1);
    batch_time_stdev = std::sqrt(std::max(var, 0.0));
  }

  // Get string for execution mode
  std::string mode_string;
  switch(m->get_execution_mode()) {
  case execution_mode::training:
    mode_string = "training epoch " + std::to_string(m->get_cur_epoch()-1);
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
    std::vector<EvalType> run_time_list(num_models);
    std::vector<EvalType> mean_list(num_models);
    std::vector<EvalType> min_list(num_models);
    std::vector<EvalType> max_list(num_models);
    std::vector<EvalType> stdev_list(num_models);
    if (comm->am_world_master()) {
      comm->intermodel_gather(run_time, run_time_list);
      comm->intermodel_gather(batch_time_mean, mean_list);
      comm->intermodel_gather(batch_time_min, min_list);
      comm->intermodel_gather(batch_time_max, max_list);
      comm->intermodel_gather(batch_time_stdev, stdev_list);
    } else {
      comm->intermodel_gather(run_time, comm->get_intermodel_master());
      comm->intermodel_gather(batch_time_mean, comm->get_intermodel_master());
      comm->intermodel_gather(batch_time_min, comm->get_intermodel_master());
      comm->intermodel_gather(batch_time_max, comm->get_intermodel_master());
      comm->intermodel_gather(batch_time_stdev, comm->get_intermodel_master());
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
