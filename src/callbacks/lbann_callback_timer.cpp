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

void lbann_callback_timer::on_epoch_begin(model *m) {
  m_epoch_start = get_time();
}

void lbann_callback_timer::on_epoch_end(model *m) {
  double end = get_time();
  double epoch_time = end - m_epoch_start;
  // Compute minibatch stats.
  double mean = std::accumulate(m_batch_times.begin(), m_batch_times.end(), 0.0) /
                m_batch_times.size();
  auto minmax = std::minmax_element(m_batch_times.begin(), m_batch_times.end());
  double stdev = 0.0;
  for (const auto& t : m_batch_times) {
    stdev += (t - mean) * (t - mean);
  }
  stdev = sqrt(stdev / (m_batch_times.size() - 1));
  m_batch_times.clear();

  // Output.
  lbann_comm *comm = m->get_comm();
  if (comm->am_model_master()) {
    if (comm->am_world_master()) {
      std::vector<double> epoch_times(comm->get_num_models());
      std::vector<double> means(comm->get_num_models());
      std::vector<double> mins(comm->get_num_models());
      std::vector<double> maxes(comm->get_num_models());
      std::vector<double> stdevs(comm->get_num_models());
      comm->intermodel_gather(epoch_time, epoch_times);
      comm->intermodel_gather(mean, means);
      comm->intermodel_gather(*(minmax.first), mins);
      comm->intermodel_gather(*(minmax.second), maxes);
      comm->intermodel_gather(stdev, stdevs);
      for (int i = 0; i < comm->get_num_models(); ++i) {
        std::cout << "Model " << i << " Epoch time: " << epoch_times[i] << "s; ";
        std::cout << "Mean minibatch time: " << means[i] << "s; ";
        std::cout << "Min: " << mins[i] << "s; ";
        std::cout << "Max: " << maxes[i] << "s; ";
        std::cout << "Stdev: " << stdevs[i] << "s" << std::endl;
      }
    } else {
      comm->intermodel_gather(epoch_time, comm->get_intermodel_master());
      comm->intermodel_gather(mean, comm->get_intermodel_master());
      comm->intermodel_gather(*(minmax.first), comm->get_intermodel_master());
      comm->intermodel_gather(*(minmax.second), comm->get_intermodel_master());
      comm->intermodel_gather(stdev, comm->get_intermodel_master());
    }
  }
}

void lbann_callback_timer::on_batch_begin(model *m) {
  m_batch_start = get_time();
}

void lbann_callback_timer::on_batch_end(model *m) {
  double end = get_time();
  double mb_time = end - m_batch_start;
  m_batch_times.push_back(mb_time);
  if (m_summarizer != nullptr) {
    m_summarizer->reduce_scalar("minibatch_time", mb_time, m->get_cur_step());
  }
}

}  // namespace lbann
