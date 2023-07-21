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
// progress_bar .hpp .cpp - Callback that prints a progress bar during epochs.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/progress_bar.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/callbacks.pb.h"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/serialize.hpp"

#include <iomanip>
#include <iostream>
#include <omp.h> // Simple way to get the current time with omp_wtime

static inline void print_progress(int iteration, int total, double avg_time)
{
  // Preamble
  std::cout << iteration << "/" << total << "  [";

  // Clamp value
  iteration = (iteration < 0) ? 0 : iteration;
  iteration = (iteration > total) ? total : iteration;
  float percentage = static_cast<float>(iteration) / total;
  int bars = static_cast<int>(percentage * LBANN_PBAR_WIDTH);

  // Bar
  for (int i = 0; i < bars; ++i)
    std::cout << "#";
  for (int i = bars; i < LBANN_PBAR_WIDTH; ++i)
    std::cout << " ";

  // Some stats
  std::cout << "] " << static_cast<int>(percentage * 100) << "% ";
  if (iteration > 0) {
    if (avg_time < 1.0)
      std::cout << std::setprecision(2) << (1.0 / avg_time) << " iters/sec";
    else
      std::cout << std::setprecision(2) << avg_time << " sec/iter";

    int iterations_left = (total - iteration);
    std::cout << ". ETA " << static_cast<int>(iterations_left * avg_time)
              << " sec          "; // Extra whitespace to clear previous prompts
  }

  if (iteration == total)
    std::cout << std::endl;
  else
    std::cout << "\r" << std::flush;
}

namespace lbann {
namespace callback {

template <class Archive>
void progress_bar::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)));
}

void progress_bar::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_progress_bar();
  msg->set_interval(this->m_batch_interval);
}

void progress_bar::on_epoch_begin(model* m)
{
  data_coordinator& dc = get_trainer().get_data_coordinator();
  lbann_comm* comm = m->get_comm();
  this->m_print = comm->am_world_master();
  this->m_training_iterations =
    dc.get_num_iterations_per_epoch(execution_mode::training);
  this->m_current_iteration = 0;
  this->m_last_time = 0.0;
  this->m_moving_avg_time.fill(0.0);
}

void progress_bar::on_forward_prop_begin(model* m)
{
  if (m_print) {
    double avg_time = 0.0;

    // Gather first batch of statistics
    if (m_current_iteration == 0) {
      m_last_time = omp_get_wtime();
      m_moving_avg_time.fill(0.0);
    }
    else {
      double cur_time = omp_get_wtime();
      double interval = cur_time - m_last_time;
      m_last_time = cur_time;
      m_moving_avg_time[m_current_iteration %
                        LBANN_PBAR_MOVING_AVERAGE_LENGTH] = interval;

      int to_avg =
        std::min(m_current_iteration, LBANN_PBAR_MOVING_AVERAGE_LENGTH);
      for (int i = 0; i < to_avg; ++i)
        avg_time += m_moving_avg_time[i];
      avg_time /= to_avg;
    }
    print_progress(m_current_iteration, m_training_iterations, avg_time);

    m_current_iteration += 1;
  }
}

std::unique_ptr<callback_base> build_progress_bar_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackProgressBar&>(proto_msg);
  return std::make_unique<progress_bar>(params.interval());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::progress_bar
#define LBANN_CLASS_LIBNAME callback_progress_bar
#include <lbann/macros/register_class_with_cereal.hpp>
