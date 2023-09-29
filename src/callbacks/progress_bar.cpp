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
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/models/model.hpp"
#include "lbann/objective_functions/layer_term.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/proto/callbacks.pb.h"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

static inline void
print_progress(int iteration, int total, double avg_time, std::string prefix)
{
  // Preamble
  std::cout << "Step " << (iteration + 1) << "/" << total << "    " << prefix
            << "[";

  // Clamp value
  iteration = (iteration < 0) ? 0 : iteration;
  iteration = (iteration > (total - 1)) ? (total - 1) : iteration;
  float percentage = static_cast<float>(iteration + 1) / total;
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
      std::cout << std::fixed << std::setprecision(2) << (1.0 / avg_time)
                << " iters/sec";
    else
      std::cout << std::fixed << std::setprecision(2) << avg_time
                << " sec/iter";

    int iterations_left = (total - iteration);
    std::cout << ". ETA " << static_cast<int>(iterations_left * avg_time)
              << " sec          "; // Extra whitespace to clear previous prompts
  }

  if (iteration >= (total - 1))
    std::cout << std::endl;
  else
    std::cout << "\r" << std::flush;
}

static inline std::string get_objective_function(lbann::model* m)
{
  std::stringstream stream;
  stream << "Objective: ";

  auto terms = m->get_objective_function()->get_terms();
  bool first = true;
  for (const auto& term : terms) {
    // Only consider layer terms
    auto lterm = dynamic_cast<lbann::layer_term*>(term);
    if (!lterm)
      continue;
    lbann::Layer* layer = &lterm->get_layer();

    // Try as an EvalType evaluation layer
    lbann::EvalType objective = lbann::EvalType(-999.0);
    auto eval_layer =
      dynamic_cast<lbann::abstract_evaluation_layer<lbann::EvalType>*>(layer);
    // If not working, try as a DataType layer
    if (!eval_layer) {
      auto eval_layer_data =
        dynamic_cast<lbann::abstract_evaluation_layer<lbann::DataType>*>(layer);
      if (!eval_layer_data)
        continue;
      objective = static_cast<lbann::EvalType>(eval_layer_data->get_value());
    }
    else {
      objective = eval_layer->get_value();
    }

    if (!first)
      stream << ", ";
    stream << std::fixed << std::setprecision(4) << objective;
    first = false;
  }

  stream << " ";
  return stream.str();
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
  msg->set_newline_interval(this->m_newline_interval);
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
    std::string prefix;

    // Gather first batch of statistics
    if (m_current_iteration == 0) {
      m_last_time = ::lbann::get_time();
      m_moving_avg_time.fill(0.0);
      prefix = "";
    }
    else {
      double cur_time = ::lbann::get_time();
      double interval = cur_time - m_last_time;
      m_last_time = cur_time;
      m_moving_avg_time[m_current_iteration %
                        LBANN_PBAR_MOVING_AVERAGE_LENGTH] = interval;
      int to_avg =
        std::min(m_current_iteration, LBANN_PBAR_MOVING_AVERAGE_LENGTH);
      for (int i = 0; i < to_avg; ++i)
        avg_time += m_moving_avg_time[i];
      avg_time /= to_avg;
      prefix = get_objective_function(m);
    }
    print_progress(m_current_iteration,
                   m_training_iterations,
                   avg_time,
                   prefix);

    m_current_iteration += 1;

    if (m_newline_interval > 0 &&
        m_current_iteration % m_newline_interval == 0) {
      std::cout << std::endl << std::flush;
    }
  }
}

std::unique_ptr<callback_base> build_progress_bar_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackProgressBar&>(proto_msg);
  return std::make_unique<progress_bar>(params.interval(),
                                        params.newline_interval());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::progress_bar
#define LBANN_CLASS_LIBNAME callback_progress_bar
#include <lbann/macros/register_class_with_cereal.hpp>
