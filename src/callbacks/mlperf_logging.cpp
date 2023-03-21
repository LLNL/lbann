////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
// mlperf_logging .hpp .cpp - Prints mlperf compliant benchmark logs
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/mlperf_logging.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/weights/weights.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/optimizers/optimizer.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <any>
#include <sstream>

namespace lbann {
namespace callback {

namespace {
void print_value(std::ostringstream& os, int value)
{
  os << value;
}
void print_value(std::ostringstream& os, double value)
{
  os << value;
}
void print_value(std::ostringstream& os, long value)
{
  os << value;
}
void print_value(std::ostringstream& os, size_t value)
{
  os << value;
}
void print_value(std::ostringstream& os, std::string const& value)
{
  os << "\"" << value << "\"";
}
void print_value(std::ostringstream& os, char const* value)
{
  os << "\"" << value << "\"";
}
template <typename T>
void print_value(std::ostringstream& os, T value)
{
  //FIXME: Should I push the value anyway?
  os << "\"UNKNOWN_DATA_TYPE\"";
}

//FIXME: Tom's problem
int get_real_num_accelerators()
{
  return 0;
}

int get_num_nodes()
{
  if (std::getenv("SLURM_NNODES"))
    return atoi(std::getenv("SLURM_NNODES"));
  else if (std::getenv("FLUX_JOB_NNODES"))
    return atoi(std::getenv("FLUX_JOB_NNODES"));
  else return -1;
  //FIXME: count number of unique hostnames in universe?
}
}// namespace

void mlperf_logging::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_mlperf_logging();
  msg->set_sub_benchmark(m_sub_benchmark);
  msg->set_sub_org(m_sub_org);
  msg->set_sub_division(m_sub_division);
  msg->set_sub_status(m_sub_status);
  msg->set_sub_platform(m_sub_platform);
}

template <typename T>
void mlperf_logging::print(std::ostringstream& os,
                           mlperf_logging::event_type et, std::string key,
                           T value, char const* file, size_t line,
                           double epoch) const
{
  os << "{"
     << "\"namespace\": \"\", "
     << "\"time_ms\": " << get_ms_since_epoch() << ", "
     << "\"event_type\": \"";
  print_event_type(os, et);

  os << "\", "
     << "\"key\": \"" << key << "\", "
     << "\"value\": ";
  print_value(os, value);
  os << ", "
     << "\"metadata\": {\"file\": \"" << file << "\", "
     << "\"lineno\": " << line;
  if(epoch < 0)
    os << "}}";
  else
    os << ", " << "\"epoch_num\": " << epoch << "}}";

  m_logger.get().info(os.str());
  os.flush();
}

void mlperf_logging::print_event_type(std::ostringstream& os, mlperf_logging::event_type et) const
{
  switch (et) {
  case mlperf_logging::event_type::TIME_POINT: os << "POINT_IN_TIME"; break;
  case mlperf_logging::event_type::INT_START: os << "INTERVAL_START"; break;
  case mlperf_logging::event_type::INT_END: os << "INTERVAL_END"; break;
  default: os << "INVALID_EVENT_TYPE"; break;
  }
}

size_t mlperf_logging::get_ms_since_epoch()
{
  using namespace std::chrono;
  return duration_cast< milliseconds >(
    system_clock::now().time_since_epoch()).count();
}

void mlperf_logging::setup(model *m)
{
  std::ostringstream os;

  // Not a good/portable way to do this in C++
  // std::string value = "null";
  // print(os, mlperf_logging::event_type::TIME_POINT, "cache_clear", value,
  //      __FILE__, __LINE__);

  print(os, mlperf_logging::event_type::TIME_POINT, "submission_benchmark",
        m_sub_benchmark, __FILE__, __LINE__);

  print(os, mlperf_logging::event_type::TIME_POINT, "submission_org",
        m_sub_org, __FILE__, __LINE__);

  print(os, mlperf_logging::event_type::TIME_POINT, "submission_division",
        m_sub_division, __FILE__, __LINE__);

  print(os, mlperf_logging::event_type::TIME_POINT, "submission_status",
        m_sub_status, __FILE__, __LINE__);

  print(os, mlperf_logging::event_type::TIME_POINT, "submission_platform",
        m_sub_platform, __FILE__, __LINE__);

  print(os, mlperf_logging::event_type::INT_START, "init_start", "null",
        __FILE__, __LINE__);
}
void mlperf_logging::on_setup_end(model *m)
{
  std::ostringstream os;
  lbann_comm *comm = m->get_comm();
  auto const& trainer = get_const_trainer();

  print(os, mlperf_logging::event_type::TIME_POINT, "number_of_ranks",
        static_cast<int>(comm->get_procs_in_world()), __FILE__, __LINE__);

  print(os, mlperf_logging::event_type::TIME_POINT, "number_of_nodes",
        static_cast<int>(get_num_nodes()), __FILE__, __LINE__);

  auto accelerators = get_real_num_accelerators();
  print(os, mlperf_logging::event_type::TIME_POINT, "accelerators_per_node",
        static_cast<int>(accelerators), __FILE__, __LINE__);

  auto const seed = trainer.get_random_seed();
  print(os, mlperf_logging::event_type::TIME_POINT, "seed",
        seed, __FILE__, __LINE__);

  auto const& dc = trainer.get_data_coordinator();
  auto const batch_size = dc.get_global_mini_batch_size(
    execution_mode::training);
  print(os, mlperf_logging::event_type::TIME_POINT, "global_batch_size",
        batch_size, __FILE__, __LINE__);

  auto samples = dc.get_total_num_samples(execution_mode::training);
  print(os, mlperf_logging::event_type::TIME_POINT, "train_samples",
        samples, __FILE__, __LINE__);

  //FIXME: Should this be execution_mode::validation? Tom thinks no
  auto eval_samples = dc.get_total_num_samples(execution_mode::testing);
  print(os, mlperf_logging::event_type::TIME_POINT, "eval_samples",
        eval_samples, __FILE__, __LINE__);

  auto const weights = m->get_weights();
  for (auto const w : weights)
    if( w->get_optimizer() != nullptr ){
      std::string opt = w->get_optimizer()->get_type();
      print(os, mlperf_logging::event_type::TIME_POINT, "opt_name",
            opt, __FILE__, __LINE__);

      auto opt_learning_rate = w->get_optimizer()->get_learning_rate();
      print(os, mlperf_logging::event_type::TIME_POINT,
            "opt_base_learning_rate", static_cast<double>(opt_learning_rate),
            __FILE__, __LINE__);
      break;
    }

  // LBANN does not perform warmup steps.
  //  auto opt_warmup_steps = -1;
  //  print(os, mlperf_logging::event_type::TIME_POINT,
  //      "opt_learning_rate_warmup_steps",
  //      static_cast<size_t>(opt_warmup_steps),
  //      __FILE__, __LINE__);

  // auto opt_warmup_factor = -1;
  // print(os, mlperf_logging::event_type::TIME_POINT,
  //      "opt_learning_rate_warmup_factor",
  //      static_cast<double>(opt_warmup_factor),
  //      __FILE__, __LINE__);

  // FIXME (Tom's problem)
  //auto opt_decay_bound_steps = -1;
  //print(os, mlperf_logging::event_type::TIME_POINT,
  //      "opt_learning_rate_decay_boundary_steps",
  //      static_cast<size_t>(opt_decay_bound_steps),
  //      __FILE__, __LINE__);

  // auto opt_decay_factor = -1;
  // print(os, mlperf_logging::event_type::TIME_POINT,
  //      "opt_learning_rate_decay_factor",
  //      static_cast<double>(opt_decay_factor),
  //      __FILE__, __LINE__);

  print(os, mlperf_logging::event_type::INT_END, "init_stop", "null",
        __FILE__, __LINE__);
}

void mlperf_logging::on_epoch_begin(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_START, "epoch_start", "null",
        __FILE__, __LINE__, epoch);
}

void mlperf_logging::on_epoch_end(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_END, "epoch_stop", "null",
        __FILE__, __LINE__, epoch);
}

void mlperf_logging::on_train_begin(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_START, "run_start", "null",
        __FILE__, __LINE__, epoch);
}

void mlperf_logging::on_train_end(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_END, "run_stop", "null",
        __FILE__, __LINE__, epoch);
}

void mlperf_logging::on_batch_evaluate_begin(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_START, "eval_start", "null",
        __FILE__, __LINE__, epoch);
}

void mlperf_logging::on_batch_evaluate_end(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_END, "eval_stop", "null",
        __FILE__, __LINE__, epoch);

  //FIXME (Tom's problem)
  auto eval_error = -1;
  print(os, mlperf_logging::event_type::TIME_POINT, "eval_error",
        static_cast<double>(eval_error), __FILE__,
        __LINE__, epoch);
}

std::unique_ptr<callback_base>
build_mlperf_logging_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackMlperfLogging&>(proto_msg);
  return std::make_unique<mlperf_logging>(params.sub_benchmark(),
                                          params.sub_org(),
                                          params.sub_division(),
                                          params.sub_status(),
                                          params.sub_platform());
  //params.num_nodes());
}
} // namespace callback
} // namespace lbann
