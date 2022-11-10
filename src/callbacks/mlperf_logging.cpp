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

#include <callbacks.pb.h>

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <any>
#include <sstream>

namespace lbann {
namespace callback {

template <typename T>
void mlperf_logging::print(std::ostream& os, mlperf_logging::event_type et,
                           std::string key, T value, char const* file,
                           size_t line, double epoch) const
{
  os << "{"
     << "\"namespace\": \"\", "
     << "\"time_ms\": " << get_ms_since_epoch() << ", "
     << "\"event_type\": \"";
  print_event_type(os, et);

  os << "\", "
     << "\"key\": " << key << "\", "
     << "\"value\": ";
  print_value(os, value);
  os << ", "
     << "\"metadata\": {\"file\": \"" << file << "\", "
     << "\"lineno\": " << line;
  if(epoch < 0)
    os << "}}\n";
  else
    os << ", " << "\"epoch_num\": " << epoch << "}}\n";
}

void mlperf_logging::print_event_type(std::ostream& os, mlperf_logging::event_type et) const
{
  switch (et) {
  case mlperf_logging::event_type::TIME_POINT: os << "POINT_IN_TIME"; break;
  case mlperf_logging::event_type::INT_START: os << "INTERVAL_START"; break;
  case mlperf_logging::event_type::INT_END: os << "INTERVAL_END"; break;
  default: os << "INVALID_EVENT_TYPE"; break;
  }
}

void mlperf_logging::print_value(std::ostream& os, double value) const
{
  os << value;
}
void mlperf_logging::print_value(std::ostream& os, long value) const
{
  os << value;
}
void mlperf_logging::print_value(std::ostream& os, size_t value) const
{
  os << value;
}
void mlperf_logging::print_value(std::ostream& os, std::string value) const
{
  os << value;
}
/*template <typename T>
void mlperf_logging::print_value(std::ostream& os, T value) const
{
  //FIXME: Should I push the value anyway?
  os << "UNKNOWN_DATA_TYPE";
}
*/

size_t mlperf_logging::get_ms_since_epoch()
{
  using namespace std::chrono;
  return duration_cast< milliseconds >(
    system_clock::now().time_since_epoch()).count();
}

//FIXME(KLG): There is no on_setup_begin. Can I steal this as a callback hook?
void mlperf_logging::setup(model *m)
{
  std::ostringstream os;

  //FIXME: What is this?
  std::string value = "null";
  print(os, mlperf_logging::event_type::TIME_POINT, "cache_clear", value,
        __FILE__, __LINE__);

  //FIXME: Make these user input vars
  value = "oc20";
  print(os, mlperf_logging::event_type::TIME_POINT, "submission_benchmark",
        value, __FILE__, __LINE__);

  value = "LBANN";
  print(os, mlperf_logging::event_type::TIME_POINT, "submission_org",
        value, __FILE__, __LINE__);

  //FIXME: value = closed?
  value = "closed";
  print(os, mlperf_logging::event_type::TIME_POINT, "submission_division",
        value, __FILE__, __LINE__);

  //FIXME: value = onprem?
  value = "onprem";
  print(os, mlperf_logging::event_type::TIME_POINT, "submission_status",
        value, __FILE__, __LINE__);

  //FIXME:  value = SUBMISSION_PLATFORM_PLACEHOLDER?
  value = "?";
  print(os, mlperf_logging::event_type::TIME_POINT, "submission_platform",
        value, __FILE__, __LINE__);

  value = "null";
  print(os, mlperf_logging::event_type::TIME_POINT, "init_start", value,
        __FILE__, __LINE__);

  H2_INFO(os.str());
}
void mlperf_logging::on_setup_end(model *m)
{
  std::ostringstream os;
  lbann_comm *comm = m->get_comm();

  //FIXME: num_trainers or world size?
  print(os, mlperf_logging::event_type::TIME_POINT, "number_of_ranks",
        static_cast<double>(comm->get_num_trainers()), __FILE__, __LINE__);

  //FIXME
  auto nodes = -1;
  print(os, mlperf_logging::event_type::TIME_POINT, "number_of_nodes",
        static_cast<double>(nodes), __FILE__, __LINE__);

  //FIXME
  auto accelerators = -1;
  print(os, mlperf_logging::event_type::TIME_POINT, "accelerators_per_node",
        static_cast<double>(accelerators), __FILE__, __LINE__);

  //FIXME: From trainer.hpp?
  auto seed = -1;
  print(os, mlperf_logging::event_type::TIME_POINT, "seed",
        static_cast<double>(seed), __FILE__, __LINE__);

  //FIXME: Add get_minibatch_size to model or metrics?
  auto batch_size = -1;
  print(os, mlperf_logging::event_type::TIME_POINT, "global_batch_size",
        static_cast<double>(batch_size), __FILE__, __LINE__);

  metric_statistics metrics;
  auto samples = metrics.get_num_samples();
  print(os, mlperf_logging::event_type::TIME_POINT, "train_samples",
        static_cast<double>(samples), __FILE__, __LINE__);

  //FIXME
  auto eval_samples = -1;
  print(os, mlperf_logging::event_type::TIME_POINT, "eval_samples",
        static_cast<double>(eval_samples), __FILE__, __LINE__);

  //FIXME: I couldn't get this to work
  //auto* optimizer = m->get_weights().get_optimizer();
  std::string opt = "opt_name";
  print(os, mlperf_logging::event_type::TIME_POINT, "opt_name",
        opt, __FILE__, __LINE__);

  //FIXME
  auto opt_learning_rate = -1;
  print(os, mlperf_logging::event_type::TIME_POINT, "opt_base_learning_rate",
        static_cast<double>(opt_learning_rate), __FILE__, __LINE__);

  //FIXME
  auto opt_warmup_steps = -1;
  print(os, mlperf_logging::event_type::TIME_POINT,
        "opt_learning_rate_warmup_steps",
        static_cast<double>(opt_warmup_steps),
        __FILE__, __LINE__);

  //FIXME
  auto opt_warmup_factor = -1;
  print(os, mlperf_logging::event_type::TIME_POINT,
        "opt_learning_rate_warmup_factor",
        static_cast<double>(opt_warmup_factor),
        __FILE__, __LINE__);

  //FIXME
  auto opt_decay_bound_steps = -1;
  print(os, mlperf_logging::event_type::TIME_POINT,
        "opt_learning_rate_decay_boundary_steps",
        static_cast<double>(opt_decay_bound_steps),
        __FILE__, __LINE__);

  //FIXME
  auto opt_decay_factor = -1;
  print(os, mlperf_logging::event_type::TIME_POINT,
        "opt_learning_rate_decay_factor",
        static_cast<double>(opt_decay_factor),
        __FILE__, __LINE__);

  print(os, mlperf_logging::event_type::TIME_POINT, "init_stop", "null",
        __FILE__, __LINE__);

  H2_INFO(os.str());
}

void mlperf_logging::on_epoch_begin(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_START, "epoch_start", "null",
        __FILE__, __LINE__, epoch);

  H2_INFO(os.str());
}

void mlperf_logging::on_epoch_end(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_START, "epoch_stop", "null",
        __FILE__, __LINE__, epoch);

  H2_INFO(os.str());
}

void mlperf_logging::on_train_begin(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  //FIXME: run_start? Same time stamp as epoch 1 in results
  print(os, mlperf_logging::event_type::INT_START, "run_start", "null",
        __FILE__, __LINE__, epoch);

  H2_INFO(os.str());
}

void mlperf_logging::on_train_end(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  //FIXME: run_stop? End of training?
  print(os, mlperf_logging::event_type::INT_START, "run_stop", "null",
        __FILE__, __LINE__, epoch);

  H2_INFO(os.str());
}

void mlperf_logging::on_batch_evaluate_begin(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_START, "eval_start", "null",
        __FILE__, __LINE__, epoch);

  H2_INFO(os.str());
}

void mlperf_logging::on_batch_evaluate_end(model *m)
{
  std::ostringstream os;
  const auto& epoch = static_cast<const SGDExecutionContext&>(
    m->get_execution_context()).get_epoch();

  print(os, mlperf_logging::event_type::INT_START, "eval_stop", "null",
        __FILE__, __LINE__, epoch);

  //FIXME
  auto eval_error = -1;
  print(os, mlperf_logging::event_type::TIME_POINT, "eval_error",
        static_cast<double>(eval_error), __FILE__,
        __LINE__, epoch);

  H2_INFO(os.str());
}

std::unique_ptr<callback_base>
build_mlperf_logging_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackMlperfLogging&>(proto_msg);
  return std::make_unique<mlperf_logging>(params.output_filename());
}
} // namespace callback
} // namespace lbann
