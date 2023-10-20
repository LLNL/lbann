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
// print_statistics .hpp .cpp - Callback hooks to print information
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/print_statistics.hpp"
#include "lbann/comm_impl.hpp"

#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace lbann {
namespace callback {

template <class Archive>
void print_statistics::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_print_global_stat_only));
}

void print_statistics::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_print();
  msg->set_interval(m_batch_interval);
  msg->set_print_global_stat_only(m_print_global_stat_only);
}

void print_statistics::setup(model* m)
{
#ifdef LBANN_VERSION
  lbann_comm* comm = m->get_comm();
  if (comm->am_world_master()) {
    std::cout << "Training with LLNL LBANN version "
              << LBANN_MAKE_STR(LBANN_VERSION) << std::endl;
  }
#endif
}

void print_statistics::on_epoch_begin(model* m)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  data_coordinator& dc = get_trainer().get_data_coordinator();
  lbann_comm* comm = m->get_comm();
  if (comm->am_world_master()) {
    // Print message
    std::cout << "-------------------------------------------------------------"
                 "-------------------------------"
              << std::endl;
    std::cout << "[" << c.get_epoch() << "] Epoch : stats formated [tr/v/te/to]"
              << " iter/epoch ="
              << " ["
              << dc.get_num_iterations_per_epoch(execution_mode::training)
              << "/"
              << dc.get_num_iterations_per_epoch(execution_mode::validation)
              << "/" << dc.get_num_iterations_per_epoch(execution_mode::testing)
              << "/"
              << dc.get_num_iterations_per_epoch(execution_mode::tournament)
              << "]" << std::endl;
    std::cout << std::setfill(' ') << std::setw(23) << "  Global MB ="
              << " [" << std::setw(4)
              << dc.get_mini_batch_size(execution_mode::training) << "/"
              << std::setw(4)
              << dc.get_mini_batch_size(execution_mode::validation) << "/"
              << std::setw(4) << dc.get_mini_batch_size(execution_mode::testing)
              << "/" << std::setw(4)
              << dc.get_mini_batch_size(execution_mode::tournament) << "]"
              << "  Global last MB ="
              << " [" << std::setw(4)
              << dc.get_last_mini_batch_size(execution_mode::training) << "/"
              << std::setw(4)
              << dc.get_last_mini_batch_size(execution_mode::validation) << "/"
              << std::setw(4)
              << dc.get_last_mini_batch_size(execution_mode::testing) << "/"
              << std::setw(4)
              << dc.get_last_mini_batch_size(execution_mode::tournament) << "]"
              << std::endl;
    std::cout << "-------------------------------------------------------------"
                 "-------------------------------"
              << std::endl;
  }
}

void print_statistics::on_epoch_end(model* m) { report_results(m); }

void print_statistics::on_validation_end(model* m) { report_results(m); }

void print_statistics::on_test_end(model* m) { report_results(m); }

void print_statistics::report_results(model* m)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  lbann_comm* comm = m->get_comm();

  // Get string for execution mode
  const execution_mode mode = c.get_execution_mode();
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
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid execution mode for reporting results";
    throw lbann_exception(err.str());
  }

  if (comm->am_trainer_master()) {
    const int num_trainers = comm->get_num_trainers();

    auto& arg_parser = global_argument_parser();
    bool allow_global_statistics =
      arg_parser.get<bool>(LBANN_OPTION_LTFB_ALLOW_GLOBAL_STATISTICS);
    std::stringstream report;

    // Report objective function value
    const EvalType obj_fn = m->get_objective_function()->get_mean_value(mode);
    const int obj_fn_samples =
      m->get_objective_function()->get_statistics_num_samples(mode);

    if (allow_global_statistics) {
      if (comm->am_world_master()) {
        std::vector<EvalType> obj_fn_list(comm->get_num_trainers());
        std::vector<int> num_samples_list(comm->get_num_trainers());
        comm->intertrainer_gather(obj_fn, obj_fn_list);
        comm->intertrainer_gather(obj_fn_samples, num_samples_list);
        if (!m_print_global_stat_only) {
          for (int i = 0; i < num_trainers; ++i) {
            std::cout << m->get_name() << " (instance " << i << ") "
                      << mode_string << " "
                      << "objective function : " << obj_fn_list[i] << std::endl;
          }
        }
        if (num_trainers > 1) {
          const EvalType avg_obj_fn =
            (std::inner_product(num_samples_list.begin(),
                                num_samples_list.end(),
                                obj_fn_list.begin(),
                                EvalType(0)) /
             std::accumulate(num_samples_list.begin(),
                             num_samples_list.end(),
                             0));
          std::cout << m->get_name() << " global average " << mode_string << " "
                    << "objective function : " << avg_obj_fn << std::endl;
        }
      }
      else {
        comm->intertrainer_gather(obj_fn, comm->get_world_master());
        comm->intertrainer_gather(obj_fn_samples, comm->get_world_master());
      }
    }
    else {
      report << m->get_name() << " (instance " << comm->get_trainer_rank()
             << ") " << mode_string << " "
             << "objective function : " << obj_fn << std::endl;
    }

    // Report score for each metric
    for (const auto& met : m->get_metrics()) {
      const EvalType score = met->get_mean_value(mode);
      const int score_samples = met->get_statistics_num_samples(mode);
      if (allow_global_statistics) {
        if (comm->am_world_master()) {
          std::vector<EvalType> score_list(comm->get_num_trainers());
          std::vector<int> num_samples_list(comm->get_num_trainers());
          comm->intertrainer_gather(score, score_list);
          comm->intertrainer_gather(score_samples, num_samples_list);
          if (!m_print_global_stat_only) {
            for (int i = 0; i < num_trainers; ++i) {
              std::cout << m->get_name() << " (instance " << i << ") "
                        << mode_string << " " << met->name() << " : "
                        << score_list[i] << met->get_unit() << std::endl;
            }
          }
          if (num_trainers > 1) {
            const EvalType min_score =
              *std::min_element(score_list.begin(), score_list.end());
            const EvalType avg_score =
              (std::inner_product(num_samples_list.begin(),
                                  num_samples_list.end(),
                                  score_list.begin(),
                                  EvalType(0)) /
               std::accumulate(num_samples_list.begin(),
                               num_samples_list.end(),
                               0));
            const EvalType max_score =
              *std::max_element(score_list.begin(), score_list.end());
            EvalType scores_stdev = EvalType(0);
            for (const auto& t : score_list) {
              const auto& diff = t - avg_score;
              scores_stdev += diff * diff;
            }
            scores_stdev /= score_list.size() - 1;
            scores_stdev = El::Sqrt(std::max(scores_stdev, EvalType(0)));
            std::cout << m->get_name() << " (global average) " << mode_string
                      << " " << met->name() << " : " << avg_score
                      << met->get_unit() << std::endl;
            std::cout << m->get_name() << " (global min) " << mode_string << " "
                      << met->name() << " : " << min_score << met->get_unit()
                      << std::endl;
            std::cout << m->get_name() << " (global max) " << mode_string << " "
                      << met->name() << " : " << max_score << met->get_unit()
                      << std::endl;
            std::cout << m->get_name() << " (global stdev) " << mode_string
                      << " " << met->name() << " : " << scores_stdev
                      << met->get_unit() << std::endl;
          }
        }
        else {
          comm->intertrainer_gather(score, comm->get_intertrainer_master());
          comm->intertrainer_gather(score_samples,
                                    comm->get_intertrainer_master());
        }
      }
      else {
        report << m->get_name() << " (instance " << comm->get_trainer_rank()
               << ") " << mode_string << " " << met->name() << " : " << score
               << met->get_unit() << std::endl;
      }
    }

    // If global statistics are not reported have every trainer report
    // them here
    if (!allow_global_statistics) {
      std::cout << report.str() << std::flush;
    }
  }
}

std::unique_ptr<callback_base> build_print_statistics_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackPrint&>(proto_msg);
  return std::make_unique<print_statistics>(params.interval(),
                                            params.print_global_stat_only());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::print_statistics
#define LBANN_CLASS_LIBNAME callback_print_statistics
#include <lbann/macros/register_class_with_cereal.hpp>
