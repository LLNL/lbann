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
// lbann_callback_print .hpp .cpp - Callback hooks to print information
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include "lbann/callbacks/callback_print.hpp"
#include <iomanip>

namespace lbann {

void lbann_callback_print::setup(model *m) {
#ifdef LBANN_VERSION
  lbann_comm *comm = m->get_comm();
  if (comm->am_world_master()) {
    std::cout << "Training with LLNL LBANN version "
              << LBANN_MAKE_STR(LBANN_VERSION) << std::endl;
  }
#endif
}

void lbann_callback_print::on_epoch_begin(model *m) {
  lbann_comm *comm = m->get_comm();
  if (comm->am_world_master()) {
    const std::vector<Layer *>layers = m->get_layers();
    auto *layer = dynamic_cast<input_layer*>(layers[0]);
    std::cout << "--------------------------------------------------------------------------------" 
              << std::endl;
    std::cout << "[" << m->get_cur_epoch() << "] Epoch : stats formated [tr/v/te]" 
              << " iter/epoch ="
              << " ["
              << layer->get_num_iterations_per_epoch(execution_mode::training)
              << "/"
              << layer->get_num_iterations_per_epoch(execution_mode::validation)
              << "/"
              << layer->get_num_iterations_per_epoch(execution_mode::testing)
              << "]"
              << std::endl;
    std::cout << std::setfill(' ') << std::setw(23)
              << " global MB ="
              << " ["
              << std::setw(4) << layer->get_global_mini_batch_size(execution_mode::training)
              << "/"
              << std::setw(4) << layer->get_global_mini_batch_size(execution_mode::validation)
              << "/"
              << std::setw(4) << layer->get_global_mini_batch_size(execution_mode::testing)
              << "]"
              << " global last MB ="
              << " ["
              << std::setw(4) << layer->get_global_last_mini_batch_size(execution_mode::training)
              << "/"
              << std::setw(4) << layer->get_global_last_mini_batch_size(execution_mode::validation)
              << "/"
              << std::setw(4) << layer->get_global_last_mini_batch_size(execution_mode::testing)
              << "]"
              << std::endl;
    std::cout << std::setfill(' ') << std::setw(23)
              << "  local MB ="
              << " ["
              << std::setw(4) << layer->get_mini_batch_size(execution_mode::training)
              << "/"
              << std::setw(4) << layer->get_mini_batch_size(execution_mode::validation)
              << "/"
              << std::setw(4) << layer->get_mini_batch_size(execution_mode::testing)
              << "]"
              << "  local last MB ="
              << " ["
              << std::setw(4) << layer->get_last_mini_batch_size(execution_mode::training)
              << "/"
              << std::setw(4) << layer->get_last_mini_batch_size(execution_mode::validation)
              << "/"
              << std::setw(4) << layer->get_last_mini_batch_size(execution_mode::testing)
              << "]"
              << std::endl;
    std::cout << "--------------------------------------------------------------------------------"
              << std::endl;
  }
}

void lbann_callback_print::on_epoch_end(model *m) {
  report_results(m);
}

void lbann_callback_print::on_validation_end(model *m) {
  report_results(m);
}

void lbann_callback_print::on_test_end(model *m) {
  report_results(m);
}

void lbann_callback_print::report_results(model *m) {
  lbann_comm *comm = m->get_comm();

  // Get string for execution mode
  const execution_mode mode = m->get_execution_mode();
  std::string mode_string;
  switch (mode) {
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

  if (comm->am_model_master()) {
    const int num_models = comm->get_num_models();

    // Report objective function value
    const EvalType obj_fn = m->get_objective_function()->get_mean_value(mode);
    if (comm->am_world_master()) {
      std::vector<EvalType> obj_fn_list(comm->get_num_models());
      comm->intermodel_gather(obj_fn, obj_fn_list);
      for (int i = 0; i < num_models; ++i) {
        std::cout << "Model " << i << " " << mode_string << " "
                  << "objective function : " << obj_fn_list[i]
                  << std::endl;
      }
      if (num_models > 1) {
        const EvalType avg_obj_fn = (std::accumulate(obj_fn_list.begin(),
                                                     obj_fn_list.end(),
                                                     EvalType(0))
                                     / num_models);
        std::cout << "World average " << mode_string << " "
                  << "objective function : " << avg_obj_fn
                  << std::endl;
      }
    } else {
      comm->intermodel_gather(obj_fn, comm->get_world_master());
    }

    // Report score for each metric
    for (const auto& met : m->get_metrics()) {
      const EvalType score = met->get_mean_value(mode);
      const int num_samples = met->get_statistics_num_samples(mode);
      if (comm->am_world_master()) {
        std::vector<EvalType> score_list(comm->get_num_models());
        std::vector<int> num_samples_list(comm->get_num_models());
        comm->intermodel_gather(score, score_list);
        comm->intermodel_gather(num_samples, num_samples_list);
        for (int i = 0; i < num_models; ++i) {
          std::cout << "Model " << i << " " << mode_string << " "
                    << met->name() << " : " 
                    << score_list[i] << met->get_unit()
                    << std::endl;
        }
        if (num_models > 1) {
          const EvalType avg_score = (std::inner_product(score_list.begin(),
                                                         score_list.end(),
                                                         num_samples_list.begin(),
                                                         EvalType(0))
                                      / std::accumulate(num_samples_list.begin(),
                                                        num_samples_list.end(),
                                                        EvalType(0)));
          std::cout << "World " << mode_string << " "
                    << met->name() << " : "
                    << avg_score << met->get_unit()
                    << std::endl;
        }
      } else {
        comm->intermodel_gather(score, comm->get_intermodel_master());
        comm->intermodel_gather(num_samples, comm->get_intermodel_master());
      }
    }

  }
  
}

}  // namespace lbann
