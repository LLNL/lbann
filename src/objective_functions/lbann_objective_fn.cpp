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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/objective_functions/lbann_objective_fn.hpp"
#include "lbann/models/lbann_model.hpp"

using namespace std;

lbann::objective_functions::statistics *lbann::objective_functions::objective_fn::get_statistics(execution_mode mode) {
  statistics *stats;

  switch(mode) {
  case execution_mode::training:
    stats = &m_training_stats;
    break;
  case execution_mode::validation:
    stats = &m_validation_stats;
    break;
  case execution_mode::testing:
    stats = &m_testing_stats;
    break;
  default:
    throw lbann_exception("Invalid execution mode");
  };
  return stats;
}


void lbann::objective_functions::objective_fn::record_obj_fn(execution_mode mode, double avg_cost) {
  statistics *stats = get_statistics(mode);
  stats->m_last_mini_batch_avg_cost = avg_cost;
  stats->m_aggregate_avg_cost_per_epoch += avg_cost;
  stats->m_num_mini_batch_per_epoch += 1;
  return;
}

double lbann::objective_functions::objective_fn::report_obj_fn(execution_mode mode) {
  statistics *stats = get_statistics(mode);
  return stats->m_last_mini_batch_avg_cost;
}

double lbann::objective_functions::objective_fn::report_aggregate_avg_obj_fn(execution_mode mode) {
  statistics *stats = get_statistics(mode);
  if(stats->m_num_mini_batch_per_epoch == 0) {
    return std::numeric_limits<double>::max();
  } else {
    return (stats->m_aggregate_avg_cost_per_epoch / stats->m_num_mini_batch_per_epoch);
  }
}

void lbann::objective_functions::objective_fn::reset_obj_fn() {
  m_training_stats.reset_stats();
  m_validation_stats.reset_stats();
  m_testing_stats.reset_stats();
}
