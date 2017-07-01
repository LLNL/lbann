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

#include "lbann/metrics/metric.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

namespace metrics {

statistics *metric::get_statistics(execution_mode mode) {
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

void metric::record_error(double error, long num_samples) {
  statistics *stats = get_statistics(m_neural_network_model->get_execution_mode());
  stats->m_error_per_epoch += error;
  stats->m_samples_per_epoch += num_samples;
  return;
}

void metric::reset_metric() {
  statistics *stats = get_statistics(m_neural_network_model->get_execution_mode());
  stats->reset_stats();
}

}  // namespace metrics

}  // namespace lbann
