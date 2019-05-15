////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/callback_check_metric.hpp"

namespace lbann {

lbann_callback_check_metric::lbann_callback_check_metric(std::string metric_name,
                                                         std::set<execution_mode> modes,
                                                         EvalType lower_bound,
                                                         EvalType upper_bound,
                                                         bool error_on_failure)
  : m_metric_name(std::move(metric_name)),
    m_modes(std::move(modes)),
    m_lower_bound(lower_bound),
    m_upper_bound(upper_bound),
    m_error_on_failure(error_on_failure) {
  if (lower_bound > upper_bound) {
    std::stringstream err;
    err << "callback \"" << name() << "\" "
        << "got an invalid range for metric values "
        << "(lower bound " << m_lower_bound << ", "
        << "upper bound " << m_upper_bound << ")";
    LBANN_ERROR(err.str());
  }
}


void lbann_callback_check_metric::check_metric(const model& m) const {
  std::stringstream err;

  // Return immediately if execution mode is invalid
  const auto& mode = m.get_execution_mode();
  if (!m_modes.empty() && m_modes.count(mode) == 0) { return; }

  // Get metric
  const metric* met = nullptr;
  for (const auto* met_ : m.get_metrics()) {
    if (met_->name() == m_metric_name) {
      met = met_;
    }
  }
  if (met == nullptr) {
    err << "callback \"" << name() << "\" could not find "
        << "metric \"" << m_metric_name << "\"";
    LBANN_ERROR(err.str());
  }

  // Check if metric value is within expected range
  const auto& value = met->get_mean_value(mode);
  if (!(m_lower_bound <= value && value <= m_upper_bound)) {
    err << "callback \"" << name() << "\" expected "
        << "metric \"" << m_metric_name << "\" "
        << "to have a value in range "
        << "[" << m_lower_bound << "," << m_upper_bound << "], "
        << "but found a value of " << value;
    if (m_error_on_failure) {
      LBANN_ERROR(err.str());
    } else if (m.get_comm()->am_trainer_master()) {
      LBANN_WARNING(err.str());
    }
  }

}

}  // namespace lbann
