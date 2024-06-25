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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/check_metric.hpp"

#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/serialize.hpp"
#include <cereal/types/set.hpp>

#include "lbann/proto/callbacks.pb.h"

#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <utility>

namespace lbann {
namespace callback {

check_metric::check_metric(std::string metric_name,
                           std::set<execution_mode> modes,
                           EvalType lower_bound,
                           EvalType upper_bound,
                           bool error_on_failure)
  : m_metric_name(std::move(metric_name)),
    m_modes(std::move(modes)),
    m_lower_bound(lower_bound),
    m_upper_bound(upper_bound),
    m_error_on_failure(error_on_failure)
{
  if (lower_bound > upper_bound) {
    std::stringstream err;
    err << "callback \"" << name() << "\" "
        << "got an invalid range for metric values " << std::setprecision(9)
        << "(lower bound " << m_lower_bound << ", "
        << "upper bound " << m_upper_bound << ")";
    LBANN_ERROR(err.str());
  }
  if (lower_bound == upper_bound) {
    std::stringstream err;
    err << "callback \"" << name() << "\" "
        << "got an zero range for metric values " << std::setprecision(9)
        << "(lower bound " << m_lower_bound << " == "
        << "upper bound " << m_upper_bound << ")";
    LBANN_WARNING(err.str());
  }
}

check_metric::check_metric() : check_metric("", {}, 0, 0, false) {}

template <class Archive>
void check_metric::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_metric_name),
     CEREAL_NVP(m_modes),
     CEREAL_NVP(m_lower_bound),
     CEREAL_NVP(m_upper_bound),
     CEREAL_NVP(m_error_on_failure));
}

void check_metric::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_check_metric();
  msg->set_metric(m_metric_name);
  msg->set_lower_bound(m_lower_bound);
  msg->set_upper_bound(m_upper_bound);
  msg->set_error_on_failure(m_error_on_failure);
  std::string modes;
  for (auto const& mode : m_modes)
    modes += (to_string(mode) + " ");
  msg->set_execution_modes(modes);
}

void check_metric::do_check_metric(const model& m) const
{
  const auto& c = m.get_execution_context();
  std::stringstream err;

  // Return immediately if execution mode is invalid
  const auto& mode = c.get_execution_mode();
  if (!m_modes.empty() && m_modes.count(mode) == 0) {
    return;
  }

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
        << "to have a value in range " << std::setprecision(9) << "["
        << m_lower_bound << "," << m_upper_bound << "], "
        << "but found a value of " << value;
    if (m_error_on_failure) {
      LBANN_ERROR(err.str());
    }
    else if (m.get_comm()->am_trainer_master()) {
      LBANN_WARNING(err.str());
    }
  }
}

std::unique_ptr<callback_base> build_check_metric_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  std::shared_ptr<lbann_summary> const&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackCheckMetric&>(proto_msg);
  const auto& modes = parse_set<execution_mode>(params.execution_modes());
  return std::make_unique<check_metric>(params.metric(),
                                        modes,
                                        params.lower_bound(),
                                        params.upper_bound(),
                                        params.error_on_failure());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::check_metric
#define LBANN_CLASS_LIBNAME callback_check_metric
#include <lbann/macros/register_class_with_cereal.hpp>
