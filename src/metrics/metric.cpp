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

#include "lbann/metrics/metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

template <class Archive>
void metric_statistics::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_sum), CEREAL_NVP(m_num_samples));
}

void metric_statistics::add_value(EvalType total_value, int num_samples)
{
  m_sum += total_value;
  m_num_samples += num_samples;
}

EvalType metric_statistics::get_mean() const
{
  if (m_num_samples == 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to get mean metric value with no statistics";
    throw lbann_exception(err.str());
  }
  return m_sum / m_num_samples;
}

void metric_statistics::reset()
{
  m_sum = 0.0;
  m_num_samples = 0;
}

metric::metric(lbann_comm* comm) : m_comm(comm) {}

template <class Archive>
void metric::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_statistics));
}

EvalType metric::get_mean_value(execution_mode mode) const
{
  if (m_statistics.count(mode) == 0 ||
      m_statistics.at(mode).get_num_samples() == 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to get mean metric value with no samples for statistics";
    throw lbann_exception(err.str());
  }
  return m_statistics.at(mode).get_mean();
}

int metric::get_statistics_num_samples(execution_mode mode) const
{
  if (m_statistics.count(mode) == 0) {
    return 0;
  }
  else {
    return m_statistics.at(mode).get_num_samples();
  }
}

std::vector<ViewingLayerPtr> metric::get_layer_pointers() const { return {}; }

void metric::set_layer_pointers(std::vector<ViewingLayerPtr> layers)
{
  if (!layers.empty()) {
    std::stringstream err;
    err << "attempted to set layer pointers for "
        << "metric \"" << name() << "\" "
        << "with an invalid number of pointers "
        << "(expected 0, found " << layers.size() << ")";
    LBANN_ERROR(err.str());
  }
}

} // namespace lbann

#define LBANN_SKIP_CEREAL_REGISTRATION
#define LBANN_CLASS_NAME metric
#include <lbann/macros/register_class_with_cereal.hpp>

#undef LBANN_CLASS_NAME
#define LBANN_SKIP_CEREAL_REGISTRATION
#define LBANN_CLASS_NAME metric_statistics
#include <lbann/macros/register_class_with_cereal.hpp>
