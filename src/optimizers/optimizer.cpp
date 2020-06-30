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

#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

std::string to_string(optimizer_gradient_status status) {
  switch (status) {
  case optimizer_gradient_status::ready:
    return "ready";
  case optimizer_gradient_status::cleared:
    return "cleared";
  case optimizer_gradient_status::allreduce_needed:
    return "allreduce needed";
  case optimizer_gradient_status::allreduce_started:
    return "allreduce started";
  default:
    return "unknown";
  }
}

optimizer::optimizer()
  : m_comm(nullptr) {}

optimizer::optimizer(const optimizer& other)
  : m_comm(other.m_comm),
    m_gradient_sources(other.m_gradient_sources),
    m_gradient_status(other.m_gradient_status),
    m_step_time(other.m_step_time) {
  if (m_gradient_status == optimizer_gradient_status::allreduce_started) {
    LBANN_ERROR("attempted to copy optimizer while a "
                "gradient allreduce is in progress");
  }
}

optimizer& optimizer::operator=(const optimizer& other) {
  m_comm = other.m_comm;
  m_gradient_sources = other.m_gradient_sources;
  m_gradient_status = other.m_gradient_status;
  m_step_time = other.m_step_time;
  if (m_gradient_status == optimizer_gradient_status::allreduce_started) {
    LBANN_ERROR("attempted to copy optimizer while a "
                "gradient allreduce is in progress");
  }
  return *this;
}

description optimizer::get_description() const {
  description desc(get_type() + " optimizer");
  return desc;
}

El::Int optimizer::get_num_gradient_sources() const {
  return m_gradient_sources.size();
}

void optimizer::add_gradient_source(const void* source) {
  if (source != nullptr) {
    m_gradient_sources.insert(source);
  }
}

void optimizer::remove_gradient_source(const void* source) {
  m_gradient_sources.erase(nullptr);
  m_gradient_sources.erase(source);
  if (get_gradient_sources().empty()) {
    start_gradient_allreduce();
  }
}

} // namespace lbann
