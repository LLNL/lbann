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

#include "lbann/metrics/metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

void metric_statistics::add_value(EvalType total_value, int num_samples) {
  m_sum += total_value;
  m_num_samples += num_samples;
}

EvalType metric_statistics::get_mean() const {
  if (m_num_samples == 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to get mean metric value with no statistics";
    throw lbann_exception(err.str());
  }
  return m_sum / m_num_samples;
}

void metric_statistics::reset() {
  m_sum = 0.0;
  m_num_samples = 0;
}

bool metric_statistics::pack_scalars(persist& p) {
  p.write_double(persist_type::validate, "sum", m_sum);
  p.write_uint64(persist_type::validate, "num_samples", m_num_samples);
  return true;
}

bool metric_statistics::unpack_scalars(persist& p, struct packing_header *header) {
  double sum;
  uint64_t num_samples;
  p.read_double(persist_type::validate, "sum", &sum);
  p.read_uint64(persist_type::validate, "num_samples", (uint64_t *) &num_samples);
  m_sum = sum;
  m_num_samples = num_samples;
  if (header != nullptr) {
    header->sum = sum;
    header->num_samples = num_samples;
  }
  return true;
}

void metric_statistics::unpack_header(struct packing_header& header) {
  m_sum = header.sum;
  m_num_samples = header.num_samples;
}

metric::metric(lbann_comm *comm) : m_comm(comm) {}

EvalType metric::get_mean_value(execution_mode mode) const {
  if (m_statistics.count(mode) == 0
      || m_statistics.at(mode).get_num_samples() == 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to get mean metric value with no samples for statistics";
    throw lbann_exception(err.str());
  }
  return m_statistics.at(mode).get_mean();
}

int metric::get_statistics_num_samples(execution_mode mode) const {
  if (m_statistics.count(mode) == 0) {
    return 0;
  } else {
    return m_statistics.at(mode).get_num_samples();
  }
}

std::vector<Layer*> metric::get_layer_pointers() const {
  return {};
}

void metric::set_layer_pointers(std::vector<Layer*> layers) {
  if (!layers.empty()) {
    std::stringstream err;
    err << "attempted to set layer pointers for "
        << "metric \"" << name() << "\" "
        << "with an invalid number of pointers "
        << "(expected 0, found " << layers.size() << ")";
    LBANN_ERROR(err.str());
  }
}

bool metric::save_to_checkpoint_shared(persist& p) {
  // write out fields we need to save for model
  if (m_comm->am_trainer_master()) {
    m_statistics[execution_mode::training].pack_scalars(p);
    m_statistics[execution_mode::testing].pack_scalars(p);
    m_statistics[execution_mode::validation].pack_scalars(p);
  }
  return true;
}

bool metric::load_from_checkpoint_shared(persist& p) {
  struct metric_statistics::packing_header training_header, validation_header, testing_header;
  if (m_comm->am_trainer_master()) {
    m_statistics[execution_mode::training].unpack_scalars(p, &training_header);
    m_statistics[execution_mode::testing].unpack_scalars(p, &testing_header);
    m_statistics[execution_mode::validation].unpack_scalars(p, &validation_header);
  }

  m_comm->trainer_broadcast(0, training_header);
  m_comm->trainer_broadcast(0, validation_header);
  m_comm->trainer_broadcast(0, testing_header);

  m_statistics[execution_mode::training].unpack_header(training_header);
  m_statistics[execution_mode::validation].unpack_header(validation_header);
  m_statistics[execution_mode::testing].unpack_header(testing_header);
  return true;
}

bool metric::save_to_checkpoint_distributed(persist& p) {
  // write out fields we need to save for model
  m_statistics[execution_mode::training].pack_scalars(p);
  m_statistics[execution_mode::testing].pack_scalars(p);
  m_statistics[execution_mode::validation].pack_scalars(p);
  return true;
}

bool metric::load_from_checkpoint_distributed(persist& p) {
  struct metric_statistics::packing_header training_header, validation_header, testing_header;
  m_statistics[execution_mode::training].unpack_scalars(p, &training_header);
  m_statistics[execution_mode::testing].unpack_scalars(p, &testing_header);
  m_statistics[execution_mode::validation].unpack_scalars(p, &validation_header);

  m_statistics[execution_mode::training].unpack_header(training_header);
  m_statistics[execution_mode::validation].unpack_header(validation_header);
  m_statistics[execution_mode::testing].unpack_header(testing_header);
  return true;
}

}  // namespace lbann
