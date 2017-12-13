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
#include "lbann/layers/io/target/target_layer.hpp"

namespace lbann {

metric::metric(lbann_comm *comm) 
  : m_comm(comm),
    m_target_layer(nullptr) {}

void metric::setup(model& m) {

  // Set target layer if needed
  if (m_target_layer == nullptr) {
    std::vector<Layer*> layers = m.get_layers();
    for (int i = layers.size() - 1; i >= 0; --i) {
      const target_layer *target = dynamic_cast<const target_layer*>(layers[i]);
      if (target != nullptr) {
        m_target_layer = target;
        break;
      }
    }
    if (m_target_layer == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "could not setup metric with target layer";
      throw lbann_exception(err.str());
    }
  }

}

DataType metric::evaluate() {

  // Check if target layer pointer has been setup
  if (m_target_layer == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to evaluate metric without setting a target layer";
    throw lbann_exception(err.str());
  }

  // Evaluate objective function
  const DataType value = evaluate_compute(m_target_layer->get_prediction(),
                                          m_target_layer->get_ground_truth());

  // Record result in history
  m_history_values.push_back(value);
  const int mini_batch_size = m_target_layer->get_prediction().Width();
  m_history_mini_batch_sizes.push_back(mini_batch_size);

  // Return result
  return value;

}

int metric::get_history_num_samples() const {
  return std::accumulate(m_history_mini_batch_sizes.begin(),
                         m_history_mini_batch_sizes.end(),
                         0);
}

void metric::clear_history() {
  m_history_values.clear();
  m_history_mini_batch_sizes.clear();
}

DataType metric::get_history_mean_value() const {
  if (m_history_values.size() == 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to get mean metric value with no history";
    throw lbann_exception(err.str());
  }
  return (std::inner_product(m_history_values.begin(),
                             m_history_values.end(),
                             m_history_mini_batch_sizes.begin(),
                             DataType(0))
          / get_history_num_samples());
}

const target_layer& metric::get_target_layer() const {
  if (m_target_layer == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access target layer before it is set";
    throw lbann_exception(err.str());
  } 
  return *m_target_layer;
}

std::vector<Layer*> metric::get_layer_pointers() const {
  return std::vector<Layer*>(1, const_cast<target_layer *>(m_target_layer));
}

void metric::set_layer_pointers(std::vector<Layer*> layers) {
  if (layers.size() != 1) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to set layer pointers with an invalid number of pointers "
        << "(expected 1, found " << layers.size() << ")";
    throw lbann_exception(err.str());
  }
  m_target_layer = dynamic_cast<const target_layer *>(layers[0]);
}

}  // namespace lbann
