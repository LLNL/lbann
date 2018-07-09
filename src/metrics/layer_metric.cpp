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

#include "lbann/metrics/layer_metric.hpp"
#include "lbann/layers/transform/evaluation.hpp"

namespace lbann {

layer_metric::layer_metric(lbann_comm *comm, std::string unit)
  : metric(comm), m_unit(unit), m_evaluation_layer(nullptr) {}

std::string layer_metric::name() const {
  if (m_evaluation_layer != nullptr) {
    return m_evaluation_layer->get_name();
  } else {
    return "uninitialized layer metric";
  }
}

void layer_metric::setup(model& m) {
  if (m_evaluation_layer == nullptr) {
    LBANN_ERROR("attempted to setup layer metric without setting evaluation layer");
  }
}

EvalType layer_metric::evaluate(execution_mode mode,
                                int mini_batch_size) {

  // Check if evaluation layer pointer has been setup
  if (m_evaluation_layer == nullptr) {
    LBANN_ERROR("attempted to evaluate metric without setting a target layer");
  }

  // Get evaluation layer value
  const auto& start = get_time();
  auto value = m_evaluation_layer->get_value(false);
  if (m_unit == "%") { value *= 100; }
  get_evaluate_time() += get_time() - start;

  // Record result in statistics and return
  get_statistics()[mode].add_value(value * mini_batch_size, mini_batch_size);
  return value;

}

} // namespace lbann
