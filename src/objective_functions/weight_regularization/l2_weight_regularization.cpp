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

#include "lbann/objective_functions/weight_regularization/l2_weight_regularization.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

void l2_weight_regularization::setup(objective_function& obj_fn) {
  objective_function_term::setup(obj_fn);

  // Check that term has no layer pointers
  if (!m_layers.empty()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to setup L2 weight regularization with layer pointers";
    throw lbann_exception(err.str());
  }

  // Add all weights in model if no weights pointers are provided
  if (m_weights.empty()) {
    for (weights* w : m_objective_function->get_model()->get_weights()) {
      if (w->get_optimizer() != nullptr) {
        m_weights.push_back(w);
      }
    }
  }

}

DataType l2_weight_regularization::compute_value() {
  if (m_scale_factor == DataType(0)) { return DataType(0); }
  DataType value = DataType(0);
  for (weights* w : m_weights) {
    DataType norm = El::FrobeniusNorm(w->get_values());
    value += norm * norm;
  }
  return m_scale_factor * value;
}

void l2_weight_regularization::compute_gradient() {
  if (m_scale_factor == DataType(0)) { return; }
  for (weights* w : m_weights) {
    optimizer* opt = w->get_optimizer();
    opt->add_to_gradient(w->get_values(), 2 * m_scale_factor);
  }
}

}  // namespace lbann
