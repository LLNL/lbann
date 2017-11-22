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

#include "lbann/objective_functions/loss_functions/loss_function.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

loss_function::loss_function(DataType scale_factor)
  : objective_function_term(scale_factor),
    m_gradient(nullptr) {}

loss_function::loss_function(const loss_function& other)
  : objective_function_term(other),
    m_gradient(other.m_gradient) {
  if (m_gradient != nullptr) { m_gradient = m_gradient->Copy(); }
}

loss_function& loss_function::operator=(const loss_function& other) {
  objective_function_term::operator=(other);
  if (m_gradient != nullptr && other.m_gradient != nullptr
      && m_gradient->DistData() == other.m_gradient->DistData()) {
    El::Copy(*other.m_gradient, *m_gradient);
  }
  else {
    if (m_gradient != nullptr) { delete m_gradient; }
    m_gradient = other.m_gradient;
    if (m_gradient != nullptr) { m_gradient = m_gradient->Copy(); }
  }
  return *this;
}

loss_function::~loss_function() {
  if (m_gradient != nullptr) { delete m_gradient; }
}

void loss_function::set_target_layer(target_layer *layer) {
  if (m_layers.size() > 0) {
    m_layers[0] = layer;
  } else {
    m_layers.push_back(layer);
  }
}

void loss_function::setup(objective_function& obj_fn) {
  objective_function_term::setup(obj_fn);
  
  // Check that loss function has one target layer and no weights
  // Note: if target layer is not specified, choose the latest target
  // layer in the model.
  if (m_layers.size() > 1) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to setup loss function with invalid number of target layers "
        << "(expected 1, found " << m_layers.size() << ")";
    throw lbann_exception(err.str());
  }
  if (m_layers.empty()) {
    std::vector<Layer*> layers
      = this->m_objective_function->get_model()->get_layers();
    for (int i = layers.size() - 1; i >= 0; --i) {
      if (dynamic_cast<target_layer*>(layers[i]) != nullptr) {
        m_layers.push_back(layers[i]);
        break;
      }
    }
  }
  if (m_layers.size() != 1
      || dynamic_cast<target_layer*>(m_layers[0]) == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "could not setup loss function with target layer";
    throw lbann_exception(err.str());
  }
  if (m_weights.size() != 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to setup loss function with invalid number of weights "
        << "(expected 0, found " << m_weights.size() << ")";
    throw lbann_exception(err.str());
  }

  // Initialize gradient matrix
  target_layer *target = (target_layer*) m_layers[0];
  const AbsDistMat& ground_truth = target->get_activations();
  m_gradient = ground_truth.Construct(ground_truth.Grid(),
                                      ground_truth.Root());
  El::Zeros(*m_gradient, ground_truth.Height(), ground_truth.Width());

}

/// @todo Handle effective mini-batch size
DataType loss_function::compute_value() {
  if (m_scale_factor == DataType(0)) { return DataType(0); }
  target_layer *target = (target_layer*) m_layers[0];
  const AbsDistMat& prediction = target->get_prediction();
  const AbsDistMat& ground_truth = target->get_ground_truth();
  return m_scale_factor * evaluate(prediction, ground_truth);
}

void loss_function::compute_gradient() {
  if (m_scale_factor == DataType(0)) { return; }
  target_layer *target = (target_layer*) m_layers[0];
  const AbsDistMat& prediction = target->get_prediction();
  const AbsDistMat& ground_truth = target->get_ground_truth();
  El::Zeros(*m_gradient, prediction.Height(), prediction.Width());
  differentiate(prediction, ground_truth, *m_gradient);
  target->add_to_error_signal(*m_gradient, m_scale_factor);
}


}  // namespace lbann
