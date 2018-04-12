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

#include "lbann/objective_functions/layer_term.hpp"

namespace lbann {

layer_term::layer_term(EvalType scale_factor)
  : objective_function_term(scale_factor) {}

void layer_term::set_evaluation_layer(Layer* l) {
  this->m_layers.assign(1, l);
}

void layer_term::setup(model& m) {
  objective_function_term::setup(m);
  std::stringstream err;

  // Make sure layer term points to an evaluation layer
  if (m_layers.size() != 1) {
    err << "layer term in objective function points to an invalid number of layers "
        << "(expected 1, found " << m_layers.size() << ")";
    LBANN_ERROR(err.str());
  }
  if (m_layers[0] == nullptr) {
    LBANN_ERROR("layer term in objective function points to a null pointer");
  }
  auto&& eval = dynamic_cast<evaluation_layer<data_layout::DATA_PARALLEL>*>(m_layers[0]);
  if (eval == nullptr) {
    err << "layer term in objective function must point to an evaluation layer, "
        << "but " << m_layers[0]->get_name() << " is type " << m_layers[0]->get_type();
    LBANN_ERROR(err.str());
  }

  // Set scaling factor
  eval->set_scale(m_scale_factor);

}

void layer_term::start_evaluation() {}

EvalType layer_term::finish_evaluation() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  auto&& eval = dynamic_cast<evaluation_layer<data_layout::DATA_PARALLEL>*>(m_layers[0]);
  eval->set_scale(m_scale_factor);
  return eval->get_value();
}

void layer_term::differentiate() {
  auto&& eval = dynamic_cast<evaluation_layer<data_layout::DATA_PARALLEL>*>(m_layers[0]);
  eval->set_scale(m_scale_factor);
}

}  // namespace lbann
