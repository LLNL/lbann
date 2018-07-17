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

void layer_term::set_evaluation_layer(abstract_evaluation_layer* l) {
  if (l == nullptr) {
    this->m_layers.clear();
  } else {
    this->m_layers.assign(1, l);
  }
}

abstract_evaluation_layer* layer_term::get_evaluation_layer() {
  if (m_layers.empty()) {
    return nullptr;
  } else {
    return dynamic_cast<abstract_evaluation_layer*>(this->m_layers.front());
  }
}

void layer_term::setup(model& m) {
  objective_function_term::setup(m);
  std::stringstream err;

  // Make sure layer term points to an evaluation layer
  if (this->m_layers.size() != 1) {
    err << "objective function layer term points to an invalid number of layers "
        << "(expected 1, found " << this->m_layers.size() << ")";
    LBANN_ERROR(err.str());
  }
  auto&& l = this->m_layers.front();
  auto&& eval = dynamic_cast<abstract_evaluation_layer*>(l);
  if (l == nullptr) {
    LBANN_ERROR("objective function layer term points to a null pointer");
  } else if (eval == nullptr) {
    err << "objective function layer term points to "
        << l->get_type() << " layer \"" << l->get_name() << "\", "
        << "which is not an evaluation layer";
    LBANN_ERROR(err.str());
  }

  // Set scaling factor
  eval->set_scale(m_scale_factor);

}

void layer_term::start_evaluation() {}

EvalType layer_term::finish_evaluation() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  auto&& eval = dynamic_cast<abstract_evaluation_layer*>(this->m_layers.front());
  eval->set_scale(m_scale_factor);
  return eval->get_value();
}

void layer_term::differentiate() {
  auto&& eval = dynamic_cast<abstract_evaluation_layer*>(this->m_layers.front());
  eval->set_scale(m_scale_factor);
}

}  // namespace lbann
