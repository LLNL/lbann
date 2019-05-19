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

#include "lbann/objective_functions/layer_term.hpp"

namespace lbann {

layer_term::layer_term(EvalType scale_factor)
  : objective_function_term(scale_factor) {}

void layer_term::set_layer(Layer& l) {
  set_layer_pointers({&l});
}

Layer& layer_term::get_layer() {
  // Idiom from Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers.
  return *(const_cast<Layer*>(&static_cast<const layer_term&>(*this).get_layer()));
}
const Layer& layer_term::get_layer() const {
  const auto& layer_pointers = get_layer_pointers();
  if (layer_pointers.empty() || layer_pointers.front() == nullptr) {
    LBANN_ERROR("attempted to get the layer corresponding to "
                "an objective function layer term, "
                "but no such layer has been set");
  }
  return *layer_pointers.front();
}

abstract_evaluation_layer& layer_term::get_evaluation_layer() {
  auto& l = get_layer();
  auto* eval = dynamic_cast<abstract_evaluation_layer*>(&l);
  if (eval == nullptr) {
    std::stringstream err;
    err << "attempted to get the evaluation layer corresponding to "
        << "an objective function layer term, "
        << "but the layer term currently corresponds to "
        << l.get_type() << " layer \"" << l.get_name() << "\"";
    LBANN_ERROR(err.str());
  }
  return *eval;
}

void layer_term::setup(model& m) {
  objective_function_term::setup(m);
  get_evaluation_layer().set_scale(m_scale_factor);
}

void layer_term::start_evaluation() {}

EvalType layer_term::finish_evaluation() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  auto& eval = get_evaluation_layer();
  eval.set_scale(m_scale_factor);
  return eval.get_value();
}

void layer_term::differentiate() {
  get_evaluation_layer().set_scale(m_scale_factor);
}

}  // namespace lbann
