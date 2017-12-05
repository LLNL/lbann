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

#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/utils/timer.hpp"
#include <numeric>

namespace lbann {

objective_function::objective_function() 
  : m_model(nullptr) {}

objective_function::objective_function(const objective_function& other)
  : m_model(other.m_model),
    m_history(other.m_history),
    m_value_time(other.m_value_time),
    m_gradient_time(other.m_gradient_time) {
  for (objective_function_term *term : other.m_terms) {
    m_terms.push_back(term->copy());
    m_terms.back()->set_objective_function(this);
  }
}

objective_function& objective_function::operator=(const objective_function& other) {
  m_model = other.m_model;
  for (objective_function_term *term : m_terms) {
    if (term != nullptr) { delete term; }
  }
  m_terms.clear();
  for (objective_function_term *term : other.m_terms) {
    m_terms.push_back(term->copy());
    m_terms.back()->set_objective_function(this);
  }
  m_history = other.m_history;
  m_value_time = other.m_value_time;
  m_gradient_time = other.m_gradient_time;
  return *this;
}

objective_function::~objective_function() {
  for (objective_function_term *term : m_terms) {
    if (term != nullptr) { delete term; }
  }
}

void objective_function::setup(model& m) {
  m_model = &m;
  for (objective_function_term *term : m_terms) {
    if (term == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "a term in the objective function is a null pointer";
      throw lbann_exception(err.str());
    }
    term->setup(*this);
  }
}

DataType objective_function::compute_value() {
  double value_start = get_time();
  auto value = DataType(0);
  for (objective_function_term *term : m_terms) {
    value += term->compute_value();
  }
  m_history.push_back(value);
  m_value_time += get_time() - value_start;
  return value;
}

void objective_function::compute_gradient() {
  double gradient_start = get_time();
  for (objective_function_term *term : m_terms) {
    term->compute_gradient();
  }
  m_gradient_time += get_time() - gradient_start;
}

DataType objective_function::get_history_mean_value() const {
  if (m_history.size() == 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to get mean objective function value with no history";
    throw lbann_exception(err.str());
  }
  return (std::accumulate(m_history.begin(),
                          m_history.end(),
                          DataType(0))
          / m_history.size());
}

std::vector<Layer*> objective_function::get_layer_pointers() const {
  std::vector<Layer*> layers;
  for (objective_function_term *term : m_terms) {
    std::vector<Layer*> term_layers = term->get_layer_pointers();
    layers.insert(layers.end(), term_layers.begin(), term_layers.end());
  }
  return layers;
}

void objective_function::set_layer_pointers(std::vector<Layer*> layers) {
  auto it = layers.begin();
  for (objective_function_term *term : m_terms) {
    const size_t num_layers = term->get_layer_pointers().size();
    std::vector<Layer*> term_layers(it, it + num_layers);
    term->set_layer_pointers(term_layers);
    it += num_layers;
  }
  if (it != layers.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to set an invalid number of layer pointers "
        << "(expected " << it - layers.begin() << ", "
        << "found " << layers.size() << ")";
    throw lbann_exception(err.str());
  }
}

std::vector<weights*> objective_function::get_weights_pointers() const {
  std::vector<weights*> w;
  for (objective_function_term *term : m_terms) {
    std::vector<weights*> term_weights = term->get_weights_pointers();
    w.insert(w.end(), term_weights.begin(), term_weights.end());
  }
  return w;
}

void objective_function::set_weights_pointers(std::vector<weights*> w) {
  auto it = w.begin();
  for (objective_function_term *term : m_terms) {
    const size_t num_weights = term->get_weights_pointers().size();
    std::vector<weights*> term_weights(it, it + num_weights);
    term->set_weights_pointers(term_weights);
    it += num_weights;
  }
  if (it != w.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to set an invalid number of weights pointers "
        << "(expected " << it - w.begin() << ", "
        << "found " << w.size() << ")";
    throw lbann_exception(err.str());
  }
}

}  // namespace lbann
