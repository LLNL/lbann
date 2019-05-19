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

#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/profiling.hpp"
#include <numeric>

namespace lbann {

objective_function::objective_function(const objective_function& other)
  : m_statistics(other.m_statistics),
    m_evaluation_time(other.m_evaluation_time),
    m_differentiation_time(other.m_differentiation_time) {
  m_terms = other.m_terms;
  for (auto& term : m_terms) {
    term = term->copy();
  }
}

objective_function& objective_function::operator=(const objective_function& other) {
  for (const auto& term : m_terms) {
    if (term != nullptr) { delete term; }
  }
  m_terms = other.m_terms;
  for (auto& term : m_terms) {
    term = term->copy();
  }
  m_statistics = other.m_statistics;
  m_evaluation_time = other.m_evaluation_time;
  m_differentiation_time = other.m_differentiation_time;
  return *this;
}

objective_function::~objective_function() {
  for (const auto& term : m_terms) {
    if (term != nullptr) { delete term; }
  }
}

void objective_function::setup(model& m) {
  for (objective_function_term *term : m_terms) {
    if (term == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "a term in the objective function is a null pointer";
      throw lbann_exception(err.str());
    }
    term->setup(m);
  }
}

void objective_function::start_evaluation(execution_mode mode,
                                          int mini_batch_size) {
  const auto start_time = get_time();
  prof_region_begin("obj-start-eval", prof_colors[0], false);
  for (const auto& term : m_terms) {
    prof_region_begin(("obj-start-eval-" + term->name()).c_str(), prof_colors[1], false);
    term->start_evaluation();
    prof_region_end(("obj-start-eval-" + term->name()).c_str(), false);
  }
  prof_region_end("obj-start-eval", false);
  m_evaluation_time += get_time() - start_time;
}

EvalType objective_function::finish_evaluation(execution_mode mode,
                                               int mini_batch_size) {
  const auto start_time = get_time();
  EvalType value = EvalType(0);
  prof_region_begin("obj-finish-eval", prof_colors[0], false);
  for (const auto& term : m_terms) {
    prof_region_begin(("obj-finish-eval-" + term->name()).c_str(), prof_colors[1], false);
    value += term->finish_evaluation();
    prof_region_end(("obj-finish-eval-" + term->name()).c_str(), false);
  }
  prof_region_end("obj-finish-eval", false);
  m_statistics[mode].add_value(mini_batch_size * value,
                               mini_batch_size);
  m_evaluation_time += get_time() - start_time;
  return value;
}

void objective_function::differentiate() {
  const auto start_time = get_time();
  prof_region_begin("obj-differentiate", prof_colors[0], false);
  for (const auto& term : m_terms) {
    prof_region_begin(("obj-differentiate-" + term->name()).c_str(), prof_colors[1], false);
    term->differentiate();
    prof_region_end(("obj-differentiate-" + term->name()).c_str(), false);
  }
  prof_region_end("obj-differentiate", false);
  m_differentiation_time += get_time() - start_time;
}

void objective_function::compute_weight_regularization() {
  const auto start_time = get_time();
  prof_region_begin("obj-weight-regularization", prof_colors[0], false);
  for (const auto& term : m_terms) {
    prof_region_begin(("obj-weight-regularization-" + term->name()).c_str(), prof_colors[1], false);
    term->compute_weight_regularization();
    prof_region_end(("obj-weight-regularization-" + term->name()).c_str(), false);
  }
  prof_region_end("obj-weight-regularization", false);
  m_differentiation_time += get_time() - start_time;
}

EvalType objective_function::get_mean_value(execution_mode mode) const {
  if (m_statistics.count(mode) == 0
      || m_statistics.at(mode).get_num_samples() == 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to get mean objective function value with no samples for statistics";
    throw lbann_exception(err.str());
  }
  return m_statistics.at(mode).get_mean();
}

int objective_function::get_statistics_num_samples(execution_mode mode) const {
  if (m_statistics.count(mode) == 0) {
    return 0;
  } else {
    return m_statistics.at(mode).get_num_samples();
  }
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
