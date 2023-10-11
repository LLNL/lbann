////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"
#include <numeric>

namespace lbann {

objective_function::objective_function(const objective_function& other)
  : m_statistics(other.m_statistics),
    m_evaluation_time(other.m_evaluation_time),
    m_differentiation_time(other.m_differentiation_time)
{
  for (const auto& ptr : other.m_terms) {
    m_terms.emplace_back(ptr ? ptr->copy() : nullptr);
  }
}

objective_function&
objective_function::operator=(const objective_function& other)
{
  m_terms.clear();
  for (const auto& ptr : other.m_terms) {
    m_terms.emplace_back(ptr ? ptr->copy() : nullptr);
  }
  m_statistics = other.m_statistics;
  m_evaluation_time = other.m_evaluation_time;
  m_differentiation_time = other.m_differentiation_time;
  return *this;
}

template <class Archive>
void objective_function::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_statistics), CEREAL_NVP(m_terms));
}

void objective_function::add_term(std::unique_ptr<objective_function_term> term)
{
  m_terms.push_back(std::move(term));
}

std::vector<objective_function_term*> objective_function::get_terms()
{
  std::vector<objective_function_term*> ptrs;
  for (const auto& ptr : m_terms) {
    ptrs.push_back(ptr.get());
  }
  return ptrs;
}

void objective_function::setup(model& m)
{
  for (auto&& term : m_terms) {
    if (term == nullptr) {
      LBANN_ERROR("a term in the objective function is a null pointer");
    }
    term->setup(m);
  }
}

void objective_function::start_evaluation(execution_mode mode,
                                          int mini_batch_size)
{
  const auto start_time = get_time();
  prof_region_begin("obj-start-eval", prof_colors[0], false);
  for (auto&& term : m_terms) {
    prof_region_begin(("obj-start-eval-" + term->name()).c_str(),
                      prof_colors[1],
                      false);
    term->start_evaluation();
    prof_region_end(("obj-start-eval-" + term->name()).c_str(), false);
  }
  prof_region_end("obj-start-eval", false);
  m_evaluation_time += get_time() - start_time;
}

EvalType objective_function::finish_evaluation(execution_mode mode,
                                               int mini_batch_size)
{
  const auto start_time = get_time();
  EvalType value = EvalType(0);
  prof_region_begin("obj-finish-eval", prof_colors[0], false);
  for (auto&& term : m_terms) {
    prof_region_begin(("obj-finish-eval-" + term->name()).c_str(),
                      prof_colors[1],
                      false);
    value += term->finish_evaluation();
    prof_region_end(("obj-finish-eval-" + term->name()).c_str(), false);
  }
  prof_region_end("obj-finish-eval", false);
  m_statistics[mode].add_value(mini_batch_size * value, mini_batch_size);
  m_evaluation_time += get_time() - start_time;
  return value;
}

void objective_function::differentiate()
{
  const auto start_time = get_time();
  prof_region_begin("obj-differentiate", prof_colors[0], false);
  for (auto&& term : m_terms) {
    prof_region_begin(("obj-differentiate-" + term->name()).c_str(),
                      prof_colors[1],
                      false);
    term->differentiate();
    prof_region_end(("obj-differentiate-" + term->name()).c_str(), false);
  }
  prof_region_end("obj-differentiate", false);
  m_differentiation_time += get_time() - start_time;
}

void objective_function::compute_weight_regularization()
{
  const auto start_time = get_time();
  prof_region_begin("obj-weight-regularization", prof_colors[0], false);
  for (auto&& term : m_terms) {
    prof_region_begin(("obj-weight-regularization-" + term->name()).c_str(),
                      prof_colors[1],
                      false);
    term->compute_weight_regularization();
    prof_region_end(("obj-weight-regularization-" + term->name()).c_str(),
                    false);
  }
  prof_region_end("obj-weight-regularization", false);
  m_differentiation_time += get_time() - start_time;
}

EvalType objective_function::get_mean_value(execution_mode mode) const
{
  if (m_statistics.count(mode) == 0 ||
      m_statistics.at(mode).get_num_samples() == 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to get mean objective function value with no samples for "
           "statistics";
    throw lbann_exception(err.str());
  }
  return m_statistics.at(mode).get_mean();
}

int objective_function::get_statistics_num_samples(execution_mode mode) const
{
  if (m_statistics.count(mode) == 0) {
    return 0;
  }
  else {
    return m_statistics.at(mode).get_num_samples();
  }
}

std::vector<ViewingLayerPtr> objective_function::get_layer_pointers() const
{
  std::vector<ViewingLayerPtr> layers;
  for (const auto& term : m_terms) {
    auto term_layers = term->get_layer_pointers();
    layers.insert(layers.end(), term_layers.begin(), term_layers.end());
  }
  return layers;
}

void objective_function::set_layer_pointers(std::vector<ViewingLayerPtr> layers)
{
  auto it = layers.begin();
  for (auto&& term : m_terms) {
    const size_t num_layers = term->get_layer_pointers().size();
    std::vector<ViewingLayerPtr> term_layers(it, it + num_layers);
    term->set_layer_pointers(term_layers);
    it += num_layers;
  }
  if (it != layers.end()) {
    LBANN_ERROR("attempted to set an invalid number of layer pointers ",
                "(expected ",
                it - layers.begin(),
                ", found ",
                layers.size(),
                ")");
  }
}

std::vector<ViewingWeightsPtr> objective_function::get_weights_pointers() const
{
  std::vector<ViewingWeightsPtr> w;
  for (const auto& term : m_terms) {
    auto term_weights = term->get_weights_pointers();
    w.insert(w.end(), term_weights.begin(), term_weights.end());
  }
  return w;
}

void objective_function::set_weights_pointers(std::vector<ViewingWeightsPtr> w)
{
  auto it = w.begin();
  for (auto&& term : m_terms) {
    const size_t num_weights = term->get_weights_pointers().size();
    std::vector<ViewingWeightsPtr> term_weights(it, it + num_weights);
    term->set_weights_pointers(term_weights);
    it += num_weights;
  }
  if (it != w.end()) {
    LBANN_ERROR("attempted to set an invalid number of weights pointers ",
                "(expected ",
                it - w.begin(),
                ", found ",
                w.size(),
                ")");
  }
}

void objective_function::set_amp_scale(EvalType scale)
{
  for (auto&& term : m_terms) {
    term->set_amp_scale(scale);
  }
}

void objective_function::write_proto(lbann_data::ObjectiveFunction& proto) const
{
  for (auto const& term : m_terms)
    term->write_specific_proto(proto);
}

} // namespace lbann

#define LBANN_SKIP_CEREAL_REGISTRATION
#define LBANN_CLASS_NAME objective_function
#include <lbann/macros/register_class_with_cereal.hpp>
