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

#ifndef LBANN_OBJECTIVE_FUNCTION_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_HPP_INCLUDED

#include "lbann/objective_functions/objective_function_term.hpp"
#include "lbann/metrics/metric.hpp"

namespace lbann {

/** Objective function class. */
class objective_function {
 public:

  /** Default constructor. */
  objective_function() {}

  /** Copy constructor. */
  objective_function(const objective_function& other);
  /** Copy assignment operator. */
  objective_function& operator=(const objective_function& other);
  /** Destructor. */
  ~objective_function();
  /** Copy function. */
  objective_function* copy() const { return new objective_function(*this); }

  /** Add a term to the objective function.
   *  The objective function takes ownership of the objective function
   *  term and deallocates it during destruction.
   */
  void add_term(objective_function_term* term) { m_terms.push_back(term); }
  /** Get list of objective function terms. */
  std::vector<objective_function_term*> get_terms() { return m_terms; }

  /** Setup objective function. */
  void setup(model& m);

  /** Start evaluating the objective function.
   *  This function takes the model's current mini-batch size. If
   *  multiple models are being trained, the current mini-batch size
   *  may be different from the effective mini-batch size.
   *  The result is not guaranteed to be available until finish_evaluation is
   *  called.
   */
  void start_evaluation(execution_mode mode, int mini_batch_size);

  /** Complete evaluation of the objective function.
   *  The result is stored in history.
   */
  EvalType finish_evaluation(execution_mode mode, int mini_batch_size);

  /** Compute the objective function gradient.
   *  The gradient is with respect to the objective function inputs
   */
  void differentiate();

  /** Compute the gradient of the weight regularization term.
   *  The gradient is computed w.r.t. the weights.
   */
  void compute_weight_regularization();

  /** Clear all statistics. */
  void reset_statistics() { m_statistics.clear(); }
  /** Clear statistics for an execution mode. */
  void reset_statistics(execution_mode mode) { m_statistics.erase(mode); }

  /** Get mean objective function value.
   *  This is a weighted average such that each mini-batch sample makes
   *  an equal contribution.
   */
  EvalType get_mean_value(execution_mode mode) const;
  /** Get number of samples for statistics. */
  int get_statistics_num_samples(execution_mode mode) const;

  /** Get list of pointers to layers. */
  std::vector<Layer*> get_layer_pointers() const;
  /** Set list of pointers to layers. */
  void set_layer_pointers(std::vector<Layer*> layers);
  /** Get list of pointers to weights. */
  std::vector<weights*> get_weights_pointers() const;
  /** Set list of pointers to weights. */
  void set_weights_pointers(std::vector<weights*> w);

  /** Get the time spent evaluating the objective function. */
  EvalType get_evaluation_time() const { return m_evaluation_time; }
  /** Get the time spent computing the objective function gradient. */
  EvalType get_differentiation_time() const { return m_differentiation_time; }
  /** Reset time counters. */
  void reset_counters() {
    m_evaluation_time = 0.0;
    m_differentiation_time = 0.0;
  }

 private:

  /** List of objective function terms. */
  std::vector<objective_function_term*> m_terms;

  /** Objective funciton statistics. */
  std::map<execution_mode,metric_statistics> m_statistics;

  /** Time spent evaluating the objective function. */
  EvalType m_evaluation_time = EvalType(0);
  /** Time spent computing the objective function gradient. */
  EvalType m_differentiation_time = EvalType(0);

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_INCLUDED
