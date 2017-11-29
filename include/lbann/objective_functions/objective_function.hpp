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

#ifndef LBANN_OBJECTIVE_FUNCTION_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_HPP_INCLUDED

#include "lbann/objective_functions/objective_function_term.hpp"

namespace lbann {

/** Objective function class. */
class objective_function {
 public:

  /** Default constructor. */
  objective_function();

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
  
  /** Setup objective function. */
  void setup(model& m);
  
  /** Compute the objective function value.
   *  The result is stored in history.
   */
  DataType compute_value();
  
  /** Compute the objective function gradient.
   *  The gradient is with respect to the objective function inputs
   */
  void compute_gradient();

  /** Get history of objective function values. */
  std::vector<DataType> get_history() const { return m_history; }
  /** Clear history of objective function values. */
  void clear_history() { m_history.clear(); }

  /** Get mean objective function value in history. */
  DataType get_history_mean_value() const;

  /** Get model that owns this objective function. */
  model* get_model() { return m_model; }
  /** Set model that owns this objective function. */
  void set_model(model* m) { m_model = m; }

  /** Get list of pointers to layers. */
  std::vector<Layer*> get_layer_pointers();
  /** Set list of pointers to layers. */
  void set_layer_pointers(std::vector<Layer*> layers);
  /** Get list of pointers to weights. */
  std::vector<weights*> get_weights_pointers();
  /** Set list of pointers to weights. */
  void set_weights_pointers(std::vector<weights*> w);

  /** Get the time spent computing the value. */
  double get_value_time() const { return m_value_time; }
  /** Get the itme spent computing the gradient. */
  double get_gradient_time() const { return m_gradient_time; }
  /** Reset time counters. */
  void reset_counters() {
    m_value_time = 0.0;
    m_gradient_time = 0.0;
  }

 private:

  /** Pointer to model that owns this objective function. */
  model* m_model;

  /** List of objective function terms. */
  std::vector<objective_function_term*> m_terms;

  /** History of objective function values. */
  std::vector<DataType> m_history;

  /** Time spent computing the value. */
  double m_value_time = 0.0;
  /** Time spent computing the gradient. */
  double m_gradient_time = 0.0;

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_INCLUDED
