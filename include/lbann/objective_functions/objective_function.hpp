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

#include "lbann/base.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

namespace objective_functions {

/** Abstract class for objective functions. */
class objective_function {
 public:
  /** Default constructor. */
  objective_function();
  /** Copy constructor. */
  objective_function(const objective_function& other) = default;
  /** Copy assignment operator. */
  objective_function& operator=(const objective_function& other) = default;
  /** Destructor. */
  virtual ~objective_function() {}
  /** Copy function. */
  virtual objective_function* copy() const = 0;

  virtual void setup(const Layer& prev_layer) {}
  
  /** Compute the objective function value. */
  virtual void compute_value(const AbsDistMat& predictions,
                             const AbsDistMat& ground_truth) = 0;
  
  /** Compute the objective function gradient.
   *  The gradient is with respect to the predictions.
   */
  virtual void compute_gradient(const AbsDistMat& predictions,
                                const AbsDistMat& ground_truth,
                                AbsDistMat& gradient) = 0;
  
  /** Add to objective function value.
   *  Primarily used to add regularization terms.
   */
  void add_to_value(double value);

  /** Record and reset objective function value.
   *  Recorded values are used to compute statistics.
   */
  void record_and_reset_value();

  /** Get objective function value. */
  double get_value() const;

  /** Get mean objective function value.
   *  The mean is out of recorded objective function values.
   */ 
  double get_mean_value() const;
  
  /** Reset objective function statistics.
   *  The objective function value is also reset.
   */
  void reset_statistics();

  /** Get the name of the objective function. */
  virtual std::string name() const = 0;

 protected:

  /** Objective function value. */
  double m_value;
  /** Sum of recorded objective function values. */
  double m_recorded_values;
  /** Number of recorded objective function values. */
  int m_recorded_iterations;

};

}  // namespace objective_functions

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_INCLUDED
