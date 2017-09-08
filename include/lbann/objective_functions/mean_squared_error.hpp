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

#ifndef LBANN_OBJECTIVE_FUNCTION_MEAN_SQUARED_ERROR_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_MEAN_SQUARED_ERROR_HPP_INCLUDED

#include "lbann/objective_functions/objective_function.hpp"

namespace lbann {

namespace objective_functions {

/** Mean squared error objective function. */
class mean_squared_error : public objective_function {

 public:
  /** Default constructor. */
  mean_squared_error() = default;
  /** Copy constructor. */
  mean_squared_error(const mean_squared_error& other) = default;
  /** Copy assignment operator. */
  mean_squared_error& operator=(const mean_squared_error& other) = default;
  /** Destructor. */
  ~mean_squared_error() = default;
  /** Copy function. */
  mean_squared_error* copy() const { return new mean_squared_error(*this); }

  /** Compute the mean squared error objective function.
   *  Given a prediction \f$y\f$ and ground truth \f$\hat{y}\f$, the
   *  mean squared error is
   *    \f[
   *    MSE(y,\hat{y}) = \frac{1}{n} \lVert y - \hat{y} \rVert^2_2
   *    \f]
   *  This function updates the objective function value with the mean
   *  value of the mean squared error across the mini-batch.
   */
  void compute_value(const AbsDistMat& predictions,
                     const AbsDistMat& ground_truth);

  /** Compute the gradient of the mean squared error objective function.
   *  Given a prediction \f$y\f$ and ground truth \f$\hat{y}\f$, the
   *  gradient of the mean squared error is
   *    \f[
   *    \nabla_y MSE (y,\hat{y}) = \frac{2}{n} (y - \hat{y})
   *    \f]
   */
  void compute_gradient(const AbsDistMat& predictions,
                        const AbsDistMat& ground_truth,
                        AbsDistMat& gradient);

  /** Get the name of the objective function. */
  std::string name() const { return "mean squared error"; }

};

}  // namespace objective_functions

}  // namespace lbann

#endif  // LBANN_OBJECTIVE_FUNCTION_MEAN_SQUARED_ERROR_HPP_INCLUDED
