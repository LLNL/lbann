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

#ifndef LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_MEAN_SQUARED_ERROR_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_MEAN_SQUARED_ERROR_HPP_INCLUDED

#include "lbann/objective_functions/loss_functions/loss_function.hpp"

namespace lbann {

/** Mean squared error loss function.
 *  Not to be confused with mean_squared_error_metric.
 */
class mean_squared_error_loss : public loss_function {
 public:
  /** Default constructor. */
  mean_squared_error_loss(EvalType scale_factor = EvalType(1)) 
    : loss_function(scale_factor) {}

  /** Copy constructor. */
  mean_squared_error_loss(const mean_squared_error_loss& other) = default;
  /** Copy assignment operator. */
  mean_squared_error_loss& operator=(const mean_squared_error_loss& other) = default;
  /** Destructor. */
  ~mean_squared_error_loss() override = default;
  /** Copy function. */
  mean_squared_error_loss* copy() const override {
    return new mean_squared_error_loss(*this);
  }

  /** Get the name of the objective function term. */
  std::string name() const override { return "mean squared error"; }

  /** Compute the mean squared error objective function.
   *  Given a prediction \f$\hat{y}\f$ and ground truth \f$y\f$, the
   *  mean squared error is
   *    \f[
   *    MSE(\hat{y}, y) = \frac{1}{n} \lVert y - \hat{y} \rVert_2^2
   *    \f]
   *  This function updates the objective function value with the mean
   *  value of the mean absolute deviation across the mini-batch.
   */
  void start_evaluate_compute(const AbsDistMat& prediction,
                              const AbsDistMat& ground_truth) override;

  EvalType finish_evaluate_compute(const AbsDistMat& prediction,
                                   const AbsDistMat& ground_truth) override;

  /** Compute the gradient of the mean squared error objective function.
   *  Given a prediction \f$\hat{y}\f$ and ground truth \f$y\f$, the
   *  gradient of the mean squared error is
   *    \f[
   *    \nabla_y MSE (y,\hat{y}) = \frac{2}{n} (y - \hat{y})
   *    \f]
   */
  void differentiate_compute(const AbsDistMat& prediction,
                             const AbsDistMat& ground_truth,
                             AbsDistMat& gradient) override;

 private:
  /** Sum of the the squared errors. */
  EvalType m_sum;
  /** Non-blocking allreduce request. */
  Al::request m_allreduce_req;
};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_MEAN_SQUARED_ERROR_HPP_INCLUDED
