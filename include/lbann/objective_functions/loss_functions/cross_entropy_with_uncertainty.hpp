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

#ifndef LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_CROSS_ENTROPY_WITH_UNCERTAINTY_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_CROSS_ENTROPY_WITH_UNCERTAINTY_HPP_INCLUDED

#include "lbann/objective_functions/loss_functions/loss_function.hpp"

namespace lbann {

/** Cross entropy with uncertainty loss function. */
class cross_entropy_with_uncertainty : public loss_function {
 public:
  /** Default constructor. */
  cross_entropy_with_uncertainty(DataType scale_factor = DataType(1));

  /** Copy constructor. */
  cross_entropy_with_uncertainty(const cross_entropy_with_uncertainty& other);
  /** Copy assignment operator. */
  cross_entropy_with_uncertainty& operator=(const cross_entropy_with_uncertainty& other);
  /** Destructor. */
  ~cross_entropy_with_uncertainty() override;
  /** Copy function. */
  cross_entropy_with_uncertainty* copy() const override {
    return new cross_entropy_with_uncertainty(*this);
  }

  /** Get the name of the objective function term. */
  std::string name() const override { return "cross_entropy_with_uncertainty"; }

  /** Setup cross entropy term. */
  void setup(objective_function& obj_fn) override;

  /** Compute the cross entropy objective function.
   *  Given a predicted distribution \f$y\f$ and ground truth
   *  distribution \f$\hat{y}\f$, the cross entropy is
   *    \f[
   *    CE(y,\hat{y}) = - \sum\limits_{i} \hat{y}_i \log y_i
   *    \f]
   *  This function updates the objective function value with the mean
   *  cross entropy across the mini-batch. Note that each column of
   *  the predictions matrix should have non-negative entries that add up 
   *  to one.
   */
  DataType evaluate(const AbsDistMat& prediction,
                    const AbsDistMat& ground_truth) override;

  /** Compute the gradient of the cross entropy objective function.
   *  Given a predicted distribution \f$y\f$ and ground truth
   *  distribution \f$\hat{y}\f$, the gradient of the cross entropy
   *  is
   *    \f[
   *    \nabla_y CE (y,\hat{y}) = - \hat{y} . / y
   *    \f]
   */
  void differentiate(const AbsDistMat& prediction,
                     const AbsDistMat& ground_truth,
                     AbsDistMat& gradient) override;

 private:
  /** Workspace. */
  AbsDistMat* m_prediction_sums;

};

}  // namespace lbann

#endif  // LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_CROSS_ENTROPY_WITH_UNCERTAINTY_HPP_INCLUDED
