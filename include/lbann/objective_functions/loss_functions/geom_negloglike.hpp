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

#ifndef LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_GEOM_NEGLOGLIKE_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_GEOM_NEGLOGLIKE_HPP_INCLUDED

#include "lbann/objective_functions/loss_functions/loss_function.hpp"

namespace lbann {

/** Geometric negative log-likelihood objective function. */
class geom_negloglike : public loss_function {
 public:
  /** Default constructor. */
  geom_negloglike(DataType scale_factor = DataType(1)) 
    : loss_function(scale_factor) {}

  /** Copy constructor. */
  geom_negloglike(const geom_negloglike& other) = default;
  /** Copy assignment operator. */
  geom_negloglike& operator=(const geom_negloglike& other) = default;
  /** Destructor. */
  ~geom_negloglike() override = default;
  /** Copy function. */
  geom_negloglike* copy() const override { return new geom_negloglike(*this); }

  /** Get the name of the objective function term. */
  std::string name() const override { return "geometric negative log-likelihood"; }

  /** Compute the Geometric negative log-likelihood objective function.
   *  Given a prediction \f$\hat{y}\f$ and ground truth \f$y\f$, the
   *  Geometric negative log-likelihood is
   *    \f[
   *    Geom_nll(\hat{y},y) = -y \log(1-\hat{y}) + \log(\hat{y})  
   *    \f]
   *  This function updates the objective function value with the mean
   *  value of the mean squared error across the mini-batch.
   */
  DataType evaluate_compute(const AbsDistMat& prediction,
                            const AbsDistMat& ground_truth) override;

  /** Compute the gradient of the Geometric negative log-likelihood objective function.
   *  Given a prediction \f$y\f$ and ground truth \f$\hat{y}\f$, the
   *  gradient of the Geometric negative log-likelihood is
   *    \f[
   *    \nabla_{\hat{y}} \text{Geom}_{\text{nll}} (\hat{y},y) = y/(1-\hat{y}) + 1/\hat{y}
   *    \f]
   */
  void differentiate_compute(const AbsDistMat& prediction,
                             const AbsDistMat& ground_truth,
                             AbsDistMat& gradient) override;

};

}  // namespace lbann

#endif  // LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_GEOM_NEGLOGLIKE_HPP_INCLUDED
