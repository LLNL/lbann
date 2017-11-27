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

#ifndef LBANN_OBJECTIVE_FUNCTION_POISSON_NEGLOGLIKE_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_POISSON_NEGLOGLIKE_HPP_INCLUDED

#include "lbann/objective_functions/objective_function.hpp"

namespace lbann {

namespace objective_functions {

/** Poisson negative log-likelihood objective function. */
class poisson_negloglike : public objective_function {

 public:
  /** Default constructor. */
  poisson_negloglike() = default;
  /** Copy constructor. */
  poisson_negloglike(const poisson_negloglike& other) = default;
  /** Copy assignment operator. */
  poisson_negloglike& operator=(const poisson_negloglike& other) = default;
  /** Destructor. */
  ~poisson_negloglike() = default;
  /** Copy function. */
  poisson_negloglike* copy() const { return new poisson_negloglike(*this); }

  /** Compute the Poisson negative log-likelihood objective function.
   *  Given a prediction \f$\hat{y}\f$ and ground truth \f$y\f$, the
   *  Poisson negative log-likelihood is
   *    \f[
   *    Pois_nll(\hat{y}, y) = \hat{y} - y\log(\hat{y}) + \log(\Gamma(y + 1))
   *    \f]
   *  This function updates the objective function value with the mean
   *  value of the Poisson negative log-likelihood across the mini-batch.
   */
  void compute_value(const AbsDistMat& predictions,
                     const AbsDistMat& ground_truth);

  /** Compute the gradient of the Poisson negative log-likelihood objective function.
   *  Given a prediction \f$\hat{y}\f$ and ground truth \f$y\f$, the
   *  gradient of the Poisson negative log-likelihood is
   *    \f[
   *    \nabla_y Pois_nll(\hat{y},y) = 1 - y/\hat{y}
   *    \f]
   */
  void compute_gradient(const AbsDistMat& predictions,
                        const AbsDistMat& ground_truth,
                        AbsDistMat& gradient);

  /** Get the name of the objective function. */
  std::string name() const { return "poisson negative log-likelihood"; }

};

}  // namespace objective_functions

}  // namespace lbann

#endif  // LBANN_OBJECTIVE_FUNCTION_POISSON_NEGLOGLIKE_HPP_INCLUDED
