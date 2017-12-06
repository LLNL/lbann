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

#ifndef LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_POLYA_NEGLOGLIKE_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_POLYA_NEGLOGLIKE_HPP_INCLUDED

#include "lbann/objective_functions/loss_functions/loss_function.hpp"

namespace lbann {

/** Polya negative log-likelihood objective function. */
class polya_negloglike : public loss_function {
 public:
  /** Default constructor. */
  polya_negloglike(DataType scale_factor = DataType(1));

  /** Copy constructor. */
  polya_negloglike(const polya_negloglike& other);
  /** Copy assignment operator. */
  polya_negloglike& operator=(const polya_negloglike& other);
  /** Destructor. */
  ~polya_negloglike() override;
  /** Copy function. */
  polya_negloglike* copy() const override {
    return new polya_negloglike(*this);
  }

  /** Get the name of the objective function term. */
  std::string name() const override { return "polya_negloglike"; }

  /** Setup term. */
  void setup(objective_function& obj_fn) override;

  /** Compute the Polya negative log-likelihood objective function.
   *  Given the parameters \f$(\alpha_1,\dots,\alpha_K)\f$ of the predicted Polya distribution, the ground truth
   *  label counts \f$(k_1,\dots,k_K)\f$, and total counts \f$n = \sum_{i=0}^K k_i\f$ the Polya negative log-likelihood is 
   *    \f[
   *    Poly_nll(\vec{\alpha}, \vec{k}) = -\log \Gamma(\sum_{i=0}^K \hat{y}_i) + \log \Gamma(n + \sum_{i=0}^K \hat{y}_i) - \sum_{i=0}^K \left\{\log \Gamma(\hat{y}_i + \alpha_i) - \log \Gamma(\alpha_i) \right\}
   *    \f]
   *  This function updates the objective function value with the mean
   *  Polya negative log-likelihood across the mini-batch. Note that each column of
   *  the predictions and ground truth matrices should have
   *  non-negative entries.
   */
  DataType evaluate(const AbsDistMat& prediction,
                    const AbsDistMat& ground_truth) override;

  /** Compute the gradient of the Polya negative log-likelihood objective function.
   *  Given the parameters \f$(\alpha_1,\dots,\alpha_K)\f$ of the predicted Polya distribution, the ground 
   *  truth label counts \f$(k_1,\dots,k_K)\f$, and total counts \f$n = \sum_{i=0}^K k_i\f$,
   *  the gradient of the Polya negative log-likelihood function
   *  is
   *    \f[
   *    \nabla_{alpha_i} Poly_nll(\vec{\alpha}, \vec{k}) = -\Psi(\sum_{i=1}^K \alpha_i) + \Psi(n + \sum_{i=1}^K \alpha_i) - \Psi(k_i + \alpha_i) + \Psi(\alpha_i)
   *    \f]
   */
  void differentiate(const AbsDistMat& prediction,
                     const AbsDistMat& ground_truth,
                     AbsDistMat& gradient) override;

 private:
  AbsDistMat* m_counts;
  AbsDistMat* m_alpha_sums;
  AbsDistMat* m_lgamma_alpha_sums;
  AbsDistMat* m_lgamma_alpha_level_count_sums;

};

}  // namespace lbann

#endif  // LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_POLYA_NEGLOGLIKE_HPP_INCLUDED
