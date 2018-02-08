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

#ifndef LBANN_OBJECTIVE_FUNCTION_WEIGHT_REGULARIZATION_L2_WEIGHT_REGULARIZATION_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_WEIGHT_REGULARIZATION_L2_WEIGHT_REGULARIZATION_HPP_INCLUDED

#include "lbann/objective_functions/objective_function_term.hpp"

namespace lbann {

/** Objective function term for L2 weight regularization.
 *  Given weights \f$w_1,\cdots,w_n\f$, the L2 weight regularization
 *  term is
 *    \f[
 *    L2(w) = \frac{1}{2} \sum\limits_{i} w_i
 *    \f]
 *  Note the \f$1/2\f$ scaling factor.
 */
class l2_weight_regularization : public objective_function_term {
 public:

  l2_weight_regularization(EvalType scale_factor = EvalType(1))
    : objective_function_term(scale_factor) {}
  l2_weight_regularization* copy() const override { return new l2_weight_regularization(*this); }
  std::string name() const override { return "L2 weight regularization"; }
  void setup(model& m) override;
  EvalType evaluate() override;

  /** Compute the gradient w.r.t. the activations. 
   *  L2 weight regularization is independent of the activations.
   */
  void differentiate() override {};

  /** Compute the gradient w.r.t. the weights.
   *  The gradient w.r.t. the weights is
   *    \f[
   *    \nabla_w L2(w) = w
   *    \f]
   */
  void compute_weight_regularization() override;

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_WEIGHT_REGULARIZATION_L2_WEIGHT_REGULARIZATION_HPP_INCLUDED
