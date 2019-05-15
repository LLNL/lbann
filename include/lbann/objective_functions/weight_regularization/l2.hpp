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

#ifndef LBANN_OBJECTIVE_FUNCTIONS_WEIGHT_REGULARIZATION_L2_WEIGHT_REGULARIZATION_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTIONS_WEIGHT_REGULARIZATION_L2_WEIGHT_REGULARIZATION_HPP_INCLUDED

#include "lbann/objective_functions/objective_function_term.hpp"

namespace lbann {

/** @class l2_weight_regularization
 *  @brief Apply L2 regularization to a set of weights.
 *
 *  Given a weights tensor @f$ w @f$,
 *  @f[ L2(w) = \frac{1}{2} \sum\limits_{i} w(i)^2 @f]
 *  Note the @f$ 1/2 @f$ scaling factor.
 */
class l2_weight_regularization : public objective_function_term {
public:

  /** @param scale_factor   The objective function term is
   *                        @f$ \text{scale\_factor} \times \sum L2(w_i) @f$
   */
  l2_weight_regularization(EvalType scale_factor = 1);
  l2_weight_regularization* copy() const override { return new l2_weight_regularization(*this); }
  std::string name() const override { return "L2 weight regularization"; }
  void setup(model& m) override;
  void start_evaluation() override;
  EvalType finish_evaluation() override;

  /** Compute the gradient w.r.t. the activations.
   *
   *  L2 regularization is independent of forward prop output, so
   *  nothing needs to be done here.
   *
   *  @todo Come up with a better function name in the base class.
   */
  void differentiate() override {};

  /** Compute the gradient w.r.t. the weights.
   *
   *  @f[ \nabla_w L2(w) = w @f]
   */
  void compute_weight_regularization() override;

private:

  /** Contributions to evaluated value. */
  std::map<El::Device, CPUMat> m_contributions;

  /** For non-blocking allreduces. */
  Al::request m_allreduce_req;
#ifdef LBANN_HAS_GPU
  /** For non-blocking GPU-CPU memory copies. */
  cuda::event_wrapper m_copy_event;
#endif // LBANN_HAS_GPU

  /** Add the sum of squares of @c vals to @c contribution.
   *
   *  @param vals           The values to accumulate
   *  @param contribution   @f$ 1 \times 1 @f$ matrix. Used as an
   *                        accumulation variable.
   */
  template <El::Device Device>
  static void accumulate_contribution(const DMat<Device>& vals,
                                      DMat<Device>& contribution);

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTIONS_WEIGHT_REGULARIZATION_L2_WEIGHT_REGULARIZATION_HPP_INCLUDED
