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

#ifndef LBANN_CALLBACKS_CALLBACK_CHECK_GRADIENTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECK_GRADIENTS_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/** Gradient checking callback.
 *  Gradient checking is performed at the beginning of the test
 *  phase. Using a fourth-order finite difference scheme, a numerical
 *  partial derivative is computed for every weight parameter. If the
 *  numerical derivative differs signifcantly from the analytical
 *  derivative computed during backprop, the gradient check has
 *  failed.
 */
class lbann_callback_check_gradients : public lbann_callback {
public:

  /** Constructor.
   *  @param step_size          Step size for numerical
   *                            differentiation (with a step size of
   *                            zero, the step size is chosen to
   *                            minimize the numerical error).
   *  @param verbose            Whether to print results for each
   *                            parameter.
   *  @param error_on_failure   Whether to throw an exception for
   *                            large gradient errors.
   */
  lbann_callback_check_gradients(DataType step_size = DataType(0),
                                 bool verbose = false,
                                 bool error_on_failure = false);
  lbann_callback_check_gradients* copy() const override {
    return new lbann_callback_check_gradients(*this);
  }
  void on_test_begin(model *m) override;
  std::string name() const override { return "check gradients"; }

  /** Compute objective function value.
   *  It is assumed that input data has already been loaded into the
   *  activations of the first layer.
   */
  DataType compute_objective_function(model *m);

private:

  /** Step size for numerical differentiation. */
  DataType m_step_size;
  /** Whether to print results for each parameter. */
  bool m_verbose;
  /** Whether to throw an exception for large gradient errors. */
  bool m_error_on_failure;

};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CHECK_GRADIENTS_HPP_INCLUDED
