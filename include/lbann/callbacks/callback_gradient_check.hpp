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
//
// lbann_callback_gradient_check .hpp .cpp - Callback hooks for gradient check
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_GRADIENT_CHECK_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_GRADIENT_CHECK_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/** Callback hooks for gradient check. */
class lbann_callback_gradient_check : public lbann_callback {
 public:
  
  /** Constructor.
   *  @param step_size  Step size for numerical differentiation.
   *  @param verbose    Whether to print results for each parameter.
   */
  lbann_callback_gradient_check(DataType step_size = DataType(0),
                                bool verbose = false,
                                bool fail_on_error = false);

  lbann_callback_gradient_check(const lbann_callback_gradient_check&) = default;
  lbann_callback_gradient_check& operator=(const lbann_callback_gradient_check&) = default;
  lbann_callback_gradient_check* copy() const { return new lbann_callback_gradient_check(*this); }
  void on_test_begin(model *m);
  std::string name() const { return "gradient check"; }

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
  bool m_fail_on_error;

};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_GRADIENT_CHECK_HPP_INCLUDED
