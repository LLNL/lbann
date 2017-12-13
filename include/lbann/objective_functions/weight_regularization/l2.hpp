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

/** Objective function term for L2 weight regularization. */
class l2_weight_regularization : public objective_function_term {
 public:
  /** Default constructor. */
  l2_weight_regularization(DataType scale_factor = DataType(1))
    : objective_function_term(scale_factor) {}

  /** Copy constructor. */
  l2_weight_regularization(const l2_weight_regularization& other) = default;
  /** Copy assignment operator. */
  l2_weight_regularization& operator=(const l2_weight_regularization& other) = default;
  /** Destructor. */
  ~l2_weight_regularization() override = default;
  /** Copy function. */
  l2_weight_regularization* copy() const override { return new l2_weight_regularization(*this); }

  /** Get the name of the objective function term. */
  std::string name() const override { return "l2_weight_regularization"; }

  /** Setup L2 regularization term. */
  void setup(model& m) override;
  
  /** Get the value of the L2 regularization term. */
  DataType evaluate() override;

  /** Compute the gradient of the L2 regularization term.
   *  The gradient is computed w.r.t. the weights.
   */
  void differentiate() override;

 private:

  /** Compute the squared L2 norm of mat. */
  DataType local_squared_l2_norm(const Mat& mat) const;

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_WEIGHT_REGULARIZATION_L2_WEIGHT_REGULARIZATION_HPP_INCLUDED
