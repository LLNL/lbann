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

#ifndef LBANN_OBJECTIVE_FUNCTION_MEAN_SQUARED_ERROR_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_MEAN_SQUARED_ERROR_HPP_INCLUDED

#include "lbann/objective_functions/loss_functions/loss_function.hpp"

namespace lbann {

/** Mean squared error loss function. */
class mean_squared_error : public loss_function {
 public:
  /** Default constructor. */
  mean_squared_error(DataType scale_factor = DataType(1)) 
    : loss_function(scale_factor) {}

  /** Copy constructor. */
  mean_squared_error(const mean_squared_error& other) = default;
  /** Copy assignment operator. */
  mean_squared_error& operator=(const mean_squared_error& other) = default;
  /** Destructor. */
  ~mean_squared_error() override = default;
  /** Copy function. */
  mean_squared_error* copy() const override { return new mean_squared_error(*this); }

  /** Get the name of the objective function term. */
  std::string name() const override { return "mean_squared_error"; }

  /** Evaluate the cross entropy loss function. */
  DataType evaluate(const AbsDistMat& prediction,
                    const AbsDistMat& ground_truth) override;

  /** Compute the cross entropy gradient.
   *  The gradient is w.r.t. the prediction vector.
   */
  void differentiate(const AbsDistMat& prediction,
                     const AbsDistMat& ground_truth,
                     AbsDistMat& gradient) override;

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_MEAN_SQUARED_ERROR_HPP_INCLUDED
