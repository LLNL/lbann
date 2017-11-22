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

#ifndef LBANN_OBJECTIVE_FUNCTION_CROSS_ENTROPY_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_CROSS_ENTROPY_HPP_INCLUDED

#include "lbann/objective_functions/loss_functions/loss_function.hpp"

namespace lbann {

/** Cross entropy loss function. */
class cross_entropy : public loss_function {
 public:
  /** Default constructor. */
  cross_entropy(DataType scale_factor = DataType(1)) 
    : loss_function(scale_factor) {}

  /** Copy constructor. */
  cross_entropy(const cross_entropy& other) = default;
  /** Copy assignment operator. */
  cross_entropy& operator=(const cross_entropy& other) = default;
  /** Destructor. */
  ~cross_entropy() override = default;
  /** Copy function. */
  cross_entropy* copy() const override { return new cross_entropy(*this); }

  /** Get the name of the objective function term. */
  std::string name() const override { return "cross_entropy"; }

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

#endif // LBANN_OBJECTIVE_FUNCTION_CROSS_ENTROPY_HPP_INCLUDED
