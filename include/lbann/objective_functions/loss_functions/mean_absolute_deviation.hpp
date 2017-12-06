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

#ifndef LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_MEAN_ABSOLUTE_DEVIATION_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_MEAN_ABSOLUTE_DEVIATION_HPP_INCLUDED

#include "lbann/objective_functions/loss_functions/loss_function.hpp"

namespace lbann {

/** Mean absolution deviation loss function. */
class mean_absolute_deviation : public loss_function {
 public:
  /** Default constructor. */
  mean_absolute_deviation(DataType scale_factor = DataType(1)) 
    : loss_function(scale_factor) {}

  /** Copy constructor. */
  mean_absolute_deviation(const mean_absolute_deviation& other) = default;
  /** Copy assignment operator. */
  mean_absolute_deviation& operator=(const mean_absolute_deviation& other) = default;
  /** Destructor. */
  ~mean_absolute_deviation() override = default;
  /** Copy function. */
  mean_absolute_deviation* copy() const override { return new mean_absolute_deviation(*this); }

  /** Get the name of the objective function term. */
  std::string name() const override { return "mean_absolute_deviation"; }

  /** Compute the mean absolute deviation objective function.
   *  Given a prediction \f$\hat{y}\f$ and ground truth \f$y\f$, the
   *  mean absolute deviation is
   *    \f[
   *    MAD(\hat{y}, y) = \frac{1}{n} \lVert y - \hat{y} \rVert_1
   *    \f]
   *  This function updates the objective function value with the mean
   *  value of the mean absolute deviation across the mini-batch.
   */
  DataType evaluate(const AbsDistMat& prediction,
                    const AbsDistMat& ground_truth) override;

  /** Compute the mean absolution deviation gradient.
   *  The gradient is w.r.t. the prediction vector.
   */
  void differentiate(const AbsDistMat& prediction,
                     const AbsDistMat& ground_truth,
                     AbsDistMat& gradient) override;

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_MEAN_ABSOLUTE_DEVIATION_HPP_INCLUDED
