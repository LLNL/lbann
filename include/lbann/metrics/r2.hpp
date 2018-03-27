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

#ifndef LBANN_METRIC_R2_HPP
#define LBANN_METRIC_R2_HPP

#include "lbann/metrics/metric.hpp"

namespace lbann {

/** R^2 (coefficient of determination) regression metric.
 */
class r2_metric : public metric {

 public:

  /** Constructor. */
  r2_metric(lbann_comm *comm) : metric(comm) {}

  /** Copy constructor. */
  r2_metric(const r2_metric& other) = default;
  /** Copy assignment operator. */
  r2_metric& operator=(const r2_metric& other) = default;
  /** Destructor. */
  virtual ~r2_metric() = default;
  /** Copy function. */
  r2_metric* copy() const override {
    return new r2_metric(*this);
  }

  /** Return a string name for this metric. */
  std::string name() const override { return "r2"; }

 protected:

  /** Computation to evaluate the metric function.
   *  numerator = Sum(Square(ground_truth - prediction)
   *  denominator = Sum(Square(ground_truth - Mean(ground_truth)))
   *  return (1 - numerator/(denominator + epsilon))
   */
  EvalType evaluate_compute(const AbsDistMat& prediction,
                            const AbsDistMat& ground_truth) override;

};

}  // namespace lbann

#endif  // LBANN_METRIC_R2_HPP
