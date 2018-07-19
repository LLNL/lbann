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

#ifndef LBANN_METRIC_BOOLEAN_FALSE_POSITIVES_HPP
#define LBANN_METRIC_BOOLEAN_FALSE_POSITIVES_HPP

#include "lbann/metrics/metric.hpp"

namespace lbann {

/** Boolean false positives metric.
 * This thresholds predictions and ground truth to boolean values using 0.5
 * then checks for false positives (i.e. prediction is true when ground truth
 * is false). The reported false-positive percent is with respect to all values
 * and not the errors.
 */
class boolean_false_positives_metric : public metric {

 public:
  boolean_false_positives_metric(lbann_comm *comm) : metric(comm) {}
  boolean_false_positives_metric(
    const boolean_false_positives_metric& other) = default;
  boolean_false_positives_metric& operator=(
    const boolean_false_positives_metric& other) = default;
  virtual ~boolean_false_positives_metric() = default;
  boolean_false_positives_metric* copy() const override {
    return new boolean_false_positives_metric(*this);
  }

  /** Return a string name for this metric. */
  std::string name() const override { return "boolean false positives"; }
  std::string get_unit() const override { return "%"; }

 protected:
  /** Computation to evaluate the metric function.
   *  This returns the sum of metric values across the mini-batch, not
   *  the mean value.
   */
  EvalType evaluate_compute(const AbsDistMat& prediction,
                            const AbsDistMat& ground_truth) override;

};

}  // namespace lbann

#endif  // LBANN_METRIC_BOOLEAN_FALSE_POSITIVES_HPP
