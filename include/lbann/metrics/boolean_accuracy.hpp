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

#ifndef LBANN_METRIC_BOOLEAN_ACCURACY_HPP
#define LBANN_METRIC_BOOLEAN_ACCURACY_HPP

#include "lbann/metrics/metric.hpp"

namespace lbann {

/** Boolean accuracy metric.
 * This thresholds predictions and ground truth to boolean values using 0.5
 * then checks that they are equal.
 */
class boolean_accuracy_metric : public metric {

 public:
  boolean_accuracy_metric(lbann_comm *comm) : metric(comm) {}
  boolean_accuracy_metric(const boolean_accuracy_metric& other) = default;
  boolean_accuracy_metric& operator=(const boolean_accuracy_metric& other) = default;
  virtual ~boolean_accuracy_metric() = default;
  boolean_accuracy_metric* copy() const override {
    return new boolean_accuracy_metric(*this);
  }

  /** Return a string name for this metric. */
  std::string name() const override { return "boolean accuracy"; }
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

#endif  // LBANN_METRIC_BOOLEAN_ACCURACY_HPP
