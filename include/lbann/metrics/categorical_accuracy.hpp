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

#ifndef LBANN_METRIC_CATEGORICAL_ACCURACY_HPP
#define LBANN_METRIC_CATEGORICAL_ACCURACY_HPP

#include "lbann/metrics/metric.hpp"

namespace lbann {

/** Categorical accuracy metric.
 *  The value is a percentage in [0,100].
 */
class categorical_accuracy_metric : public metric {

 public:

  /** Constructor. */
  categorical_accuracy_metric(lbann_comm *comm);

  /** Copy constructor. */
  categorical_accuracy_metric(const categorical_accuracy_metric& other);
  /** Copy assignment operator. */
  categorical_accuracy_metric& operator=(const categorical_accuracy_metric& other);
  /** Destructor. */
  virtual ~categorical_accuracy_metric();
  /** Copy function. */
  categorical_accuracy_metric* copy() const override {
    return new categorical_accuracy_metric(*this);
  }

  /** Return a string name for this metric. */
  std::string name() const override { return "categorical accuracy"; }
  /** Return "%" as a display unit. */
  std::string get_unit() const override { return "%"; }

  /** Setup metric. */
  virtual void setup(model& m) override;

 protected:

  /** Computation to evaluate the metric function.
   *  This returns the sum of metric values across the mini-batch, not
   *  the mean value.
   */
  EvalType evaluate_compute(const AbsDistMat& prediction,
                            const AbsDistMat& ground_truth) override;

 private:

  /** Maximum values in columns of prediction matrix. */
  AbsDistMat *m_prediction_values;
  /** Indices of maximum values in columns of prediction matrix. */
  std::vector<int> m_prediction_indices;

};

}  // namespace lbann

#endif  // LBANN_METRIC_CATEGORICAL_ACCURACY_HPP
