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

#ifndef LBANN_METRIC_TOP_K_CATEGORICAL_ACCURACY_HPP
#define LBANN_METRIC_TOP_K_CATEGORICAL_ACCURACY_HPP

#include "lbann/metrics/metric.hpp"

namespace lbann {

/** Top-k categorical accuracy metric.
 *  A prediction is correct if any of the top k predictions are
 *  correct. The value is a percentage in [0,100].
 */
class top_k_categorical_accuracy_metric : public metric {

 public:

  /** Constructor. */
  top_k_categorical_accuracy_metric(int top_k, lbann_comm *comm);

  /** Copy constructor. */
  top_k_categorical_accuracy_metric(const top_k_categorical_accuracy_metric& other) = default;
  /** Copy assignment operator. */
  top_k_categorical_accuracy_metric& operator=(const top_k_categorical_accuracy_metric& other) = default;
  /** Destructor. */
  virtual ~top_k_categorical_accuracy_metric() = default;
  /** Copy function. */
  top_k_categorical_accuracy_metric* copy() const override {
    return new top_k_categorical_accuracy_metric(*this);
  }

  /** Return a string name for this metric. */
  std::string name() const override { return "top-" + std::to_string(m_top_k) +  " categorical accuracy"; }
  /** Return "%" as a display unit. */
  std::string get_unit() const { return "%"; }

 protected:

  /** Computation to evaluate the metric function.
   *  This returns the sum of metric values across the mini-batch, not
   *  the mean value.
   */
  double evaluate_compute(const AbsDistMat& prediction,
                          const AbsDistMat& ground_truth) override;

 private:

  /** Number of classes to check. */
  int m_top_k;

};

}  // namespace lbann

#endif  // LBANN_METRIC_TOP_K_CATEGORICAL_ACCURACY_HPP
