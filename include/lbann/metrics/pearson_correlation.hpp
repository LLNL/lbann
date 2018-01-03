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

#ifndef LBANN_METRIC_PEARSON_CORRELATION_HPP
#define LBANN_METRIC_PEARSON_CORRELATION_HPP

#include "lbann/metrics/metric.hpp"

namespace lbann {

/** Pearson correlation metric. */
class pearson_correlation_metric : public metric {

 public:

  /** Constructor. */
  pearson_correlation_metric(lbann_comm *comm);

  /** Copy constructor. */
  pearson_correlation_metric(const pearson_correlation_metric& other);
  /** Copy assignment operator. */
  pearson_correlation_metric& operator=(const pearson_correlation_metric& other);
  /** Destructor. */
  virtual ~pearson_correlation_metric();
  /** Copy function. */
  pearson_correlation_metric* copy() const override {
    return new pearson_correlation_metric(*this);
  }

  /** Return a string name for this metric. */
  std::string name() const override { return "Pearson correlation"; }

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

  /** Column-wise means of prediction matrix. */
  AbsDistMat *m_prediction_means;
  /** Column-wise standard deviations of prediction matrix. */
  AbsDistMat *m_prediction_stdevs;
  /** Column-wise means of ground truth matrix. */
  AbsDistMat *m_ground_truth_means;
  /** Column-wise standard deviations of ground truth matrix. */
  AbsDistMat *m_ground_truth_stdevs;
  /** Column-wise covariance between prediction and ground truth matrices. */
  AbsDistMat *m_covariances;

};

}  // namespace lbann

#endif  // LBANN_METRIC_PEARSON_CORRELATION_HPP
