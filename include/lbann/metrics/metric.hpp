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

#ifndef LBANN_METRIC_HPP
#define LBANN_METRIC_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

// Forward-declare this.
class model;

namespace metrics {

class statistics {
 public:
  statistics() {}
  statistics(const statistics& other) = default;
  statistics& operator=(const statistics& other) = default;
  ~statistics() {}

  void reset_stats() {
    m_total_error += m_error_per_epoch;
    m_total_num_samples += m_samples_per_epoch;
    m_error_per_epoch = 0;
    m_samples_per_epoch = 0;
  }

  /// Error is accumulated as a double -- this works for both sum of
  /// squared errors and categorical errors
  double m_error_per_epoch = 0.0;
  long m_samples_per_epoch = 0;

  double m_total_error = 0.0;
  long m_total_num_samples = 0;
};

/**
 * A metric is used to judge the performance of a model. These are similar to an
 * objective function, but metrics are not used to train the model.
 */
class metric {
 public:
  /// Constructor
  metric(lbann_comm *comm) :
    m_comm(comm) {
  }

  // m_comm and the model pointer are not changed-- copy by value.
  metric(const metric& other) = default;
  metric& operator=(const metric& other) = default;

  /// Destructor
  virtual ~metric() {};

  virtual metric* copy() const = 0;

  virtual void setup(int num_neurons, int mini_batch_size) {}
  virtual void fp_set_std_matrix_view(int cur_mini_batch_size) {}
  virtual double compute_metric(AbsDistMat& predictions_v, AbsDistMat& groundtruth_v) {
    return 0.0;
  }
  virtual double report_metric(execution_mode mode) {
    return 0.0;
  }
  virtual double report_lifetime_metric(execution_mode mode) {
    return 0.0;
  }

  statistics *get_statistics(execution_mode mode);

  void record_error(double error, long num_samples);
  void reset_metric();

  /** Return a string name for this metric. */
  virtual std::string name() const = 0;
  /** Return a display unit, e.g. %, for this metric. */
  virtual std::string display_unit() const { return ""; }

 protected:
  statistics m_training_stats;
  statistics m_validation_stats;
  statistics m_testing_stats;

  lbann_comm *m_comm;

 public:
  model *m_neural_network_model;
};

}  // namespace metrics

}  // namespace lbann

#endif  // LBANN_METRIC_HPP
