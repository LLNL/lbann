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

#ifndef LBANN_OBJECTIVE_FN_HPP_INCLUDED
#define LBANN_OBJECTIVE_FN_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/layers/lbann_layer.hpp"

namespace lbann {

namespace objective_functions {

class statistics {
 public:
  statistics() {}
  statistics(const statistics& other) = default;
  statistics& operator=(const statistics& other) = default;

  ~statistics() {}

  void reset_stats() {
    m_last_mini_batch_avg_cost = 0.0;
    m_aggregate_avg_cost_per_epoch = 0.0;
    m_num_mini_batch_per_epoch = 0;
  }

  /// Error is accumulated as a double -- this works for both sum of
  /// squared errors and categorical errors
  double m_last_mini_batch_avg_cost = 0.0;
  double m_aggregate_avg_cost_per_epoch = 0.0;
  long m_num_mini_batch_per_epoch = 0;
};

/**
 * Objective functions / loss functions are computed and averaged on a batch by batch basis.
 * Additionally, the average loss per batch is averaged over the epoch.
 */
class objective_fn {
 public:
  objective_fn() {}
  objective_fn(const objective_fn& other) = default;
  objective_fn& operator=(const objective_fn& other) = default;
  virtual ~objective_fn() {}
  virtual void setup(int num_neurons, int mini_batch_size) {}
  virtual void fp_set_std_matrix_view(int cur_mini_batch_size) {}
  /// Compute the object function -- Note that it is averaged across a mini-batch
  virtual double compute_obj_fn(const AbsDistMat& predictions_v,
                                const AbsDistMat& groundtruth_v) {
    return 0.0;
  }
  virtual void compute_obj_fn_derivative(const Layer& prev_layer,
                                         const AbsDistMat& predictions_v,
                                         const AbsDistMat& groundtruth_v,
                                         AbsDistMat& error_signal_v) {}

  statistics *get_statistics(execution_mode mode);
  double report_obj_fn(execution_mode mode);
  double report_aggregate_avg_obj_fn(execution_mode mode);
  void record_obj_fn(execution_mode mode, double avg_cost);
  void reset_obj_fn();

  /** Return a string name for this objective function. */
  virtual std::string name() const = 0;

 protected:
  statistics m_training_stats;
  statistics m_validation_stats;
  statistics m_testing_stats;
};

}  // namespace objective_functions

}  // namespace lbann

#endif  // LBANN_OBJECTIVE_FN_INCLUDED
