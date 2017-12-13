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

#ifndef LBANN_METRIC_HPP_INCLUDED
#define LBANN_METRIC_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

// Forward declarations
class model;
class Layer;
class target_layer;

/** Abstract base class for metric functions.
 *  A metric function can be used to evaluate the performance of a
 *  model without affecting the training process.
 */
class metric {

 public:

  /** Constructor. */
  metric(lbann_comm *comm);

  /** Copy constructor. */
  metric(const metric& other) = default;
  /** Copy assignment operator. */
  metric& operator=(const metric& other) = default;
  /** Destructor. */
  virtual ~metric() = default;
  /** Copy function. */
  virtual metric* copy() const = 0;

  /** Return a string name for this metric. */
  virtual std::string name() const = 0;
  /** Return a display unit for this metric.
   *  Default is an empty string. This is overriden if the metric has
   *  units, e.g. "%" or "sec".
   */
  virtual std::string get_unit() const { return ""; }

  /** Setup metric. */
  virtual void setup(model& m);
  
  /** Evaluate the metric value. */
  DataType evaluate();

  /** Get history of metric values. */
  std::vector<DataType> get_history_values() const {
    return m_history_values;
  }
  /** Get history of mini-batch sizes. */
  std::vector<int> get_history_mini_batch_sizes() const {
    return m_history_mini_batch_sizes;
  }
  /** Get total number of samples in history.
   *  This is the sum of mini-batch sizes in history.
   */
  int get_history_num_samples() const;
  /** Clear history of metric values. */
  void clear_history();

  /** Get mean metric value in history.
   *  If mini-batch sizes are not identical, the mean is over the
   *  sample values rather than over the mini-batch mean values.
   */
  DataType get_history_mean_value() const;

  /** Set pointer to target layer. */
  void set_target_layer(const target_layer *target) { m_target_layer = target; }
  /** Get target layer. */
  const target_layer& get_target_layer() const;

  /** Get list of pointers to layers. */
  std::vector<Layer*> get_layer_pointers() const;
  /** Set list of pointers to layers. */
  void set_layer_pointers(std::vector<Layer*> layers);

 protected:

  /** Get LBANN communicator. */
  lbann_comm *get_comm() { return m_comm; }

  /** Computation to evaluate the metric function. */
  virtual DataType evaluate_compute(const AbsDistMat& prediction,
                                    const AbsDistMat& ground_truth) = 0;

 private:

  /** LBANN communicator. */
  lbann_comm *m_comm;

  /** Pointer to target layer. */
  const target_layer *m_target_layer;

  /** History of metric values. */
  std::vector<DataType> m_history_values;
  /** History of mini-batch sizes. */
  std::vector<int> m_history_mini_batch_sizes;

};

}  // namespace lbann

#endif  // LBANN_METRIC_HPP_INCLUDED
