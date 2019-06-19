////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
#include "lbann/io/persist.hpp"

namespace lbann {

// Forward declarations
class model;
class Layer;

/** Metric statistics. */
struct metric_statistics {
  /** Sum of metric values. */
  EvalType m_sum;
  /** Number of samples. */
  int m_num_samples;
  /** Default constructor. */
  metric_statistics() { reset(); }
  /** Move constructor. */
  metric_statistics(metric_statistics& other) = default;
  /** Copy constructor. */
  metric_statistics(const metric_statistics& other) = default;
  /** Move assignment operator. */
  metric_statistics& operator=(metric_statistics& other) = default;
  /** Copy assignment operator. */
  metric_statistics& operator=(const metric_statistics& other) = default;
  /** Destructor. */
  ~metric_statistics() = default;
  /** Add metric value to statistics. */
  void add_value(EvalType value, int num_samples = 1);
  /** Get mean metric value.
   *  If mini-batch sizes are not identical, the mean is over the
   *  sample values rather than over the mini-batch mean values.
   */
  EvalType get_mean() const;
  /** Get number of samples. */
  int get_num_samples() const { return m_num_samples; }
  /** Reset statistics. */
  void reset();

  //************************************************************************
  // Checkpointing
  //************************************************************************
  /** struct used to serialize mode fields in file and MPI transfer */
  struct packing_header {
    double sum;
    uint64_t num_samples;
  };
  bool pack_scalars(persist& p);
  bool unpack_scalars(persist& p, struct packing_header *header);
  void unpack_header(struct packing_header& header);

};

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
  virtual void setup(model& m) {}

  /** Evaluate the metric value.
   *  This function takes the model's current mini-batch size. If
   *  multiple models are being trained, the current mini-batch size
   *  may be different from the effective mini-batch size. The result
   *  is stored in history.
   */
  virtual EvalType evaluate(execution_mode mode, int mini_batch_size) = 0;

  /** Clear all statistics. */
  void reset_statistics() { m_statistics.clear(); }
  /** Clear statistics for an execution mode. */
  void reset_statistics(execution_mode mode) { m_statistics.erase(mode); }

  /** Get mean metric value.
   *  If mini-batch sizes are not identical, the mean is over the
   *  sample values rather than over the mini-batch mean values.
   */
  EvalType get_mean_value(execution_mode mode) const;
  /** Get number of samples for statistics. */
  int get_statistics_num_samples(execution_mode mode) const;

  /** Get list of pointers to layers. */
  virtual std::vector<Layer*> get_layer_pointers() const;
  /** Set list of pointers to layers. */
  virtual void set_layer_pointers(std::vector<Layer*> layers);

  /** Get the time spent in evaluation for this metric (const). */
  EvalType get_evaluate_time() const { return m_evaluate_time; }
  /** Get the time spent in evaluation for this metric. */
  EvalType& get_evaluate_time() { return m_evaluate_time; }

  /** Reset timing counters. */
  void reset_counters() {
    m_evaluate_time = 0.0;
  }

  /** Save metric state to checkpoint. */
  virtual bool save_to_checkpoint_shared(persist& p);
  /** Load metric state from checkpoint. */
  virtual bool load_from_checkpoint_shared(persist& p);

  virtual bool save_to_checkpoint_distributed(persist& p);
  virtual bool load_from_checkpoint_distributed(persist& p);

 protected:

  /** Computation to evaluate the metric function.
   *  This should return the sum of metric values across the
   *  mini-batch, not the mean value.
   */
  virtual EvalType evaluate_compute(const AbsDistMat& prediction,
                                    const AbsDistMat& ground_truth) = 0;

  /** Get LBANN communicator. */
  lbann_comm& get_comm() { return *m_comm; }

  /** Get metric statistics. */
  std::map<execution_mode,metric_statistics>& get_statistics() { return m_statistics; }

 private:

  /** LBANN communicator. */
  lbann_comm *m_comm;

  /** Metric statistics. */
  std::map<execution_mode,metric_statistics> m_statistics;

  /** Runtime for the metric evaluation. */
  EvalType m_evaluate_time = 0.0;

};

}  // namespace lbann

#endif  // LBANN_METRIC_HPP_INCLUDED
