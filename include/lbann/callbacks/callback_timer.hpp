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

#ifndef LBANN_CALLBACKS_CALLBACK_TIMER_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_TIMER_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <chrono>
#include <map>
#include <vector>

namespace lbann {

/** Record and report model timing results.
 *  Reports the total time and mini-batch time statistics for training
 *  epochs and for model evaluations. This reports times for the
 *  master process in each model.
 */
class lbann_callback_timer : public lbann_callback {
public:

  lbann_callback_timer(lbann_summary *summarizer = nullptr)
    : lbann_callback(1, summarizer) {}
  lbann_callback_timer(const lbann_callback_timer&) = default;
  lbann_callback_timer& operator=(const lbann_callback_timer&) = default;
  lbann_callback_timer* copy() const override {
    return new lbann_callback_timer(*this);
  }

  /** Start timing for a training epoch. */
  void on_epoch_begin(model *m) override      { timing_begin(*m); }
  /** Report timing for a training epoch. */
  void on_epoch_end(model *m) override        { timing_end(*m);   }
  /** Start timing for validation. */
  void on_validation_begin(model *m) override { timing_begin(*m); }
  /** Report timing for validation. */
  void on_validation_end(model *m) override   { timing_end(*m);   }
  /** Start timing for testing. */
  void on_test_begin(model *m) override       { timing_begin(*m); }
  /** Report timing for testing. */
  void on_test_end(model *m) override         { timing_end(*m);   }
  /** Record training mini-batch start time. */
  void on_batch_begin(model *m) override          { batch_timing_begin(*m); }
  /** Record training mini-batch run time. */
  void on_batch_end(model *m) override            { batch_timing_end(*m);   }
  /** Record evaluation mini-batch start time. */
  void on_batch_evaluate_begin(model *m) override { batch_timing_begin(*m); }
  /** Record evaluation mini-batch run time. */
  void on_batch_evaluate_end(model *m) override   { batch_timing_end(*m);   }

  /** Callback name. */
  std::string name() const override { return "timer"; }

private:

  /** Timing session start times. */
  std::map<execution_mode,EvalType> m_start_times;
  /** Mini-batch timing session start times. */
  std::map<execution_mode,EvalType> m_batch_start_times;
  /** Mini-batch times. */
  std::map<execution_mode,std::vector<EvalType>> m_batch_times;

  /** Start timing session. */
  void timing_begin(const model& m);
  /** End timing session.
   *  Prints results to standard output.
   */
  void timing_end(model& m);
  /** Start mini-batch timing session. */
  void batch_timing_begin(const model& m);
  /** End mini-batch timing session.
   *  Prints results to standard output.
   */
  void batch_timing_end(const model& m);

};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_timer_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_TIMER_HPP_INCLUDED
