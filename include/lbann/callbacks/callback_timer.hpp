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
//
// lbann_callback_timer .hpp .cpp - Callback hooks to time training
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_TIMER_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_TIMER_HPP_INCLUDED

#include <chrono>
#include <vector>
#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Record the time to execute minibatches and epochs and report it at the end of
 * each epoch.
 * Right now this reports times only for the master node of each model.
 */
class lbann_callback_timer : public lbann_callback {
 public:
  lbann_callback_timer(lbann_summary *summarizer = nullptr) :
    lbann_callback(1, summarizer) {}
  lbann_callback_timer(const lbann_callback_timer&) = default;
  lbann_callback_timer& operator=(const lbann_callback_timer&) = default;
  /** Start recording time for the epoch. */
  void on_epoch_begin(model *m);
  /** Report epoch and mean minibatch times. */
  void on_epoch_end(model *m);
  /** Start record time for a batch. */
  void on_batch_begin(model *m);
  /** Stop and save time for a batch. */
  void on_batch_end(model *m);
  std::string name() const { return "timer"; }
 private:
  /** Start time for the current epoch. */
  double m_epoch_start;
  /** Start time for the current batch. */
  double m_batch_start;
  /** History of batch times for the current epoch. */
  std::vector<double> m_batch_times;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_TIMER_HPP_INCLUDED
