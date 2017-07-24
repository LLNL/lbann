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
// lbann_callback_debug .hpp .cpp - Callback hooks to time training
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DEBUG_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DEBUG_HPP_INCLUDED

#include <chrono>
#include <vector>
#include "lbann/callbacks/lbann_callback.hpp"

namespace lbann {

/**
 * Record the time to execute minibatches and epochs and report it at the end of
 * each epoch.
 * Right now this reports times only for the master node of each model.
 */
class lbann_callback_debug : public lbann_callback {
public:
  lbann_callback_debug(execution_mode phase = execution_mode::invalid, lbann_summary* _summarizer = nullptr) :
    lbann_callback(1, _summarizer), m_debug_phase(phase) {
      set_name("debug");
    }
  /** Start recording time for the epoch. */
  void on_epoch_begin(model* m);
  /** Report epoch and mean minibatch times. */
  void on_epoch_end(model* m);
  /** Start record time for a batch. */
  void on_batch_begin(model* m);
  /** Stop and save time for a batch. */
  void on_batch_end(model* m);
  void on_forward_prop_begin(model* m, Layer* l);
  void on_forward_prop_end(model* m, Layer* l);
  void on_backward_prop_begin(model* m, Layer* l);
  void on_backward_prop_end(model* m, Layer* l);

  /** Start record time for a batch. */
  void on_batch_evaluate_begin(model* m);
  /** Stop and save time for a batch. */
  void on_batch_evaluate_end(model* m);
  void on_evaluate_forward_prop_begin(model* m, Layer* l);
  void on_evaluate_forward_prop_end(model* m, Layer* l);
private:
  execution_mode m_debug_phase;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DEBUG_HPP_INCLUDED
