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
// lbann_callback_debug .hpp .cpp - Callback hooks to debug LBANN
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DEBUG_IO_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DEBUG_IO_HPP_INCLUDED

#include <chrono>
#include <vector>
#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Print status updates on where training is.
 */
class lbann_callback_debug_io : public lbann_callback {
 public:
  using lbann_callback::on_forward_prop_begin;
  using lbann_callback::on_forward_prop_end;
  using lbann_callback::on_backward_prop_begin;
  using lbann_callback::on_backward_prop_end;
  using lbann_callback::on_evaluate_forward_prop_begin;
  using lbann_callback::on_evaluate_forward_prop_end;

  /**
   * Debug a particular phase; use invalid to debug every phase.
   */
  lbann_callback_debug_io(execution_mode phase = execution_mode::invalid,
                       lbann_summary *summarizer = nullptr) :
    lbann_callback(1, summarizer), m_debug_phase(phase) {}
  lbann_callback_debug_io(const lbann_callback_debug_io&) = default;
  lbann_callback_debug_io& operator=(
    const lbann_callback_debug_io&) = default;
  lbann_callback_debug_io* copy() const { return new lbann_callback_debug_io(*this); }
  /** Print that a batch is being started. */
  void on_batch_begin(model *m);
  /** Print that forward prop for a layer is beginning. */
  void on_forward_prop_begin(model *m, Layer *l);

#if 0
  /** Print that an evaluation batch is being started. */
  void on_batch_evaluate_begin(model *m);
  /** Print that an evaluation batch has completed. */
  void on_batch_evaluate_end(model *m);
  /** Print that an evaluation forward prop is beginning. */
  void on_evaluate_forward_prop_begin(model *m, Layer *l);
  /** Print that an evaluation forward prop has completed. */
  void on_evaluate_forward_prop_end(model *m, Layer *l);
#endif

  std::string name() const { return "debug"; }
 private:
  /** The phase to debug. */
  execution_mode m_debug_phase;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DEBUG_IO_HPP_INCLUDED
