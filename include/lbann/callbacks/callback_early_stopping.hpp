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
//
// lbann_early_stopping .hpp .cpp - Callback hooks for early stopping
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_EARLY_STOPPING_HPP_INCLUDED
#define LBANN_CALLBACKS_EARLY_STOPPING_HPP_INCLUDED

#include <unordered_set>
#include <unordered_map>
#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Stop training after validation error stops improving.
 */
class lbann_callback_early_stopping : public lbann_callback {
 public:
  /**
   * Continue training until score has not improved for patience epochs.
   */
  lbann_callback_early_stopping(int64_t patience);
  lbann_callback_early_stopping(const lbann_callback_early_stopping&) = default;
  lbann_callback_early_stopping& operator=(
    const lbann_callback_early_stopping&) = default;
  lbann_callback_early_stopping* copy() const override {
    return new lbann_callback_early_stopping(*this);
  }
  /** Update validation score and check for early stopping. */
  void on_validation_end(model *m) override;
  std::string name() const override { return "early stopping"; }
 private:
  /** Number of epochs to wait for improvements. */
  int64_t m_patience;
  /** Last recorded score. */
  EvalType m_last_score = std::numeric_limits<EvalType>::max();
  /** Current number of epochs without improvement. */
  int64_t m_wait = 0;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_EARLY_STOPPING_HPP_INCLUDED
