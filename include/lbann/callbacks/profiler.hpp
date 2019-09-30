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
// lbann_callback_timer .hpp .cpp - Callback hooks to time training
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_PROFILER_HPP_INCLUDED
#define LBANN_CALLBACKS_PROFILER_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 */
class lbann_callback_profiler : public lbann_callback {
 public:
  lbann_callback_profiler(bool sync = false, bool skip_init = false);
  lbann_callback_profiler(const lbann_callback_profiler&) = default;
  lbann_callback_profiler& operator=(const lbann_callback_profiler&) = default;
  lbann_callback_profiler* copy() const override {
    return new lbann_callback_profiler(*this);
  }
  void on_epoch_begin(model *m) override;
  void on_epoch_end(model *m) override;
  void on_validation_begin(model *m) override;
  void on_validation_end(model *m) override;
  void on_test_begin(model *m) override;
  void on_test_end(model *m) override;
  void on_batch_begin(model *m) override;
  void on_batch_end(model *m) override;
  void on_batch_evaluate_begin(model *m) override;
  void on_batch_evaluate_end(model *m) override;
  void on_forward_prop_begin(model *m) override;
  void on_forward_prop_end(model *m) override;
  void on_evaluate_forward_prop_begin(model *m) override;
  void on_evaluate_forward_prop_end(model *m) override;
  void on_backward_prop_begin(model *m) override;
  void on_backward_prop_end(model *m) override;
  void on_forward_prop_begin(model *m, Layer *l) override;
  void on_forward_prop_end(model *m, Layer *l) override;
  void on_evaluate_forward_prop_begin(model *m, Layer *l) override;
  void on_evaluate_forward_prop_end(model *m, Layer *l) override;
  void on_backward_prop_begin(model *m, Layer *l) override;
  void on_backward_prop_end(model *m, Layer *l) override;
  void on_optimize_begin(model *m) override;
  void on_optimize_end(model *m) override;
  void on_optimize_begin(model *m, weights *w) override;
  void on_optimize_end(model *m, weights *w) override;
  std::string name() const override { return "profiler"; }
 private:
  /** Get a color to use in the profiler for a layer. */
  int get_color(Layer *l);
  /** Whether to synchronize the when setting up profile regions. */
  bool m_sync;
  /** Whether to skip initial iterations. */
  bool m_skip_init;
};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_profiler_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

}  // namespace lbann

#endif  // LBANN_CALLBACKS_PROFILER_HPP_INCLUDED
