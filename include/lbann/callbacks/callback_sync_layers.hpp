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
// callback_sync_layers.hpp - Callback to synchronize layers
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_SYNC_LAYERS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SYNC_LAYERS_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/** Synchronize layers after forward and backward prop.
 * Additionally updates layer timing information to account for this.
 * Note that this callback should come before the summarizer callback to report
 * time correctly (otherwise it will be shifted by one mini-batch).
 */
class lbann_callback_sync_layers : public lbann_callback {
 public:
  /**
   * @param sync_gpus The GPU stream will be synchronized.
   * @param sync_mpi A global barrier will synchronize processes.
   * @param only_input The only synchronization will be after the input layer in
   * forward prop.
   */
  lbann_callback_sync_layers(bool sync_gpus = true, bool sync_mpi = true,
                             bool only_input = false) :
    lbann_callback(1), m_sync_gpus(sync_gpus), m_sync_mpi(sync_mpi),
    m_only_input(only_input) {}
  lbann_callback_sync_layers(const lbann_callback_sync_layers&) = default;
  lbann_callback_sync_layers& operator=(
    const lbann_callback_sync_layers&) = default;
  lbann_callback_sync_layers* copy() const override {
    return new lbann_callback_sync_layers(*this);
  }
  std::string name() const override { return "sync_layers"; }

  using lbann_callback::on_forward_prop_end;
  using lbann_callback::on_backward_prop_end;

  void on_forward_prop_end(model *m, Layer *l) override;
  void on_backward_prop_end(model *m, Layer *l) override;

 protected:
  /** Whether to synchronize GPUs. */
  bool m_sync_gpus;
  /** Whether to do a global synchronization. */
  bool m_sync_mpi;
  /** Whether to only synchronize after the input layer. */
  bool m_only_input;

  virtual void do_sync(Layer *l);
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SYNC_LAYERS_HPP_INCLUDED
