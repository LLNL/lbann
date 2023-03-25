////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
namespace callback {

/** Synchronize layers after forward and backward prop.
 * Additionally updates layer timing information to account for this.
 * Note that this callback should come before the summarizer callback to report
 * time correctly (otherwise it will be shifted by one mini-batch).
 */
class sync_layers : public callback_base
{
public:
  /**
   * @param sync_gpus The GPU stream will be synchronized.
   * @param sync_mpi A global barrier will synchronize processes.
   * @param only_input The only synchronization will be after the input layer in
   * forward prop.
   */
  sync_layers(bool sync_gpus = true,
              bool sync_mpi = true,
              bool only_input = false)
    : callback_base(1),
      m_sync_gpus(sync_gpus),
      m_sync_mpi(sync_mpi),
      m_only_input(only_input)
  {}
  sync_layers(const sync_layers&) = default;
  sync_layers& operator=(const sync_layers&) = default;
  sync_layers* copy() const override { return new sync_layers(*this); }
  std::string name() const override { return "sync_layers"; }

  using callback_base::on_backward_prop_end;
  using callback_base::on_forward_prop_end;

  void on_forward_prop_end(model* m, Layer* l) override;
  void on_backward_prop_end(model* m, Layer* l) override;

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

protected:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /** Whether to synchronize GPUs. */
  bool m_sync_gpus;
  /** Whether to do a global synchronization. */
  bool m_sync_mpi;
  /** Whether to only synchronize after the input layer. */
  bool m_only_input;

  virtual void do_sync(Layer* l);
};

// Builder function
std::unique_ptr<callback_base>
build_sync_layers_callback_from_pbuf(const google::protobuf::Message&,
                                     std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_SYNC_LAYERS_HPP_INCLUDED
