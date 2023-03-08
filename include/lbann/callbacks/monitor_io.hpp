////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
// monitor_io .hpp .cpp - Callback hooks for I/O monitoring
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_IO_HPP_INCLUDED
#define LBANN_CALLBACKS_IO_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

#include <google/protobuf/message.h>

#include <set>
#include <string>
#include <vector>

namespace lbann {
namespace callback {

/**
 * Print information on the amount of IO that layers do.
 */
class monitor_io : public callback_base
{
public:
  monitor_io() = default;
  /** Only apply to specific layers. */
  monitor_io(std::vector<std::string> const& layers)
    : m_layers(layers.begin(), layers.end())
  {}

  monitor_io(const monitor_io&) = default;
  monitor_io& operator=(const monitor_io&) = default;
  monitor_io* copy() const override { return new monitor_io(*this); }
  /** Report how much I/O has occured per data reader */
  void on_epoch_end(model* m) override;
  void on_test_end(model* m) override;
  std::string name() const override { return "monitor_io"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /** Indicies of layers to monitor. */
  std::unordered_set<std::string> m_layers;
};

// Builder function
std::unique_ptr<callback_base>
build_monitor_io_callback_from_pbuf(const google::protobuf::Message&,
                                    std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_IO_HPP_INCLUDED
