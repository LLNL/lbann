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
// lbann_callback_io .hpp .cpp - Callback hooks for I/O monitoring
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_IO_HPP_INCLUDED
#define LBANN_CALLBACKS_IO_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

#include <google/protobuf/message.h>

#include <string>
#include <vector>

namespace lbann {

/**
 * Print information on the amount of IO that layers do.
 */
class lbann_callback_io : public lbann_callback {
 public:
  lbann_callback_io() = default;
  /** Only apply to specific layers. */
  lbann_callback_io(std::vector<std::string> const& layers)
    : m_layers(layers.begin(), layers.end()) {}

  lbann_callback_io(const lbann_callback_io&) = default;
  lbann_callback_io& operator=(const lbann_callback_io&) = default;
  lbann_callback_io* copy() const override {
    return new lbann_callback_io(*this);
  }
  /** Report how much I/O has occured per data reader */
  void on_epoch_end(model *m) override;
  void on_test_end(model *m) override;
  std::string name() const override { return "io"; }
 private:
  /** Indicies of layers to monitor. */
  std::unordered_set<std::string> m_layers;
};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_disp_io_stats_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

}  // namespace lbann

#endif  // LBANN_CALLBACKS_IO_HPP_INCLUDED
