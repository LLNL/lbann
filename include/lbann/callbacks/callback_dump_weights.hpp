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
// lbann_callback_dump_weights .hpp .cpp - Callbacks to dump weight matrices
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_WEIGHTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_WEIGHTS_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Dump weight matrices to files.
 * This will dump each hidden layer's weight/bias matrix after each epoch.
 * The matrices are written to files using Elemental's simple ASCII format. This
 * is not meant for checkpointing, but for exporting weight matrices for
 * analysis that isn't easily done in LBANN.
 */
class lbann_callback_dump_weights : public lbann_callback {
 public:
  /**
   * @param basename The basename for writing files.
   */
  lbann_callback_dump_weights(std::string basename) :
    lbann_callback(), m_basename(std::move(basename)) {}
  lbann_callback_dump_weights(const lbann_callback_dump_weights&) = default;
  lbann_callback_dump_weights& operator=(
    const lbann_callback_dump_weights&) = default;
  lbann_callback_dump_weights* copy() const override {
    return new lbann_callback_dump_weights(*this);
  }
  void on_train_begin(model *m) override;
  void on_epoch_end(model *m) override;
  std::string name() const override { return "dump weights"; }
 private:
  /** Basename for writing files. */
  std::string m_basename;
  /// Dump weights from learning layers.
  void dump_weights(model *m, std::string s = "");
};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_dump_weights_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary*);

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DUMP_WEIGHTS_HPP_INCLUDED
