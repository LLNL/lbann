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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_ERROR_SIGNALS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_ERROR_SIGNALS_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/** Dump gradients w.r.t. inputs to file.
 *  After each layer performs a backward prop step, this callback will
 *  dump the gradients w.r.t. inputs (the "error signals") to a
 *  human-readable ASCII file. This is slow and produces a lot of output.
 */
class lbann_callback_dump_error_signals : public lbann_callback {
 public:

  /** Constructor.
   *  @param basename The basename for output files.
   */
  lbann_callback_dump_error_signals(std::string basename = "")
    : lbann_callback(), m_basename(basename) {}
  lbann_callback_dump_error_signals* copy() const override {
    return new lbann_callback_dump_error_signals(*this);
  }
  std::string name() const override { return "dump error signals"; }

  /** Write error signals to file after each backward prop step. */
  void on_backward_prop_end(model *m, Layer *l) override;

 private:
  /** Basename for output files. */
  std::string m_basename;

};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_dump_error_signals_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DUMP_ERROR_SIGNALS_HPP_INCLUDED
