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
// lbann_callback_dump_gradients .hpp .cpp - Callbacks to dump gradients
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_GRADIENTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_GRADIENTS_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Dump gradient matrices to files.
 * This will dump each hidden layer's gradient matrix after each minibatch.
 * The matrices are written to files using Elemental's simple ASCII format. This
 * is not meant for checkpointing, but for exporting gradient matrices for
 * analysis that isn't easily done in LBANN.
 * Note this dumps matrices during each mini-batch. This will be slow and
 * produce a lot of output.
 */
class lbann_callback_dump_gradients : public lbann_callback {
 public:
  using lbann_callback::on_backward_prop_end;

  /**
   * @param basename The basename for writing files.
   */
  lbann_callback_dump_gradients(std::string basename, int batch_interval = 1) :
    lbann_callback(batch_interval), m_basename(basename) {}
  lbann_callback_dump_gradients(
    const lbann_callback_dump_gradients&) = default;
  lbann_callback_dump_gradients& operator=(
    const lbann_callback_dump_gradients&) = default;
  lbann_callback_dump_gradients* copy() const {
    return new lbann_callback_dump_gradients(*this);
  }
  void on_backward_prop_end(model *m, Layer *l);
  std::string name() const { return "dump gradients"; }
 private:
  /** Basename for writing files. */
  std::string m_basename;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DUMP_GRADIENTS_HPP_INCLUDED
