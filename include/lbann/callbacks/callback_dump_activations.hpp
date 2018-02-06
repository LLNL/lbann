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
// lbann_callback_dump_activations .hpp .cpp - Callbacks to dump activations
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_ACTIVATIONS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_ACTIVATIONS_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Dump activations matrices to files.
 * This will dump each or selected hidden layer's activation matrix after each minibatch or 
 * at the end of an epoch.
 * The matrices are written to files using Elemental's simple ASCII format. This
 * is not meant for checkpointing, but for exporting acitvation matrices for
 * analysis that isn't easily done in LBANN.
 * This will be slow and produce a lot of output if matrices are dumped during each minibatch.
 * If list of layers at which to dump activations is provided, activations/inferences will be dumped 
 * at the end of testing
 */
class lbann_callback_dump_activations : public lbann_callback {
 public:
  using lbann_callback::on_forward_prop_end;

  /**
   * @param basename The basename for writing files.
   */
  lbann_callback_dump_activations(std::string basename, int batch_interval = 1, 
    std::vector<std::string> layer_names=std::vector<std::string>()) :
    lbann_callback(batch_interval), m_basename(std::move(basename)), m_layer_names(layer_names) {}
  lbann_callback_dump_activations(
    const lbann_callback_dump_activations&) = default;
  lbann_callback_dump_activations& operator=(
    const lbann_callback_dump_activations&) = default;
  lbann_callback_dump_activations* copy() const override {
    return new lbann_callback_dump_activations(*this);
  }
  void on_forward_prop_end(model *m, Layer *l) override;
  /** Write activations/inferences to file on test end. */
  void on_test_end(model *m) override;
  std::string name() const override { return "dump activations"; }
 private:
  /** Basename for writing files. */
  std::string m_basename;
  /** List of layers at which to save activations/inferences*/
  std::vector<std::string> m_layer_names;

};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DUMP_ACTIVATIONS_HPP_INCLUDED
