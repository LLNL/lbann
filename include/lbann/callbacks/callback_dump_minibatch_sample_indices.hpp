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
// lbann_callback_dump_minibatch_sample_indices .hpp .cpp - Callbacks
// to dump the list of indices per minibatch
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_MINIBATCH_SAMPLE_INDICES_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_MINIBATCH_SAMPLE_INDICES_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Dump sample indices for each minibatch to files.
 * This will dump the list of indices from the training / validation /
 * testing data that was processed
 * Note this dumps vectors during each mini-batch. This will be slow and
 * produce a lot of output.
 */
class lbann_callback_dump_minibatch_sample_indices : public lbann_callback {
 public:
  using lbann_callback::on_forward_prop_end;
  using lbann_callback::on_evaluate_forward_prop_end;

  /**
   * @param basename The basename for writing files.
   */
  lbann_callback_dump_minibatch_sample_indices(std::string basename, int batch_interval = 1) :
    lbann_callback(batch_interval), m_basename(basename) {}
  lbann_callback_dump_minibatch_sample_indices(
    const lbann_callback_dump_minibatch_sample_indices&) = default;
  lbann_callback_dump_minibatch_sample_indices& operator=(
    const lbann_callback_dump_minibatch_sample_indices&) = default;
  lbann_callback_dump_minibatch_sample_indices* copy() const {
    return new lbann_callback_dump_minibatch_sample_indices(*this);
  }
  void on_forward_prop_end(model *m, Layer *l);
  void on_evaluate_forward_prop_end(model *m, Layer *l);

  void dump_to_file(model *m, Layer *l, int64_t step);

  std::string name() const { return "dump minibatch sample indices"; }
 private:
  /** Basename for writing files. */
  std::string m_basename;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DUMP_MINIBATCH_SAMPLE_INDICES_HPP_INCLUDED
