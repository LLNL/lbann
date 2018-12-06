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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_OUTPUTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_OUTPUTS_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/** Dump layer output tensors to files.
 *  This callback generates a text file for each output tensor of each
 *  selected layer, computed at each mini-batch step. Each line
 *  contains flattened tensor data corresponding to one mini-batch
 *  sample (note that this is the transpose of the column-major matrix
 *  representation we use internally). Output files are in the form
 *  "<prefix><model>-<mode>-epoch<#>-step<#>-<layer>-output<#>.csv". This
 *  is primarily intended as a debugging tool, although it can be used
 *  for inference when performance is not critical.
 *
 *  Currently, only CSV output files are supported.
 */
class lbann_callback_dump_outputs : public lbann_callback {
public:

  /** Constructor.
   *  @param layer_names    Names of layers with output dumps
   *                        (default: dump outputs for all layers).
   *  @param modes          Execution modes with output dumps
   *                        (default: dump outputs for all modes).
   *  @param batch_interval Frequency of output dumps (default: dump
   *                        outputs at each mini-batch step).
   *  @param file_prefix    Prefix for output file names.
   */
  lbann_callback_dump_outputs(std::set<std::string> layer_names = {},
                              std::set<execution_mode> modes = {},
                              El::Int batch_interval = 0,
                              std::string file_prefix = "");
  lbann_callback_dump_outputs* copy() const override {
    return new lbann_callback_dump_outputs(*this);
  }
  std::string name() const override { return "dump outputs"; }

  void on_forward_prop_end(model* m, Layer* l) override          { dump_outputs(*m, *l); }
  void on_evaluate_forward_prop_end(model* m, Layer* l) override { dump_outputs(*m, *l); }

private:

  /** Names of layers with output dumps.
   *  If empty, outputs will be dumped for all layers.
   */
  std::set<std::string> m_layer_names;

  /** Execution modes with output dumps.
   *  If empty, outputs will be dumped for all execution modes.
   */
  std::set<execution_mode> m_modes;

  /** Prefix for output files. */
  std::string m_file_prefix;

  /** Delimiter for output files.
   *  Currently hard-coded to output CSV files.
   */
  std::string m_delimiter = ",";

  /** Dump outputs to file.
   *  Returns immediately if an output dump is not needed.
   */
  void dump_outputs(const model& m, const Layer& l);

};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DUMP_OUTPUTS_HPP_INCLUDED
