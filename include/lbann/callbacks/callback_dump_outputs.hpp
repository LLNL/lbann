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

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_OUTPUTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_OUTPUTS_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

#include <set>
#include <string>

namespace lbann {

/** @brief Dump layer output tensors to files.
 *
 *  Saves a file for each output tensor of each selected layer,
 *  computed at each mini-batch step. Output files have the form
 *  "<model>-<mode>-epoch<#>-step<#>-<layer>-output<#>.<format>".
 *  This is primarily intended as a debugging tool, although it can be
 *  used for inference when performance is not critical.
 *
 *  For NumPy file formats (npy and npz), tensor dimensions are
 *  recorded. For text file formats (CSV and TSV), each line contains
 *  flattened tensor data corresponding to one mini-batch sample
 *  (which is the transpose of the column-major matrix representation
 *  we use internally).
 *
 *  CNPY is required to export to NumPy file formats (npy and npz).
 */
class lbann_callback_dump_outputs : public lbann_callback {
public:

  /** @brief Construct a callback to dump outputs.
   *
   *  @param layer_names    Names of layers with output dumps
   *                        (default: dump outputs for all layers).
   *  @param modes          Execution modes with output dumps
   *                        (default: dump outputs for all modes).
   *  @param batch_interval Frequency of output dumps (default: dump
   *                        outputs at each mini-batch step).
   *  @param directory      Directory for output files (default: current
   *                        working directory).
   *  @param file_format    Output file format. Options are csv, tsv,
   *                        npy, npz (default: csv).
   */
  lbann_callback_dump_outputs(
    std::set<std::string> layer_names,// = std::set<std::string>(),
    std::set<execution_mode> modes, // = std::set<std::string>(),
    El::Int batch_interval = 0,
    std::string directory = "",
    std::string file_format = "");

  lbann_callback_dump_outputs* copy() const override {
    return new lbann_callback_dump_outputs(*this);
  }
  std::string name() const override { return "dump outputs"; }

  void on_forward_prop_end(model* m, Layer* l) override          { dump_outputs(*m, *l); }
  void on_evaluate_forward_prop_end(model* m, Layer* l) override {
       if(m->get_step() % m_batch_interval == 0) { 
         dump_outputs(*m, *l); 
       }
  }

private:

  /** @brief   Names of layers with output dumps.
   *  @details If empty, outputs will be dumped for all layers.
   */
  std::set<std::string> m_layer_names;

  /** @brief   Execution modes with output dumps.
   *  @details If empty, outputs will be dumped for all execution modes.
   */
  std::set<execution_mode> m_modes;

  /** @brief   Directory for output files.
   *  @details Pathname has trailing '/'.
   */
  std::string m_directory;

  /** @brief Output file format. */
  std::string m_file_format;

  /** @brief   Dump outputs to file.
   *  @details Returns immediately if an output dump is not needed.
   */
  void dump_outputs(const model& m, const Layer& l);

};

} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_DUMP_OUTPUTS_HPP_INCLUDED
