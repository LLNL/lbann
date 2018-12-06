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

#include "lbann/callbacks/callback_dump_outputs.hpp"

namespace lbann {

lbann_callback_dump_outputs::lbann_callback_dump_outputs(std::set<std::string> layer_names,
                                                         std::set<execution_mode> modes,
                                                         El::Int batch_interval,
                                                         std::string file_prefix)
  : lbann_callback(std::max(batch_interval, El::Int(1))),
    m_layer_names(std::move(layer_names)),
    m_modes(std::move(modes)),
    m_file_prefix(std::move(file_prefix)) {}

void lbann_callback_dump_outputs::dump_outputs(const model& m) {

  // Get mini-batch step information
  const auto& mode = m.get_execution_mode();
  const auto& epoch = m.get_cur_epoch();
  El::Int step = 0;
  switch (mode) {
  case execution_mode::training:
    step = m.get_cur_step();            break;
  case execution_mode::validation:
    step = m.get_cur_validation_step(); break;
  case execution_mode::testing:
    step = m.get_cur_testing_step();    break;
  default: LBANN_ERROR("invalid execution mode");
  }

  // Quit if current mini-batch step doesn't need output dump
  if (!m_modes.empty() && m_modes.count(mode) == 0) { return; }
  if (m_batch_interval != 0 && step % m_batch_interval != 0) { return; }

  // Save layer outputs
  // Note: Each line corresponds to a mini-batch sample. This is the
  // transpose of our internal column-major matrix representation.
  for (const auto& l : m.get_layers()) {
    if (!m_layer_names.empty()
        && m_layer_names.count(l->get_name()) == 0) { continue; }
    for (int i = 0; i < l->get_num_children(); ++i) {
      const CircMat<El::Device::CPU> circ_data(l->get_activations(i));
      if (circ_data.CrossRank() == circ_data.Root()) {
        const auto& data = circ_data.LockedMatrix();
        const std::string filename = (m_file_prefix
                                      + m.get_name()
                                      + "-" + _to_string(mode)
                                      + "-epoch" + std::to_string(epoch)
                                      + "-step" + std::to_string(step)
                                      + "-" + l->get_name()
                                      + "-output" + std::to_string(i)
                                      + ".csv");
        std::ofstream fs(filename.c_str());
        for (El::Int col = 0; col < data.Width(); ++col) {
          for (El::Int row = 0; row < data.Height(); ++row) {
            fs << (row > 0 ? m_delimiter : "") << data(row, col);
          }
          fs << "\n";
        }
      }
    }
  }

}

} // namespace lbann
