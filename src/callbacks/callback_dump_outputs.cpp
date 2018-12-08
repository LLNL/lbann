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
                                                         std::string file_prefix,
                                                         std::string file_format)
  : lbann_callback(std::max(batch_interval, El::Int(1))),
    m_layer_names(std::move(layer_names)),
    m_modes(std::move(modes)),
    m_file_prefix(std::move(file_prefix)),
    m_file_format(std::move(file_format)) {

  // Initialize file format
  if (m_file_format.empty()) { m_file_format = "csv"; }
  if (m_file_format == "csv") {
    m_delimiter = ",";
  } else if (m_file_format == "txt") {
    m_delimiter = ",";
  } else if (m_file_format == "tsv") {
    m_delimiter = "\t";
  } else if (m_file_format == "bin") {
  } else {
    LBANN_ERROR("invalid file format (" + m_file_format + ")");
  }

}

void lbann_callback_dump_outputs::dump_outputs(const model& m, const Layer& l) {

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

  // Quit if output dump isn't needed
  if (!m_modes.empty() && m_modes.count(mode) == 0) { return; }
  if (!m_layer_names.empty()
      && m_layer_names.count(l.get_name()) == 0) { return; }

  // Save layer outputs on root process
  // Note: Each line corresponds to a mini-batch sample. This is the
  // transpose of our internal column-major matrix representation.
  for (int i = 0; i < l.get_num_children(); ++i) {
    const CircMat<El::Device::CPU> circ_data(l.get_activations(i));
    if (circ_data.CrossRank() == circ_data.Root()) {
      const auto& data = circ_data.LockedMatrix();

      // Open output file
      const std::string filename = (m_file_prefix
                                    + m.get_name()
                                    + "-" + _to_string(mode)
                                    + "-epoch" + std::to_string(epoch)
                                    + "-step" + std::to_string(step)
                                    + "-" + l.get_name()
                                    + "-output" + std::to_string(i)
                                    + "." + m_file_format);
      std::ios_base::openmode fs_flags = (m_file_format == "bin" ?
                                          std::ios_base::out | std::ios_base::binary :
                                          std::ios_base::out);
      std::ofstream fs(filename.c_str(), fs_flags);
      if (!fs.is_open()) {
        std::stringstream err;
        err << "callback \"" << name() << "\" "
            << "failed to open output file (" << filename << ")";
        LBANN_ERROR(err);
      }

      // Write to output file
      if (m_file_format == "bin") {
        std::vector<float> float_data(data.Height() * data.Width());
        LBANN_OMP_PARALLEL_FOR_COLLAPSE2
        for (El::Int col = 0; col < data.Width(); ++col) {
          for (El::Int row = 0; row < data.Height(); ++row) {
            float_data[row + col * data.Height()] = data(row, col);
          }
        }
        fs.write(reinterpret_cast<const char*>(float_data.data()),
                 float_data.size() * sizeof(float));
      } else {
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
