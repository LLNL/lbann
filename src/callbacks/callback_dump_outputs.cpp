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
#include "lbann/io/file_io.hpp"

#include <libgen.h>

#ifdef LBANN_HAS_CNPY
#include <cnpy.h>
#endif // LBANN_HAS_CNPY

namespace lbann {

namespace {

/** Save text file.
 *  Each line corresponds to a mini-batch sample. This is the
 *  transpose of our internal column-major matrix representation.
 */
void save_text(const std::string& file_name,
               std::string delimiter,
               const CPUMat& data) {
  std::ofstream fs(file_name.c_str());
  if (!fs.is_open()) {
    LBANN_ERROR("failed to open output file (" + file_name + ")");
  }
  for (El::Int col = 0; col < data.Width(); ++col) {
    for (El::Int row = 0; row < data.Height(); ++row) {
      fs << (row > 0 ? delimiter : "") << data(row, col);
    }
    fs << "\n";
  }
}

#ifdef LBANN_HAS_CNPY

/** Save NumPy binary file. */
void save_npy(const std::string& file_name,
              const std::vector<int>& dims,
              const CPUMat& data) {
  if (!data.Contiguous()) {
    LBANN_ERROR("expected contiguous data matrix");
  }
  std::vector<size_t> shape;
  shape.push_back(data.Width());
  for (const auto& d : dims) { shape.push_back(d); }
  cnpy::npy_save(file_name, data.LockedBuffer(), shape);
}

/** Save NumPy zip file. */
void save_npz(const std::string& file_name,
              const std::string& tensor_name,
              const std::vector<int>& dims,
              const CPUMat& data) {
  if (!data.Contiguous()) {
    LBANN_ERROR("expected contiguous data matrix");
  }
  std::vector<size_t> shape;
  shape.push_back(data.Width());
  for (const auto& d : dims) { shape.push_back(d); }
  cnpy::npz_save(file_name, tensor_name, data.LockedBuffer(), shape);
}

#endif // LBANN_HAS_CNPY

} // namespace

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
  if (m_file_format != "csv" && m_file_format != "tsv"
#ifdef LBANN_HAS_CNPY
      && m_file_format != "npy" && m_file_format != "npz"
#endif // LBANN_HAS_CNPY
      ) {
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

  // Create directory if needed
  // Note: dirname takes a non-const C-string as input and ignores
  // trailing '/' characters.
  std::vector<char> prefix_(m_file_prefix.size() + 2);
  m_file_prefix.copy(prefix_.data(), m_file_prefix.size());
  prefix_[prefix_.size()-2] = 'a';
  prefix_[prefix_.size()-1] = '\0';
  const std::string dir = dirname(prefix_.data());
  if (dir != ".") {
    lbann::makedir(dir.c_str());
  }

  // Save layer outputs on root process
  for (int i = 0; i < l.get_num_children(); ++i) {
    const CircMat<El::Device::CPU> circ_data(l.get_activations(i));
    if (circ_data.CrossRank() == circ_data.Root()) {
      const auto& data = static_cast<const CPUMat&>(circ_data.LockedMatrix());
      const std::string file_name = (m_file_prefix
                                     + m.get_name()
                                     + "-" + _to_string(mode)
                                     + "-epoch" + std::to_string(epoch)
                                     + "-step" + std::to_string(step)
                                     + "-" + l.get_name()
                                     + "-output" + std::to_string(i)
                                     + "." + m_file_format);
      if (m_file_format == "csv") {
        save_text(file_name, ",", data);
      } else if (m_file_format == "tsv") {
        save_text(file_name, "\t", data);
      }
#ifdef LBANN_HAS_CNPY
      else if (m_file_format == "npy") {
        save_npy(file_name, l.get_output_dims(i), data);
      } else if (m_file_format == "npz") {
        save_npz(file_name,
                 l.get_name() + "_output" + std::to_string(i),
                 l.get_output_dims(i),
                 data);
      }
#endif // LBANN_HAS_CNPY
    }
  }

}

} // namespace lbann
