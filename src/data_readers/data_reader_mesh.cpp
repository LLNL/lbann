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
// data_reader_mesh .hpp .cpp - data reader for mesh data
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_mesh.hpp"
#include "lbann/utils/glob.hpp"
#include <omp.h>

namespace lbann {

mesh_reader::mesh_reader(bool shuffle)
  : generic_data_reader(shuffle) {}

void mesh_reader::load() {
  if (m_data_height == 0 || m_data_width == 0) {
    throw lbann_exception("mesh_reader: data shape must be non-zero");
  }
  // Compute total number of samples based on number of targets.
  std::vector<std::string> matches = glob(
    get_file_dir() + m_target_name + m_suffix + "/*.bin");
  if (matches.size() == 0) {
    throw lbann_exception("mesh_reader: could not find any targets");
  }
  m_num_samples = matches.size();
  // Set up buffers to load data into.
  m_load_bufs.resize(omp_get_max_threads());
  for (auto&& buf : m_load_bufs) {
    buf.resize(m_data_height * m_data_width);
  }
  // Set up the format string.
  if (std::pow(10, m_index_length) <= m_num_samples) {
    throw lbann_exception("mesh_reader: index length too small");
  }
  m_index_format_str = "%0" + std::to_string(m_index_length) + "d";
  // Set up to record flipping if needed.
  if (m_random_flips) {
    m_flip_choices.resize(m_num_samples);
  }
  // Reset indices.
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

bool mesh_reader::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  if (m_random_flips) {
    fast_rng_gen& gen = get_fast_io_generator();
    std::uniform_int_distribution<int> dist(0, 1);
    m_flip_choices[data_id].first = dist(gen);
    m_flip_choices[data_id].second = dist(gen);
  }
  for (size_t i = 0; i < m_channels.size(); ++i) {
    Mat X_view = El::View(
      X, El::IR(i*m_data_height*m_data_width, (i+1)*m_data_height*m_data_width),
      El::IR(mb_idx));
    load_file(data_id, m_channels[i], X_view);
  }
  return true;
}

bool mesh_reader::fetch_response(CPUMat& Y, int data_id, int mb_idx) {
  Mat Y_view = El::View(Y, El::ALL, El::IR(mb_idx));
  load_file(data_id, m_target_name, Y_view);
  return true;
}

void mesh_reader::load_file(int data_id, const std::string channel, Mat& mat) {
  const std::string filename = construct_filename(channel, data_id);
  std::ifstream f(filename, std::ios::binary);
  if (f.fail()) {
    throw lbann_exception("mesh_reader: failed to open " + filename);
  }
  // Load into a local buffer.
  DataType* buf = m_load_bufs[omp_get_thread_num()].data();
  if (!f.read((char*) buf, m_data_height * m_data_width * sizeof(float))) {
    throw lbann_exception("mesh_reader: failed to read " + filename);
  }
  if (std::is_same<float, DataType>::value) {
    // Need to transpose from row-major to column-major order.
    Mat tmp_mat(m_data_width, m_data_height, buf, m_data_width);
    Mat mat_reshape(m_data_height, m_data_width, mat.Buffer(), m_data_height);
    El::Transpose(tmp_mat, mat_reshape);
    // Flip if needed.
    if (m_random_flips) {
      if (m_flip_choices[data_id].first) {
        horizontal_flip(mat_reshape);
      }
      if (m_flip_choices[data_id].second) {
        vertical_flip(mat_reshape);
      }
    }
  } else {
    // Need to transpose and convert from float. Not yet supported.
    throw lbann_exception("mesh_reader: does not support DataType != float");
  }
}

std::string mesh_reader::construct_filename(std::string channel, int data_id) {
  std::string filename = get_file_dir() + channel + m_suffix + "/" + channel;
  char idx[m_index_length + 1];
  std::snprintf(idx, m_index_length + 1, m_index_format_str.c_str(), data_id);
  return filename + std::string(idx) + ".bin";
}

void mesh_reader::horizontal_flip(CPUMat& mat) {
  // TODO: Could probably optimize this for better locality.
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  for (El::Int row = 0; row < height; ++row) {
    for (El::Int col = 0; col < (width / 2); ++col) {
      DataType tmp = mat(row, col);
      mat(row, col) = mat(row, width - col - 1);
      mat(row, width - col - 1) = tmp;
    }
  }
}

void mesh_reader::vertical_flip(CPUMat& mat) {
  // TODO: Could probably optimize this for better locality.
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  for (El::Int row = 0; row < (height / 2); ++row) {
    for (El::Int col = 0; col < width; ++col) {
      DataType tmp = mat(row, col);
      mat(row, col) = mat(height - row - 1, col);
      mat(height - row - 1, col) = tmp;
    }
  }
}

}  // namespace lbann
