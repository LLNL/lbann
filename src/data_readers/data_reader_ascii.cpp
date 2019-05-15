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
// lbann_data_reader_ascii .hpp .cpp - generic_data_reader class for ASCII text files
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_ascii.hpp"
#include <cstdio>
namespace lbann {

ascii_reader::ascii_reader(int sequence_length, bool shuffle)
  : generic_data_reader(shuffle), m_sequence_length(sequence_length) {}

bool ascii_reader::fetch_datum(CPUMat& X, int data_id, int mb_idx) {

  // Get text sequence from file
  const int pos = data_id - m_sequence_length;
  const int num_chars = (std::min(pos + m_sequence_length, m_file_size)
                         - std::max(pos, 0));
  std::vector<char> sequence(m_sequence_length, 0);
  if (num_chars > 0) {
    std::ifstream fs(get_file_dir() + get_data_filename(),
                     std::fstream::in);
    fs.seekg(std::max(pos, 0));
    fs.read(&sequence[std::max(-pos, 0)], num_chars);
    fs.close();
  }

  // Convert text sequence to binary vector
  for (int i = 0; i < m_sequence_length; ++i) {
    auto current_char = (int) sequence[i];
    if (current_char < 0 || current_char >= 128) {
      current_char = 0;
    }
    X(128 * i + current_char, mb_idx) = DataType(1);
  }

  return true;
}

bool ascii_reader::fetch_label(CPUMat& Y, int data_id, int mb_idx) {

  // Get text sequence from file
  const int pos = data_id - m_sequence_length + 1;
  const int num_chars = (std::min(pos + m_sequence_length, m_file_size)
                         - std::max(pos, 0));
  std::vector<char> sequence(m_sequence_length, 0);
  if (num_chars > 0) {
    std::ifstream fs(get_file_dir() + get_data_filename(),
                     std::fstream::in);
    fs.seekg(std::max(pos, 0));
    fs.read(&sequence[std::max(-pos, 0)], num_chars);
    fs.close();
  }

  // Convert text sequence to binary vector
  for (int i = 0; i < m_sequence_length; ++i) {
    auto current_char = (int) sequence[i];
    if (current_char < 0 || current_char >= 128) {
      current_char = 0;
    }
    Y(128 * i + current_char, mb_idx) = DataType(1);
  }

  return true;
}

//===================================================

void ascii_reader::load() {

  // Make sure directory path ends with a slash
  if (m_file_dir.back() != '/') {
    m_file_dir.push_back('/');
  }

  // Get length of data file
  std::ifstream fs(get_file_dir() + get_data_filename(),
                   std::fstream::in | std::fstream::ate);
  m_file_size = fs.tellg();
  fs.close();

  // Reset indices
  m_shuffled_indices.resize(m_file_size + m_sequence_length);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  if (is_master()) {
    std::cerr << "calling select_subset_of_data; m_shuffled_indices.size: " <<
      m_shuffled_indices.size() << std::endl;
  }
  select_subset_of_data();

}

}  // namespace lbann
