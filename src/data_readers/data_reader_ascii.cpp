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
// lbann_data_reader_ascii .hpp .cpp - generic_data_reader class for ASCII text files
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_ascii.hpp"
#include <stdio.h>
namespace lbann {

ascii_reader::ascii_reader(int sequence_length, bool shuffle)
  : generic_data_reader(shuffle), m_sequence_length(sequence_length) {}

bool ascii_reader::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {

  // Get file
  const int file_id = (std::upper_bound(m_file_indices.begin(),
                                        m_file_indices.end(),
                                        data_id)
                       - m_file_indices.begin() - 1);

  // Get text sequence from file
  const int file_length = (m_file_indices[file_id+1]
                           - m_file_indices[file_id]
                           - m_sequence_length);
  const int pos = data_id - m_file_indices[file_id] - m_sequence_length;
  const int num_chars = (std::min(pos + m_sequence_length, file_length)
                         - std::max(pos, 0));
  std::vector<char> sequence(m_sequence_length, 0);
  if (num_chars > 0) {
    std::ifstream fs(get_file_dir() + m_file_list[file_id],
                     std::fstream::in);
    fs.seekg(pos);
    fs.read(&sequence[std::max(-pos, 0)], num_chars);
    fs.close();
  }

  // Convert text sequence to binary vector
  for (int i = 0; i < m_sequence_length; ++i) {
    int current_char = (int) sequence[i];
    if (current_char < 0 || current_char >= 128) {
      current_char = 0;
    }
    X(128 * i + current_char, mb_idx) = DataType(1);
  }

  return true;
}

bool ascii_reader::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {

  // Get file
  const int file_id = (std::upper_bound(m_file_indices.begin(),
                                        m_file_indices.end(),
                                        data_id)
                       - m_file_indices.begin() - 1);

  // Get text sequence from file
  const int file_length = (m_file_indices[file_id+1]
                           - m_file_indices[file_id]
                           - m_sequence_length);
  const int pos = data_id - m_file_indices[file_id] - m_sequence_length + 1;
  const int num_chars = (std::min(pos + m_sequence_length, file_length)
                         - std::max(pos, 0));
  std::vector<char> sequence(m_sequence_length, 0);
  if (num_chars > 0) {
    std::ifstream fs(get_file_dir() + m_file_list[file_id],
                     std::fstream::in);
    fs.seekg(pos);
    fs.read(&sequence[std::max(-pos, 0)], num_chars);
    fs.close();
  }

  // Convert text sequence to binary vector
  for (int i = 0; i < m_sequence_length; ++i) {
    int current_char = (int) sequence[i];
    if (current_char < 0 || current_char >= 128) {
      current_char = 0;
    }
    Y(128 * i + current_char, mb_idx) = DataType(1);
  }

  return true;
}

//===================================================

void ascii_reader::load() {
  std::ifstream fs;

  // Make sure directory path ends with a slash
  if (m_file_dir.back() != '/') {
    m_file_dir.push_back('/');
  }

  // Get list of files
  fs.open(get_file_dir() + get_data_filename(), std::fstream::in);
  while (fs.good()) {
    std::string file;
    std::getline(fs, file);
    if (file.size() > 0) {
      m_file_list.push_back(file);
    }
  }
  fs.close();
  
  // Get length of each file
  m_file_indices.resize(1, 0);
  for (const std::string &file : m_file_list) {
    fs.open(get_file_dir() + file, std::fstream::in | std::fstream::ate);
    const int file_size = fs.tellg();
    m_file_indices.push_back(m_file_indices.back()
                             + file_size + m_sequence_length);
    fs.close();
  }

  // Reset indices
  m_shuffled_indices.resize(m_file_indices.back());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  if (is_master()) {
    std::cerr << "calling select_subset_of_data; m_shuffled_indices.size: " <<
      m_shuffled_indices.size() << std::endl;
  }
  select_subset_of_data();

}

}  // namespace lbann
