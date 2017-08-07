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
// lbann_data_reader_csv .hpp .cpp - generic_data_reader class for CSV files
////////////////////////////////////////////////////////////////////////////////

#include <unordered_set>
#include "lbann/data_readers/data_reader_csv.hpp"

namespace lbann {

csv_reader::csv_reader(int batch_size, int label_col,
                       bool shuffle)
  : generic_data_reader(batch_size, shuffle), m_label_col(label_col) {}

csv_reader::~csv_reader() {
  for (auto&& ifs : m_ifstreams) {
    delete ifs;
  }
}

void csv_reader::load() {
  // Set up the ifstreams.
  m_ifstreams.resize(omp_get_max_threads());
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    m_ifstreams[i] = new std::ifstream(
      get_data_filename(), std::ios::in | std::ios::binary);
    if (m_ifstreams[i]->fail()) {
      throw lbann_exception(
        "csv_reader: failed to open " + get_data_filename());
    }
  }
  std::ifstream& ifs = *m_ifstreams[0];
  // TODO: Only one (or a subset) of ranks should read this, then distribute
  // the results to avoid every rank reading the same file.
  std::string line;
  // Parse the header to determine how many columns there are.
  // Skip rows if needed.
  skip_rows(ifs, m_skip_rows);
  std::streampos header_start = ifs.tellg();
  // TODO: Skip comment lines.
  if (std::getline(ifs, line)) {
    m_num_cols = std::count(line.begin(), line.end(), m_separator) + 1;
    if (m_skip_cols >= m_num_cols) {
      throw lbann_exception(
        "csv_reader: asked to skip more columns than are present");
    }
    if (m_label_col < 0) {
      // Last column becomes the label column.
      m_label_col = m_num_cols - 1;
    }
    if (m_label_col >= m_num_cols) {
      throw lbann_exception(
        "csv_reader: label column" + std::to_string(m_label_col) +
        " is not present");
    }
  } else {
    throw lbann_exception(
      "csv_reader: failed to read header in " + get_data_filename());
  }
  if (ifs.eof()) {
    throw lbann_exception(
      "csv_reader: reached EOF after reading header");
  }
  // If there was no header, skip back to the beginning.
  if (!m_has_header) {
    ifs.clear();
    ifs.seekg(header_start, std::ios::beg);
  }
  // Construct an index mapping each line (sample) to its offset.
  // TODO: Skip comment lines.
  // Used to count the number of label classes.
  std::unordered_set<int> label_classes;
  m_index.push_back(ifs.tellg());
  int line_num = 0;
  while (std::getline(ifs, line)) {
    ++line_num;
    // Verify the line has the right number of columns.
    if (std::count(line.begin(), line.end(), m_separator) + 1 != m_num_cols) {
      throw lbann_exception(
        "csv_reader: line " + std::to_string(line_num) +
        " does not have right number of entries");
    }
    m_index.push_back(ifs.tellg());
    // Extract the label.
    size_t cur_pos = 0;
    for (int col = 0; col < m_num_cols; ++col) {
      size_t end_pos = line.find_first_of(m_separator, cur_pos);
      if (col == m_label_col) {
        label_classes.insert(
          m_label_transform(line.substr(cur_pos, end_pos - cur_pos)));
        break;
      }
      cur_pos = end_pos + 1;
    }
  }
  if (!ifs.eof()) {
    // If we didn't get to EOF, something went wrong.
    throw lbann_exception(
      "csv_reader: did not reach EOF");
  }
  // Do some simple validation checks on the classes.
  // Ensure the elements begin with 0, and there are no gaps.
  auto minmax = std::minmax_element(label_classes.begin(), label_classes.end());
  if (*minmax.first != 0) {
    throw lbann_exception(
      "csv_reader: classes are not indexed from 0");
  }
  if (*minmax.second != label_classes.size() - 1) {
    throw lbann_exception(
      "csv_reader: label classes are not contiguous");
  }
  m_num_samples = m_index.size() - 1;
  m_num_labels = label_classes.size();
  // End of file offset.
  m_index.push_back(ifs.tellg());
  // Seek back to the beginning.
  ifs.clear();
  ifs.seekg(0, std::ios::beg);
  // Reset indices.
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
  // Allocate space to store labels.
  m_labels.resize(m_num_samples);
}

bool csv_reader::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  auto line = fetch_line_and_label(data_id);
  // Todo: Avoid the unneeded copies.
  m_labels[data_id] = line.second;
  for (size_t i = 0; i < line.first.size(); ++i) {
    X(i, mb_idx) = line.first[i];
  }
  return true;
}

bool csv_reader::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {
  Y(m_labels[data_id], mb_idx) = 1;
  return true;
}

std::vector<DataType> csv_reader::fetch_line(int data_id) {
  std::string line = fetch_raw_line(data_id);
  std::vector<DataType> parsed_line;
  // Note: load already verified that every line is properly formatted.
  size_t cur_pos = 0;  // Current *start* of a column.
  for (int col = 0; col < m_num_cols; ++col) {
    // Note for last column, this returns std::npos, which substr handles.
    size_t end_pos = line.find_first_of(m_separator, cur_pos);
    // Skip columns if needed.
    if (col < m_skip_cols) {
      cur_pos = end_pos + 1;
      continue;
    }
    // Note: This results in a copy, switch to string_view when we can.
    std::string str_val = line.substr(cur_pos, end_pos - cur_pos);
    cur_pos = end_pos + 1;
    DataType val;
    if (m_col_transforms.count(col)) {
      val = m_col_transforms[col](str_val);
    } else {
      // No easy way to parameterize based on DataType, so always use double.
      val = std::stod(str_val);
    }
    parsed_line.push_back(val);
  }
  return parsed_line;
}

std::pair<std::vector<DataType>, DataType> csv_reader::fetch_line_and_label(
  int data_id) {
  std::string line = fetch_raw_line(data_id);
  std::vector<DataType> parsed_line;
  DataType label = 0;
  // Note: load already verified that every line is properly formatted.
  size_t cur_pos = 0;  // Current *start* of a column.
  for (int col = 0; col < m_num_cols; ++col) {
    // Note for last column, this returns std::npos, which substr handles.
    size_t end_pos = line.find_first_of(m_separator, cur_pos);
    // Handle the label.
    if (col == m_label_col) {
      std::string str_val = line.substr(cur_pos, end_pos - cur_pos);
      label = m_label_transform(str_val);
      cur_pos = end_pos + 1;
      continue;
    }
    // Skip columns if needed.
    if (col < m_skip_cols) {
      cur_pos = end_pos + 1;
      continue;
    }
    // Note: This results in a copy, switch to string_view when we can.
    std::string str_val = line.substr(cur_pos, end_pos - cur_pos);
    cur_pos = end_pos + 1;
    DataType val;
    if (m_col_transforms.count(col)) {
      val = m_col_transforms[col](str_val);
    } else {
      // No easy way to parameterize based on DataType, so always use double.
      try {
        val = std::stod(str_val);
      } catch (std::invalid_argument& e) {
        throw lbann_exception(
          "csv_reader: could not convert '" + str_val + "'");
      }
    }
    parsed_line.push_back(val);
  }
  return std::make_pair(parsed_line, label);
}

std::string csv_reader::fetch_raw_line(int data_id) {
  std::ifstream& ifs = *m_ifstreams[omp_get_thread_num()];
  // Seek to the start of this datum's line.
  ifs.seekg(m_index[data_id], std::ios::beg);
  // Compute the length of the line to read, excluding newline.
  std::streamsize cnt = m_index[data_id+1] - m_index[data_id] - 1;
  // Read directly into a string buffer.
  std::string line;
  line.resize(cnt);
  if (!ifs.read(&line[0], cnt)) {
    throw lbann_exception(
      "csv_reader: error on reading " + std::to_string(data_id));
  }
  return line;
}

void csv_reader::skip_rows(std::ifstream& s, int rows) {
  std::string unused;  // Unused buffer for extracting lines.
  for (int i = 0; i < rows; ++i) {
    if (!std::getline(s, unused)) {
      throw lbann_exception("csv_reader: error on skipping rows");
    }
  }
}

}  // namespace lbann
