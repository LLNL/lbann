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
// lbann_data_reader_csv .hpp .cpp - generic_data_reader class for CSV files
////////////////////////////////////////////////////////////////////////////////

#include <unordered_set>
#include "lbann/data_readers/data_reader_csv.hpp"
#include "lbann/utils/options.hpp"
#include <omp.h>

namespace lbann {

csv_reader::csv_reader(bool shuffle)
  : generic_data_reader(shuffle) {}

csv_reader::csv_reader(const csv_reader& other) :
  generic_data_reader(other),
  m_separator(other.m_separator),
  m_skip_cols(other.m_skip_cols),
  m_skip_rows(other.m_skip_rows),
  m_has_header(other.m_has_header),
  m_label_col(other.m_label_col),
  m_response_col(other.m_response_col),
  m_disable_labels(other.m_disable_labels),
  m_disable_responses(other.m_disable_responses),
  m_num_cols(other.m_num_cols),
  m_num_samples(other.m_num_samples),
  m_num_labels(other.m_num_labels),
  m_index(other.m_index),
  m_labels(other.m_labels),
  m_responses(other.m_responses),
  m_col_transforms(other.m_col_transforms),
  m_label_transform(other.m_label_transform),
  m_response_transform(other.m_response_transform) {
  if (!other.m_ifstreams.empty()) {
    // Need to set these up again manually.
    setup_ifstreams();
  }
}

csv_reader& csv_reader::operator=(const csv_reader& other) {
  generic_data_reader::operator=(other);
  m_separator = other.m_separator;
  m_skip_cols = other.m_skip_cols;
  m_skip_rows = other.m_skip_rows;
  m_has_header = other.m_has_header;
  m_label_col = other.m_label_col;
  m_response_col = other.m_response_col;
  m_disable_labels = other.m_disable_labels;
  m_disable_responses = other.m_disable_responses;
  m_num_cols = other.m_num_cols;
  m_num_samples = other.m_num_samples;
  m_num_labels = other.m_num_labels;
  m_index = other.m_index;
  m_labels = other.m_labels;
  m_responses = other.m_responses;
  m_col_transforms = other.m_col_transforms;
  m_label_transform = other.m_label_transform;
  m_response_transform = other.m_response_transform;
  if (!other.m_ifstreams.empty()) {
    // Possibly free our current ifstreams, set them up again.
    for (std::ifstream* ifs : m_ifstreams) {
      delete ifs;
    }
    setup_ifstreams();
  }
  return *this;
}

csv_reader::~csv_reader() {
  for (auto&& ifs : m_ifstreams) {
    delete ifs;
  }
}

void csv_reader::load() {
  bool master = m_comm->am_world_master();
  setup_ifstreams();
  std::ifstream& ifs = *m_ifstreams[0];
  const El::mpi::Comm& world_comm = m_comm->get_world_comm();
  // Parse the header to determine how many columns there are.
  // Skip rows if needed.
  if (master) {
    skip_rows(ifs, m_skip_rows);
  }
  m_comm->broadcast<int>(0, m_skip_rows, world_comm);

  //This will be broadcast from root to other procs, and will
  //then be converted to std::vector<int> m_labels; this is because
  //El::mpi::Broadcast<std::streampos> doesn't work
  std::vector<long long> index;

  if (master) {
    std::string line;
    std::streampos header_start = ifs.tellg();
    // TODO: Skip comment lines.
    if (std::getline(ifs, line)) {
      m_num_cols = std::count(line.begin(), line.end(), m_separator) + 1;
      if (m_skip_cols >= m_num_cols) {
        throw lbann_exception(
          "csv_reader: asked to skip more columns than are present");
      }

      if (!m_disable_labels) {
        if (m_label_col < 0) {
          // Last column becomes the label column.
          m_label_col = m_num_cols - 1;
        }
        if (m_label_col >= m_num_cols) {
          throw lbann_exception(
            "csv_reader: label column" + std::to_string(m_label_col) +
            " is not present");
        }
      }

      if (!m_disable_responses) {
        if (m_response_col < 0) {
          // Last column becomes the response column.
          m_response_col = m_num_cols - 1;
        }
        if (m_response_col >= m_num_cols) {
          throw lbann_exception(
            "csv_reader: response column" + std::to_string(m_response_col) +
            " is not present");
        }
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
    index.push_back(ifs.tellg());

    int num_samples_to_use = get_absolute_sample_count();
    int line_num = 0;
    if (num_samples_to_use == 0) {
      num_samples_to_use = -1;
    }
    while (std::getline(ifs, line)) {
      if (line_num == num_samples_to_use) {
        break;
      }
      ++line_num;

      // Verify the line has the right number of columns.
      if (std::count(line.begin(), line.end(), m_separator) + 1 != m_num_cols) {
        throw lbann_exception(
          "csv_reader: line " + std::to_string(line_num) +
          " does not have right number of entries");
      }
      index.push_back(ifs.tellg());
      // Extract the label.
      if (!m_disable_labels) {
        size_t cur_pos = 0;
        for (int col = 0; col < m_num_cols; ++col) {
          size_t end_pos = line.find_first_of(m_separator, cur_pos);
          if (col == m_label_col) {
            int label = m_label_transform(line.substr(cur_pos, end_pos - cur_pos));
            label_classes.insert(label);
            m_labels.push_back(label);
            break;
          }
          cur_pos = end_pos + 1;
        }
      }
      // Possibly extract the response.
      if (!m_disable_responses) {
        size_t cur_pos = 0;
        for (int col = 0; col < m_num_cols; ++col) {
          size_t end_pos = line.find_first_of(m_separator, cur_pos);
          if (col == m_response_col) {
            DataType response = m_response_transform(
              line.substr(cur_pos, end_pos - cur_pos));
            m_responses.push_back(response);
            break;
          }
          cur_pos = end_pos + 1;
        }
      }
    }

    if (!ifs.eof() && num_samples_to_use == 0) {
       //If we didn't get to EOF, something went wrong.
      throw lbann_exception(
        "csv_reader: did not reach EOF");
    }
    if (!m_disable_labels) {
      // Do some simple validation checks on the classes.
      // Ensure the elements begin with 0, and there are no gaps.
      auto minmax = std::minmax_element(label_classes.begin(), label_classes.end());
      if (*minmax.first != 0) {
        throw lbann_exception(
          "csv_reader: classes are not indexed from 0");
      }
      if (*minmax.second != (int) label_classes.size() - 1) {
        throw lbann_exception(
          "csv_reader: label classes are not contiguous");
      }
      m_num_labels = label_classes.size();
    }
    ifs.clear();
  } // if (master)

  m_comm->broadcast<int>(0, m_num_cols, world_comm);
  m_label_col = m_num_cols - 1;

  //bcast the index vector
  m_comm->world_broadcast<long long>(0, index);
  m_num_samples = index.size() - 1;
  if (m_master) std::cerr << "num samples: " << m_num_samples << "\n";

  m_index.reserve(index.size());
  for (auto t : index) {
    m_index.push_back(t);
  }

  //optionally bcast the response vector
  if (!m_disable_responses) {
    m_response_col = m_num_cols - 1;
    m_comm->world_broadcast<DataType>(0, m_responses);
  }

  //optionally bcast the label vector
  if (!m_disable_labels) {
    m_comm->world_broadcast<int>(0, m_labels);
    m_num_labels = m_labels.size();
  }

  // Reset indices.
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

bool csv_reader::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  auto line = fetch_line_label_response(data_id);
  // TODO: Avoid unneeded copies.
  for (size_t i = 0; i < line.size(); ++i) {
    X(i, mb_idx) = line[i];
  }
  return true;
}

bool csv_reader::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  Y(m_labels[data_id], mb_idx) = 1;
  return true;
}

bool csv_reader::fetch_response(CPUMat& Y, int data_id, int mb_idx) {
  Y(0, mb_idx) = m_responses[data_id];
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
    // @TODO Note: This results in a copy, switch to string_view when we can.
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

std::vector<DataType> csv_reader::fetch_line_label_response(
  int data_id) {
  std::string line = fetch_raw_line(data_id);
  std::vector<DataType> parsed_line;
  // Note: load already verified that every line is properly formatted.
  size_t cur_pos = 0;  // Current *start* of a column.
  for (int col = 0; col < m_num_cols; ++col) {
    // Note for last column, this returns std::npos, which substr handles.
    size_t end_pos = line.find_first_of(m_separator, cur_pos);
    // Skip the label, response, and any columns if needed.
    if ((!m_disable_labels && col == m_label_col) ||
        (!m_disable_responses && col == m_response_col) ||
        col < m_skip_cols) {
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
  return parsed_line;
}

std::string csv_reader::fetch_raw_line(int data_id) {
static int n = 0;
  std::ifstream& ifs = *m_ifstreams[omp_get_thread_num()];
  // Seek to the start of this datum's line.
  ifs.seekg(m_index[data_id], std::ios::beg);
  // Compute the length of the line to read, excluding newline.
  std::streamsize cnt = m_index[data_id+1] - m_index[data_id] - 1;
  // Read directly into a string buffer.
  std::string line;
  line.resize(cnt);
  if (!ifs.read(&line[0], cnt)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "csv_reader: error on reading data_id: " << data_id << "\n"
        <<"index.size(): " << m_index.size() << "  m_index["<< data_id
        << ")=" <<m_index[data_id] << "; index[" << data_id+1
        << ")= " << m_index[data_id+1] << "\ncnt: " << cnt << " role: "
        << get_role() << " gcount: " << ifs.gcount() << " n: " << n;
    throw lbann_exception(err.str());
  }
++n;
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

void csv_reader::setup_ifstreams() {
  m_ifstreams.resize(omp_get_max_threads());
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    m_ifstreams[i] = new std::ifstream(
      get_file_dir() + get_data_filename(), std::ios::in | std::ios::binary);
    if (m_ifstreams[i]->fail()) {
      throw lbann_exception(
        "csv_reader: failed to open " + get_file_dir() + get_data_filename());
    }
  }
}

}  // namespace lbann
