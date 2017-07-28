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

#ifndef LBANN_DATA_READER_CSV_HPP
#define LBANN_DATA_READER_CSV_HPP

#include "data_reader.hpp"
#include "image_preprocessor.hpp"

namespace lbann {

/**
 * Data reader for CSV (and similar) files.
 * This will parse a header to determine how many columns of data there are, and
 * will return each row split based on a separator. This does not handle quotes
 * or escape sequences. The label column is by default converted to an integer.
 * @note This does not currently support comments or blank lines.
 */
class csv_reader : public generic_data_reader {
 public:
  /**
   * @param label_col The column containing labels; -1 for the last column.
   * @param separator The separator between columns, default ','.
   * @param skip_cols The number of columns to skip (from the left), default 0.
   * If columns are skipped, the label column is calculated after that occurs.
   */
  csv_reader(int batch_size, int label_col = -1, char separator = ',',
             int skip_cols = 0, bool shuffle = true);
  csv_reader(const csv_reader&) = default;
  csv_reader& operator=(const csv_reader&) = default;
  ~csv_reader();

  csv_reader* copy() const { return new csv_reader(*this); }

  /**
   * Supply a custom transform to convert an input string to a numerical value.
   * @param col The column to apply this transform to; do not account for
   * skipped columns.
   * @param f The transform to apply.
   */
  void set_column_transform(int col,
                            std::function<DataType(const std::string&)> f) {
    m_col_transforms[col] = f;
  }

  /**
   * Supply a custom transform to convert the label column to an integer.
   * Note that the label should be an integer starting from 0.
   */ 
  void set_label_transform(std::function<int(const std::string&)> f) {
    m_label_transform = f;
  }

  /**
   * This parses the header of the CSV to determine column information.
   */
  void load();

  int get_linearized_data_size() const {
    // Account for label and skipped columns.
    return m_num_cols - 1 - m_skip_cols;
  }
  int get_linearized_label_size() const {
    return m_num_labels;
  }
  const std::vector<int> get_data_dims() const {
    return {get_linearized_data_size()};
  }

// protected:
  /**
   * Fetch the data associated with data_id.
   * Note this does *not* normalize the data.
   */
  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);
  /// Fetch the label associated with data_id.
  bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid);

  /** Return a raw line from the CSV file. */
  std::string fetch_raw_line(int data_id);
  /**
   * Return the parsed CSV line. This does not extract the label.
   */
  std::vector<DataType> fetch_line(int data_id);
  /**
   * Return the parsed CSV line and the label.
   * The label is not present in the vector.
   */
  std::pair<std::vector<DataType>, DataType> fetch_line_and_label(int data_id);

  /// String value that separates data.
  char m_separator;
  /// Number of columns (from the left) to skip.
  int m_skip_cols;
  /// Column containing label data.
  int m_label_col;
  /// Number of columns (including the label column).
  int m_num_cols;
  /// Number of samples.
  int m_num_samples;
  /// Number of label classes.
  int m_num_labels;
  /// Input file streams (per-thread).
  std::vector<std::ifstream*> m_ifstreams;
  /**
   * Index mapping lines (samples) to their start offset within the file.
   * This excludes the header, but includes a final entry indicating the length
   * of the file.
   */
  std::vector<std::streampos> m_index;
  /// Store labels.
  std::vector<int> m_labels;
  /// Per-column transformation functions.
  std::unordered_map<int, std::function<DataType(const std::string&)>>
    m_col_transforms;
  /// Label transform function that converts to an int.
  std::function<int(const std::string&)> m_label_transform =
    [] (const std::string& s) -> int { return std::stoi(s); };
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_CSV_HPP
