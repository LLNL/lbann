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
   * This defaults to using the last column for the label/response.
   */
  csv_reader(int batch_size, bool shuffle = true);
  csv_reader(const csv_reader&);
  csv_reader& operator=(const csv_reader&);
  ~csv_reader();

  csv_reader* copy() const { return new csv_reader(*this); }

  /// Set the label column.
  void set_label_col(int col) { m_label_col = col; }
  /// Set the response column.
  void set_response_col(int col) { m_response_col = col; }
  /// Disable fetching labels.
  void disable_labels(bool b = true) { m_disable_labels = b; }
  /// Enable fetching responses (disabled by default).
  void enable_responses(bool b = false) { m_disable_responses = b; }
  /// Set the column separator (default is ',').
  void set_separator(char sep) { m_separator = sep; }
  /// Set the number of columns (from the left) to skip; default 0.
  void set_skip_cols(int cols) { m_skip_cols = cols; }
  /// Set the number of rows (from the top) to skip; default 0.
  void set_skip_rows(int rows) { m_skip_rows = rows; }
  /// Set whether the CSV file has a header; default true.
  void set_has_header(bool b) { m_has_header = b; }

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
   * Supply a custom transform to convert the response column to a DataType.
   */
  void set_response_transform(std::function<DataType(const std::string&)> f) {
    m_response_transform = f;
  }

  /**
   * This parses the header of the CSV to determine column information.
   */
  void load();

  int get_num_labels() const { return m_num_labels; }
  int get_linearized_data_size() const {
    // Account for label and skipped columns.
    if (m_label_col < m_skip_cols) {
      return m_num_cols - m_skip_cols;
    } else {
      return m_num_cols - 1 - m_skip_cols;
    }
  }
  int get_linearized_label_size() const {
    return m_num_labels;
  }
  const std::vector<int> get_data_dims() const {
    return {get_linearized_data_size()};
  }

 protected:
  /**
   * Fetch the data associated with data_id.
   * Note this does *not* normalize the data.
   */
  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);
  /// Fetch the label associated with data_id.
  bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid);
  /// Fetch the response associated with data_id.
  bool fetch_response(Mat& Y, int data_id, int mb_idx, int tid);

  /** Return a raw line from the CSV file. */
  std::string fetch_raw_line(int data_id);
  /**
   * Return the parsed CSV line. This does not extract the label/response.
   */
  std::vector<DataType> fetch_line(int data_id);
  /**
   * Return the parsed CSV line and store the label and response in the m_labels
   * and m_responses vectors, respectively. The label and response are not
   * present in the vector.
   */
  std::vector<DataType> fetch_line_label_response(int data_id);

  /// Skip rows in an ifstream.
  void skip_rows(std::ifstream& s, int rows);

  /// Initialize the ifstreams vector.
  void setup_ifstreams();

  /// String value that separates data.
  char m_separator = ',';
  /// Number of columns (from the left) to skip.
  int m_skip_cols = 0;
  /// Number of rows to skip.
  int m_skip_rows = 0;
  /// Whether the CSV file has a header.
  bool m_has_header = true;
  /**
   * Column containing labels. -1 is used for the last column.
   * The label column ignores skipped columns, and can be among columns that are
   * skipped.
   */
  int m_label_col = -1;
  /// Column containing responses; functions the same as the label column.
  int m_response_col = -1;
  /// Whether to fetch labels.
  bool m_disable_labels = false;
  /// Whether to fetch responses.
  bool m_disable_responses = true;
  /// Number of columns (including the label column and skipped columns).
  int m_num_cols = 0;
  /// Number of samples.
  int m_num_samples = 0;
  /// Number of label classes.
  int m_num_labels = 0;
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
  /// Store responses.
  std::vector<DataType> m_responses;
  /// Per-column transformation functions.
  std::unordered_map<int, std::function<DataType(const std::string&)>>
    m_col_transforms;
  /// Label transform function that converts to an int.
  std::function<int(const std::string&)> m_label_transform =
    [] (const std::string& s) -> int { return std::stoi(s); };
  /// Response transform function that converts to a DataType.
  std::function<DataType(const std::string&)> m_response_transform =
    [] (const std::string& s) -> DataType { return std::stod(s); };
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_CSV_HPP
