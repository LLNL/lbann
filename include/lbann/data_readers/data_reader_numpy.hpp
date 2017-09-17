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
// lbann_data_reader_numpy .hpp .cpp - generic_data_reader class for numpy dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_NUMPY_HPP
#define LBANN_DATA_READER_NUMPY_HPP

#include "data_reader.hpp"
#include <cnpy.h>

namespace lbann {

/**
 * Data reader for data stored in numpy (.npy) files.
 * This assumes that the zero'th axis is the sample axis and that all subsequent
 * axes can be flattened to form a sample.
 * This supports fetching labels, but only from the last column. (This can be
 * relaxed if necessary.) Ditto responses.
 */
class numpy_reader : public generic_data_reader {
 public:
  numpy_reader(int batch_size, bool shuffle = true);
  // These need to be explicit because of some issue with the cnpy copy
  // constructor/assignment operator not linking correctly otherwise.
  numpy_reader(const numpy_reader&);
  numpy_reader& operator=(const numpy_reader&);
  ~numpy_reader() {}

  numpy_reader* copy() const { return new numpy_reader(*this); }

  /// Set whether to fetch labels.
  void set_has_labels(bool b) { m_has_labels = b; }
  /// Set whether to fetch responses.
  void set_has_responses(bool b) { m_has_responses = b; }

  void load();

  int get_num_labels() const { return m_num_labels; }
  int get_linearized_data_size() const { return m_num_features; }
  int get_linearized_label_size() const { return m_num_labels; }
  const std::vector<int> get_data_dims() const {
    std::vector<int> dims(m_data.shape.begin() + 1,
                          m_data.shape.end());
    if (m_has_labels || m_has_responses) {
      dims.back() -= 1;
    }
    return dims;
  }

 protected:
  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);
  bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid);
  bool fetch_response(Mat& Y, int data_id, int mb_idx, int tid);

  /// Number of samples.
  int m_num_samples = 0;
  /// Number of features in each sample.
  int m_num_features = 0;
  /// Number of label classes.
  int m_num_labels = 0;
  /// Whether to fetch a label from the last column.
  bool m_has_labels = true;
  /// Whether to fetch a response from the last column.
  bool m_has_responses = false;
  /**
   * Underlying numpy data.
   * Note raw data is managed with shared smart pointer semantics (relevant
   * for copying).
   */
  cnpy::NpyArray m_data;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_NUMPY_HPP
