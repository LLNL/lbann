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
// data_reader_merge_features .hpp .cpp - Merge features from multiple data readers
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_MERGE_FEATURES_HPP
#define LBANN_DATA_READER_MERGE_FEATURES_HPP

#include "data_reader.hpp"

namespace lbann {

/**
 * Data reader for merging multiple data readers.
 * This can take any positive number of data readers, which will be concatenated
 * in the order provided to provide the data, and a single data reader to
 * provide the label. This data reader uses the fetch_datum method of its
 * subsidiary data readers to fetch all data, including the labels.
 */
class data_reader_merge_features : public generic_data_reader {
 public:
  data_reader_merge_features(int batch_size,
                             std::vector<generic_data_reader*> data_readers,
                             generic_data_reader *label_reader,
                             bool shuffle = true);
  data_reader_merge_features(const data_reader_merge_features&);
  data_reader_merge_features& operator=(const data_reader_merge_features&);
  ~data_reader_merge_features();
  data_reader_merge_features* copy() const {
    return new data_reader_merge_features(*this);
  }

  /// Call load on the subsidiary data readers.
  void load();

  int get_num_labels() const { return m_label_reader->get_num_labels(); }
  int get_linearized_data_size() const { return m_data_size; }
  int get_linearized_label_size() const {
    return m_label_reader->get_linearized_label_size();
  }
  const std::vector<int> get_data_dims() const {
    // Todo: Can we merge the dimensions of each reader sensibly?
    return {get_linearized_data_size()};
  }
 protected:
  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);
  bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid);
  bool fetch_response(Mat& Y, int data_id, int mb_idx, int tid);

  /// List of readers providing data.
  std::vector<generic_data_reader*> m_data_readers;
  /// Reader providing label data.
  generic_data_reader *m_label_reader;
  /// Sum of the size of data from all the data readers.
  int m_data_size = 0;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_MERGE_FEATURES_HPP
