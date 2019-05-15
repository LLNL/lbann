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
// data_reader_merge_features .hpp .cpp - Merge features from multiple data readers
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_MERGE_FEATURES_HPP
#define LBANN_DATA_READER_MERGE_FEATURES_HPP

#include "compound_data_reader.hpp"

namespace lbann {

/**
 * Data reader for merging multiple data readers.
 * This can take any positive number of data readers, which will be concatenated
 * in the order provided to provide the data, and a single data reader to
 * provide the label. This data reader uses the fetch_datum method of its
 * subsidiary data readers to fetch all data, including the labels.
 * label data reader is optional
 */
class data_reader_merge_features : public generic_compound_data_reader {
 public:
  data_reader_merge_features(std::vector<generic_data_reader*> data_readers,
                             generic_data_reader *label_reader = nullptr,
                             bool shuffle = true);
  data_reader_merge_features(const data_reader_merge_features&);
  data_reader_merge_features& operator=(const data_reader_merge_features&);
  ~data_reader_merge_features() override;
  data_reader_merge_features* copy() const override {
    return new data_reader_merge_features(*this);
  }

  std::string get_type() const override {
    return "data_reader_merge_features";
  }

  /// Call load on the subsidiary data readers.
  void load() override;

  int get_num_labels() const override { return m_label_reader->get_num_labels(); }
  int get_linearized_data_size() const override { return m_data_size; }
  int get_linearized_label_size() const override {
    return m_label_reader->get_linearized_label_size();
  }
  const std::vector<int> get_data_dims() const override {
    // Todo: Can we merge the dimensions of each reader sensibly?
    return {get_linearized_data_size()};
  }

 protected:
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;

  /// Reader providing label data.
  generic_data_reader *m_label_reader;
  /// Sum of the size of data from all the data readers.
  int m_data_size = 0;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_MERGE_FEATURES_HPP
