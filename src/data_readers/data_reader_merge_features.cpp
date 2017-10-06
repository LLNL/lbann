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

#include "lbann/data_readers/data_reader_merge_features.hpp"

namespace lbann {

data_reader_merge_features::data_reader_merge_features(
  int batch_size, std::vector<generic_data_reader*> data_readers,
  generic_data_reader *label_reader, bool shuffle) :
  generic_compound_data_reader(batch_size, data_readers, shuffle),
  m_label_reader(label_reader) {}

data_reader_merge_features::data_reader_merge_features(
  const data_reader_merge_features& other) :
  generic_compound_data_reader(other),
  m_data_size(other.m_data_size) {
  m_label_reader = other.m_label_reader->copy();
}

data_reader_merge_features& data_reader_merge_features::operator=(
  const data_reader_merge_features& other) {
  generic_compound_data_reader::operator=(other);
  m_data_size = other.m_data_size;
  if (m_label_reader) {
    delete m_label_reader;
  }
  m_label_reader = other.m_label_reader->copy();
  return *this;
}

data_reader_merge_features::~data_reader_merge_features() {
  delete m_label_reader;
}

void data_reader_merge_features::load() {
  // Load each data reader separately.
  for (auto&& reader : m_data_readers) {
    reader->load();
    m_data_size += reader->get_linearized_data_size();
  }
  // Verify the readers have the same number of samples.
  int num_samples = m_data_readers[0]->get_num_data();
  for (auto&& reader : m_data_readers) {
    if (num_samples != reader->get_num_data()) {
      throw lbann_exception(
        "data_reader_merge_features: data readers do not have the same amount of data");
    }
  }
  m_label_reader->load();
  // Reset indices.
  m_shuffled_indices.resize(num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

bool data_reader_merge_features::fetch_datum(Mat& X, int data_id, int mb_idx,
                                             int tid) {
  int start = 0;
  for (auto&& reader : m_data_readers) {
    auto X_view = X(El::IR(start, start + reader->get_linearized_data_size()),
                    El::ALL);
    reader->fetch_datum(X_view, data_id, mb_idx, tid);
    start += reader->get_linearized_data_size();
  }
  return true;
}

bool data_reader_merge_features::fetch_label(Mat& Y, int data_id, int mb_idx,
                                             int tid) {
  return m_label_reader->fetch_datum(Y, data_id, mb_idx, tid);
}

bool data_reader_merge_features::fetch_response(Mat& Y, int data_id, int mb_idx,
                                                int tid) {
  return m_label_reader->fetch_datum(Y, data_id, mb_idx, tid);
}

}  // namespace lbann
