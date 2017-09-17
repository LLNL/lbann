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
// data_reader_merge_samples.hpp .cpp - Merge samples from multiple data readers
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_merge_samples.hpp"

namespace lbann {

data_reader_merge_samples::data_reader_merge_samples(
  int batch_size, std::vector<generic_data_reader*> data_readers,
  bool shuffle) :
  generic_data_reader(batch_size, shuffle),
  m_data_readers(data_readers) {
  if (m_data_readers.empty()) {
    throw lbann_exception(
      "data_reader_merge_samples: data reader list empty");
  }
}

data_reader_merge_samples::data_reader_merge_samples(
  const data_reader_merge_samples& other) :
  generic_data_reader(other),
  m_num_samples_psum(other.m_num_samples_psum) {
  for (auto&& reader : other.m_data_readers) {
    m_data_readers.push_back(reader->copy());
  }
}

data_reader_merge_samples& data_reader_merge_samples::operator=(
  const data_reader_merge_samples& other) {
  generic_data_reader::operator=(other);
  m_num_samples_psum = other.m_num_samples_psum;
  for (auto&& reader : m_data_readers) {
    delete reader;
  }
  m_data_readers.clear();
  for (auto&& reader : other.m_data_readers) {
    m_data_readers.push_back(reader->copy());
  }
  return *this;
}

data_reader_merge_samples::~data_reader_merge_samples() {
  for (auto&& reader : m_data_readers) {
    delete reader;
  }
}

void data_reader_merge_samples::load() {
  // Load each subsidiary data reader.
  for (auto&& reader : m_data_readers) {
    reader->load();
  }
  // Compute the total number of samples and do some sanity checks.
  int num_samples = 0;
  int num_labels = m_data_readers[0]->get_num_labels();
  int data_size = m_data_readers[0]->get_linearized_data_size();
  int label_size = m_data_readers[0]->get_linearized_label_size();
  const std::vector<int> data_dims = m_data_readers[0]->get_data_dims();
  // Prepend a 0 to make things easier.
  m_num_samples_psum.push_back(0);
  for (auto&& reader : m_data_readers) {
    m_num_samples_psum.push_back(reader->get_num_data());
    num_samples += reader->get_num_data();
    // TODO: It would be good to relax this.
    // Large data sets may not have every label in every file.
    if (num_labels != reader->get_num_labels()) {
      throw lbann_exception(
        "data_reader_merge_samples: data readers do not have the same number of labels");
    }
    if (data_size != reader->get_linearized_data_size()) {
      throw lbann_exception(
        "data_reader_merge_samples: data readers do not have the same data size");
    }
    if (label_size != reader->get_linearized_label_size()) {
      throw lbann_exception(
        "data_reader_merge_samples: data readers do not have the same label size");
    }
    if (data_dims != reader->get_data_dims()) {
      throw lbann_exception(
        "data_reader_merge_samples: data readers do not have the same data dims");
    }
  }
  std::partial_sum(m_num_samples_psum.begin(), m_num_samples_psum.end(),
                   m_num_samples_psum.begin());
  // Set up our indices.
  // Note each subsidiary data reader presumably shuffled its indices as well.
  // That's not strictly necessary, but does not impact anything.
  m_shuffled_indices.resize(num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

bool data_reader_merge_samples::fetch_datum(Mat& X, int data_id, int mb_idx,
                                            int tid) {
  // Find the right data reader to delegate to.
  for (size_t i = 0; i < m_data_readers.size(); ++i) {
    if (data_id < m_num_samples_psum[i + 1]) {
      data_id -= m_num_samples_psum[i];
      return m_data_readers[i]->fetch_datum(X, data_id, mb_idx, tid);
    }
  }
  throw lbann_exception(
    "data_reader_merge_samples: do not have data ID " +
    std::to_string(data_id));
}

bool data_reader_merge_samples::fetch_label(Mat& Y, int data_id, int mb_idx,
                                            int tid) {
  // Find the right data reader to delegate to.
  for (size_t i = 0; i < m_data_readers.size(); ++i) {
    if (data_id < m_num_samples_psum[i + 1]) {
      data_id -= m_num_samples_psum[i];
      return m_data_readers[i]->fetch_label(Y, data_id, mb_idx, tid);
    }
  }
  throw lbann_exception(
    "data_reader_merge_samples: do not have data ID " +
    std::to_string(data_id));
}

bool data_reader_merge_samples::fetch_response(Mat& Y, int data_id, int mb_idx,
                                               int tid) {
  // Find the right data reader to delegate to.
  for (size_t i = 0; i < m_data_readers.size(); ++i) {
    if (data_id < m_num_samples_psum[i + 1]) {
      data_id -= m_num_samples_psum[i];
      return m_data_readers[i]->fetch_response(Y, data_id, mb_idx, tid);
    }
  }
  throw lbann_exception(
    "data_reader_merge_samples: do not have data ID " +
    std::to_string(data_id));
}

}  // namespace lbann
