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
////////////////////////////////////////////////////////////////////////////////

#ifndef __DATA_STORE_CSV_HPP__
#define __DATA_STORE_CSV_HPP__

#include "lbann/data_store/generic_data_store.hpp"
#include <unordered_map>

namespace lbann {

class csv_reader;
class data_store_merge_features;

/**
 * todo
 */

class data_store_csv : public generic_data_store {
 public:

  //! ctor
  data_store_csv(generic_data_reader *reader, model *m);

  //! copy ctor
  data_store_csv(const data_store_csv&) = default;

  //! operator=
  data_store_csv& operator=(const data_store_csv&) = default;

  data_store_csv * copy() const override { return new data_store_csv(*this); }

  //! dtor
  ~data_store_csv() override;

  void get_data_buf_DataType(int data_id, std::vector<DataType> *&buf) override;

  void setup() override;

  // set shuffled indices without calling exchange_data
  void set_shuffled_indices(const std::vector<int> *indices) override {
    m_shuffled_indices = indices;
  }

protected :

  friend data_store_merge_features;

  csv_reader *m_csv_reader;

  /// size of the vectors that are returned by 
  /// reader->fetch_line_label_response(data_id)
  int m_vector_size;

  /// maps: shuffled index to offset in owner's data store
  std::unordered_map<int, size_t> m_offset_mapping;

  /// buffers for data that will be passed to the data reader's fetch_datum method
  std::unordered_map<int, std::vector<DataType>> m_my_minibatch_data;

  /// retrive data needed for passing to the data reader for the next epoch
  void exchange_data() override;
  /// returns, in "indices," the set of indices that processor "p"
  /// needs for the next epoch. Called by exchange_data
  void get_indices(std::unordered_set<int> &indices, int p);

  /// returns, in "indices," the subset of indices that processor "p"
  /// needs for the next epoch and that this processor owns. 
  /// Called by exchange_data
  void get_my_indices(std::unordered_set<int> &indices, int p);

  /// will contain the data that this processor owns; 
  /// Maps a global index to its associated data
  std::map<int, std::vector<DataType>> m_data;
  //std::unordered_map<int, std::vector<DataType>> m_data;

  /// fills in m_data (the data store)
  void populate_datastore();
};

}  // namespace lbann

#endif  // __DATA_STORE_CSV_HPP__
