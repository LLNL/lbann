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
// lbann_data_reader_nci .hpp .cpp - generic_data_reader class for National Cancer Institute (NCI) dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_NCI_HPP
#define LBANN_DATA_READER_NCI_HPP

#include "data_reader.hpp"

#define NCI_HAS_HEADER

namespace lbann {

class data_reader_nci : public generic_data_reader {
 public:
  data_reader_nci(int batchSize, bool shuffle);
  data_reader_nci(int batchSize);
  data_reader_nci(const data_reader_nci& source) = default;
  data_reader_nci& operator=(const data_reader_nci& source) = default;
  ~data_reader_nci() {}
  data_reader_nci* copy() const { return new data_reader_nci(*this); }

  void preprocess_data_source(int tid);
  void postprocess_data_source(int tid);

  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);
  bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid);
  bool fetch_response(Mat& Y, int data_id, int mb_idx, int tid);

  int get_num_labels() const {
    // m_num_responses should be equivalent
    return m_num_labels;  //@todo; check if used
  }

  void load();

  size_t get_num_samples() const {
    return m_num_samples;
  }
  size_t get_num_features() const {
    return m_num_features;
  }
  inline int map_label_2int(const std::string label);

  int get_linearized_data_size() const {
    return m_num_features;
  }
  int get_linearized_label_size() const {
    return m_num_labels;  // m_num_responses should be equivalent
  }
  const std::vector<int> get_data_dims() const {
    return {static_cast<int>(m_num_features)};
  }

 private:
  //@todo add response mode {binary, ternary, continuous}
  int m_num_labels;  //2 for binary response mode
  int m_num_responses;
  size_t m_num_samples; //rows
  size_t m_num_features; //cols
  std::vector<int> m_labels;
  std::vector<DataType> m_responses;
  std::vector<std::streampos> m_index_map; // byte offset of each line in the input file
  std::string m_infile; //input file name
  std::vector<ifstream*> m_ifs;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_NCI_HPP
