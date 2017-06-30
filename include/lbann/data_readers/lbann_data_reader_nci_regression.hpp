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
// lbann_data_reader_nci_regression .hpp .cpp - generic_data_reader class for National Cancer Institute (NCI) dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_NCI_REGRESSION_HPP_INCLUDED
#define LBANN_DATA_READER_NCI_REGRESSION_HPP_INCLUDED

#include "lbann_data_reader.hpp"



namespace lbann {
//@todo rewrite data_reader class to follow coding convention
class data_reader_nci_regression : public generic_data_reader {
 private:
  //@todo add response mode {binary,ternary, continuous}
  int m_num_responses;
  size_t m_num_samples; //rows
  size_t m_num_features; //cols
  std::vector<DataType> m_responses;
  std::vector<std::streampos> m_index_map; // byte offset of each line in the input file
  std::string m_infile; //input file name

 public:
  data_reader_nci_regression(int batchSize, bool shuffle=true);
  //data_reader_nci_regression(const data_reader_nci_regression& source); //copy constructor
  //data_reader_nci_regression& operator=(const data_reader_nci_regression& source); //assignment operator
  ~data_reader_nci_regression(void);

  int fetch_data(Mat& X);
  int fetch_response(Mat& Y);
  int get_num_labels(void) const {
    return m_num_responses;  //@todo; check if used
  }

  void load(void);

  size_t get_num_samples(void) const {
    return m_num_samples;
  }
  size_t get_num_features(void) const {
    return m_num_features;
  }

  int get_linearized_data_size(void) const {
    return m_num_features;
  }
  int get_linearized_response_size(void) const {
    return m_num_responses;
  }
  const std::vector<int> get_data_dims(void) const {
    return {static_cast<int>(m_num_features)};
  }

};

}

#endif // LBANN_DATA_READER_NCI_REGRESSION_HPP_INCLUDED
