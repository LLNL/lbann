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
// lbann_cnpy_reader .hpp .cpp
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_cnpy.hpp"
#include <stdio.h>
#include <string>
#include <cnpy.h>

namespace lbann {

cnpy_reader::cnpy_reader(int batchSize, bool shuffle)
  : generic_data_reader(batchSize, shuffle), m_num_features(0), m_num_samples(0) {
}

cnpy_reader::~cnpy_reader() {
  m_data.destruct();
}

int cnpy_reader::fetch_data(Mat& X) {
  if(!generic_data_reader::position_valid()) {
    return 0;
  }
  int current_batch_size = getm_batch_size();

  int n = 0;
  for (n = m_current_pos; n < m_current_pos + current_batch_size; ++n) {
    if (n >= (int)m_shuffled_indices.size()) {
      break;
    }

    int k = n - m_current_pos;
    int index = m_shuffled_indices[n];

    if (m_data.word_size == 4) {
      float *tmp = (float *)m_data.data;
      float *data = tmp + index;
      for (int j=0; j<m_num_features; j++) {
        X.Set(j, k, data[j]);
      }
    } else if (m_data.word_size == 8) {
      double *tmp = (double *)m_data.data;
      double *data = tmp + index;
      for (int j=0; j<m_num_features; j++) {
        X.Set(j, k, data[j]);
      }
    } else {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " unknown word size: " + std::to_string(m_data.word_size) +
        " we only support 4 (float) or 8 (double)");
    }
  }

  return (n - m_current_pos);
}

void cnpy_reader::load() {
  std::string infile = get_data_filename();
  std::ifstream ifs(infile);
  if (!ifs) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " cnpy_reader::load() - can't open file : " + infile);
  }
  ifs.close();

  m_data = cnpy::npy_load(infile.c_str());
  m_num_samples = m_data.shape[0];
  m_num_features = m_data.shape[1];

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  for (size_t n = 0; n < m_shuffled_indices.size(); ++n) {
    m_shuffled_indices[n] = n;
  }

  select_subset_of_data();
}

cnpy_reader::cnpy_reader(const cnpy_reader& source) :
  generic_data_reader((const generic_data_reader&) source), m_num_features(source.m_num_features),
  m_num_samples(source.m_num_samples), m_data(source.m_data) {
  int n = m_num_features * m_num_samples * m_data.word_size;
  m_data.data = new char[n];
  memcpy(m_data.data, source.m_data.data, n);
}

cnpy_reader& cnpy_reader::operator=(const cnpy_reader& source) {

  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  // Call the parent operator= function
  generic_data_reader::operator=(source);

  this->m_num_features = source.m_num_features;
  this->m_num_samples = source.m_num_samples;
  this->m_data = source.m_data;
  int n = m_num_features * m_num_samples * m_data.word_size;
  m_data.data = new char[n];
  memcpy(m_data.data, source.m_data.data, n);
  return *this;
}

}  // namespace lbann
