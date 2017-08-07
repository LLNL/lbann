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
// lbann_numpy_reader .hpp .cpp
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_numpy.hpp"
#include <stdio.h>
#include <string>
#include <cnpy.h>

namespace lbann {

numpy_reader::numpy_reader(int batch_size, bool shuffle)
  : generic_data_reader(batch_size, shuffle), m_num_samples(0),
    m_num_features(0) {}

numpy_reader::numpy_reader(const numpy_reader& source) :
  generic_data_reader((const generic_data_reader&) source),
  m_num_samples(source.m_num_samples),
  m_num_features(source.m_num_features),
  m_data(source.m_data) {
  int n = m_num_features * m_num_samples * m_data.word_size;
  m_data.data = new char[n];
  memcpy(m_data.data, source.m_data.data, n);
}

numpy_reader& numpy_reader::operator=(const numpy_reader& source) {
  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  generic_data_reader::operator=(source);

  this->m_num_features = source.m_num_features;
  this->m_num_samples = source.m_num_samples;
  this->m_data = source.m_data;
  int n = m_num_features * m_num_samples * m_data.word_size;
  m_data.data = new char[n];
  memcpy(m_data.data, source.m_data.data, n);
  return *this;
}

numpy_reader::~numpy_reader() {
  m_data.destruct();
}

void numpy_reader::load() {
  std::string infile = get_data_filename();
  // Ensure the file exists.
  std::ifstream ifs(infile);
  if (!ifs) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " numpy_reader::load() - can't open file : " + infile);
  }
  ifs.close();

  m_data = cnpy::npy_load(infile);
  m_num_samples = m_data.shape[0];
  m_num_features = std::accumulate(
    m_data.shape.begin() + 1, m_data.shape.end(), (unsigned) 1,
    std::multiplies<unsigned>());

  // Ensure we understand the word size.
  if (!(m_data.word_size == 4 || m_data.word_size == 8)) {
    throw lbann_exception(
      "numpy_reader: word size " + std::to_string(m_data.word_size) +
      " not supported");
  }
  // Fortran order not yet supported.
  if (m_data.fortran_order) {
    throw lbann_exception(
      "numpy_reader: fortran order not supported");
  }

  // Reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

bool numpy_reader::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  if (m_data.word_size == 4) {
    float *data = ((float *) m_data.data) + data_id * m_num_features;
    for (int j = 0; j < m_num_features; j++) {
      X(j, mb_idx) = data[j];
    }
  } else if (m_data.word_size == 8) {
    double *data = ((double *) m_data.data) + data_id * m_num_features;
    for (int j=0; j<m_num_features; j++) {
      X(j, mb_idx) = data[j];
    }
  }
  return true;
}

}  // namespace lbann
