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
// lbann_data_reader_synthetic .hpp .cpp - generic_data_reader class for synthetic (unit testing) data
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_synthetic.hpp"
#include "lbann/utils/random.hpp"
#include <cstdio>
#include <string>

namespace lbann {

namespace {

void fill_matrix(CPUMat& mat) {
  std::normal_distribution<DataType> dist(DataType(0), DataType(1));
  auto& gen = get_fast_io_generator();
  const El::Int height = mat.Height();  // Width is 1.
  DataType * __restrict__ buf = mat.Buffer();
  for (El::Int i = 0; i < height; ++i) {
    buf[i] = dist(gen);
  }
}

}  // anonymous namespace

data_reader_synthetic::data_reader_synthetic(int num_samples, int num_features,
                                             bool shuffle)
  : data_reader_synthetic(num_samples, {num_features}, 0, shuffle) {}

data_reader_synthetic::data_reader_synthetic(int num_samples,
                                             std::vector<int> dims,
                                             int num_labels, bool shuffle)
  : generic_data_reader(shuffle), m_num_samples(num_samples),
    m_num_labels(num_labels), m_dimensions(dims) {}

data_reader_synthetic::data_reader_synthetic(int num_samples,
                                             std::vector<int> dims,
                                             std::vector<int> response_dims,
                                             bool shuffle)
  : generic_data_reader(shuffle), m_num_samples(num_samples),
    m_num_labels(0), m_dimensions(dims), m_response_dimensions(response_dims) {}

bool data_reader_synthetic::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  auto X_v = El::View(X, El::ALL, El::IR(mb_idx, mb_idx + 1));
  fill_matrix(X_v);
  return true;
}

bool data_reader_synthetic::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  if (m_num_labels == 0) {
    LBANN_ERROR("Synthetic data reader does not have labels");
  }
  Y.Set(fast_rand_int(get_fast_io_generator(), m_num_labels), mb_idx, 1);
  return true;
}

bool data_reader_synthetic::fetch_response(CPUMat& Y, int data_id, int mb_idx) {
  if (m_response_dimensions.empty()) {
    LBANN_ERROR("Synthetic data reader does not have responses");
  }
  auto Y_v = El::View(Y, El::ALL, El::IR(mb_idx, mb_idx + 1));
  fill_matrix(Y_v);
  return true;
}

void data_reader_synthetic::load() {
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

}  // namespace lbann
