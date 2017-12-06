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
// lbann_data_reader_synthetic .hpp .cpp - generic_data_reader class for synthetic (unit testing) data
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_synthetic.hpp"
#include <cstdio>
#include <string>

namespace lbann {

data_reader_synthetic::data_reader_synthetic(int num_samples, int num_features, bool shuffle)
  : generic_data_reader(shuffle) {
  m_num_samples = num_samples;
  m_num_features = num_features;
}

/// Generate one datum of the mini-batch
bool data_reader_synthetic::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  Mat X_v;
  El::View(X_v, X, El::ALL, El::IR(mb_idx, mb_idx + 1));
  //@todo: generalize to take different data distribution/generator
  El::Gaussian(X_v, m_num_features, 1, DataType(0), DataType(1));

  return true;
}

void data_reader_synthetic::load() {
  //set indices/ number of features
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);

  select_subset_of_data();
}

}  // namespace lbann
