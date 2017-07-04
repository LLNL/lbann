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
#include <stdio.h>
#include <string>

using namespace std;
using namespace El;



lbann::data_reader_synthetic::data_reader_synthetic(int batch_size, int num_samples, int num_features, bool shuffle)
  : generic_data_reader(batch_size, shuffle) {
  m_num_samples = num_samples;
  m_num_features = num_features;
}

//copy constructor
lbann::data_reader_synthetic::data_reader_synthetic(const data_reader_synthetic& source)
  : generic_data_reader((const generic_data_reader&) source),
    m_num_samples(source.m_num_samples), m_num_features(source.m_num_features)
{ }

lbann::data_reader_synthetic::~data_reader_synthetic() {

}



//fetch one MB of data
int lbann::data_reader_synthetic::fetch_data(Mat& X) {
  if(!generic_data_reader::position_valid()) {
    return 0;
  }

  int current_batch_size = getm_batch_size();
  //@todo: generalize to take different data distribution/generator
  El::Gaussian(X, m_num_features, current_batch_size, DataType(0), DataType(1));

  return current_batch_size;
}



void lbann::data_reader_synthetic::load() {
  //set indices/ number of features
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);

  select_subset_of_data();
}

lbann::data_reader_synthetic& lbann::data_reader_synthetic::operator=(const data_reader_synthetic& source) {

  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  // Call the parent operator= function
  generic_data_reader::operator=(source);


  this->m_num_samples = source.m_num_samples;
  this->m_num_features = source.m_num_features;

  return *this;
}
