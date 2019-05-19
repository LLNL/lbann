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
// lbann_data_generator .hpp .cpp - Synthetic Data Generator
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_generator.hpp"
#include "lbann/utils/random.hpp"
#include <stdio.h>

lbann::DataGenerator::DataGenerator(Int num_samples, Int width, Int height, Int batchSize)
  : DataReader(batchSize, true)
{
  m_num_samples = num_samples;
  m_data_width = width;
  m_data_height = height;
}

lbann::DataGenerator::DataGenerator(const DataGenerator& source)
  : DataReader((const DataReader&) source),
    m_data_width(source.m_data_width), m_data_height(source.m_data_height)
{
  // No need to deallocate data on a copy constuctor

  //  clone_image_data(source);
}

lbann::DataGenerator::~DataGenerator()
{
  //  this->free();
}

void lbann::DataGenerator::load() {
  ShuffledIndices.clear();
  ShuffledIndices.resize(m_num_samples);
  for (size_t n = 0; n < ShuffledIndices.size(); n++) {
    ShuffledIndices[n] = n;
  }
  uniform_fill_procdet(m_data, get_linearized_data_size(), m_num_samples, 128, 128);
}

int lbann::DataGenerator::fetch_data(Mat& X)
{
  if(!DataReader::position_valid()) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Data Generator load error: !position_valid";
    throw lbann_exception(err.str());
  }

  int current_batch_size = getBatchSize();

  int n = 0;
  for (n = CurrentPos; n < CurrentPos + current_batch_size; n++) {
    if (n >= (int)ShuffledIndices.size())
      break;

    int k = n - CurrentPos;
    int index = ShuffledIndices[n];

    for (int p = 0; p < get_linearized_data_size(); p++) {
      X.Set(p, k, m_data.GetLocal(p, index));
    }
  }

  return (n - CurrentPos);
}

// Assignment operator
lbann::DataGenerator& lbann::DataGenerator::operator=(const DataGenerator& source)
{
  // check for self-assignment
  if (this == &source)
    return *this;

  // Call the parent operator= function
  DataReader::operator=(source);

  this->m_data_width = source.m_data_width;
  this->m_data_height = source.m_data_height;

  return *this;
}
