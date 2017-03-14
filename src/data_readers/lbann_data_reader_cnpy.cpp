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
// lbann_DataReader_cnpy .hpp .cpp 
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_cnpy.hpp"
#include <stdio.h>
#include <string>
#include <cnpy.h>


using namespace std;
using namespace El;

lbann::DataReader_cnpy::DataReader_cnpy(int batchSize, bool shuffle)
  : DataReader(batchSize, shuffle), m_num_features(0), m_num_samples(0)
{
}

lbann::DataReader_cnpy::~DataReader_cnpy()
{
  m_data.destruct();
}


int lbann::DataReader_cnpy::fetch_data(Mat& X)
{
  if(!DataReader::position_valid()) {
    return 0;
  }
  int current_batch_size = getBatchSize();

  int n = 0;
  for (n = CurrentPos; n < CurrentPos + current_batch_size; ++n) {
    if (n >= (int)ShuffledIndices.size())
      break;

    int k = n - CurrentPos;
    int index = ShuffledIndices[n];

    if (m_data.word_size == 4) {
      float *tmp = (float*)m_data.data;
      float *data = tmp + index;
      for (int j=0; j<m_num_features; j++) {
        X.Set(j, k, data[j]);
      }
    } else if (m_data.word_size == 8) {
      double *tmp = (double*)m_data.data;
      double *data = tmp + index;
      for (int j=0; j<m_num_features; j++) {
        X.Set(j, k, data[j]);
      }
    } else {
      stringstream err;
      err << __FILE__ << " " << __LINE__ << " unknown word size: " << m_data.word_size
          << " we only support 4 (float) or 8 (double)";
      throw lbann_exception(err.str());
    }
  }

  return (n - CurrentPos);
}

void lbann::DataReader_cnpy::load()
{
  string infile = get_data_filename();
  ifstream ifs(infile.c_str());
  if (!ifs) { 
    stringstream err;
    err << endl << __FILE__ << " " << __LINE__ 
         << "  DataReader_cnpy::load() - can't open file : " << infile;  
    throw lbann_exception(err.str());
  }
  ifs.close();

  m_data = cnpy::npy_load(infile.c_str());
  m_num_samples = m_data.shape[0];
  m_num_features = m_data.shape[1];

  // reset indices
  ShuffledIndices.clear();
  ShuffledIndices.resize(m_num_samples);
  for (size_t n = 0; n < ShuffledIndices.size(); ++n) {
    ShuffledIndices[n] = n;
  }

  if (has_max_sample_count()) {
    size_t max_sample_count = get_max_sample_count();
    bool firstN = get_firstN();
    load(max_sample_count, firstN);
  } 
  
  else if (has_use_percent()) {
    double use_percent = get_use_percent();
    bool firstN = get_firstN();
    load(use_percent, firstN);
  }
}

lbann::DataReader_cnpy::DataReader_cnpy(const DataReader_cnpy& source) :
  DataReader((const DataReader&) source), m_num_features(source.m_num_features),
  m_num_samples(source.m_num_samples), m_data(source.m_data) {
  int n = m_num_features * m_num_samples * m_data.word_size;
  m_data.data = new char[n];
  memcpy(m_data.data, source.m_data.data, n);
}



lbann::DataReader_cnpy& lbann::DataReader_cnpy::operator=(const DataReader_cnpy& source)
{

  // check for self-assignment
  if (this == &source)
    return *this;

  // Call the parent operator= function
  DataReader::operator=(source);

  this->m_num_features = source.m_num_features;
  this->m_num_samples = source.m_num_samples;
  this->m_data = source.m_data;
  int n = m_num_features * m_num_samples * m_data.word_size;
  m_data.data = new char[n];
  memcpy(m_data.data, source.m_data.data, n);
  return *this;
}

void lbann::DataReader_cnpy::load(size_t max_sample_count, bool firstN)
{
  if(max_sample_count > getNumData() || ((long) max_sample_count) < 0) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: cnpy: data reader load error: invalid number of samples selected";
    throw lbann_exception(err.str());
  }
  select_subset_of_data(max_sample_count, firstN);
}

void lbann::DataReader_cnpy::load(double use_percentage, bool firstN)
{
  size_t max_sample_count = rint(getNumData()*use_percentage);

  if(max_sample_count > getNumData() || ((long) max_sample_count) < 0) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: cnpy: data reader load error: invalid number of samples selected";
    throw lbann_exception(err.str());
  }
  select_subset_of_data(max_sample_count, firstN);
}
