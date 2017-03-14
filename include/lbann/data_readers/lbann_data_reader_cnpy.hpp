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
// lbann_data_reader_cnpy .hpp .cpp - DataReader class for numpy dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_CNPY_HPP
#define LBANN_DATA_READER_CNPY_HPP

#include "lbann_data_reader.hpp"
#include <cnpy.h>

namespace lbann
{
class DataReader_cnpy : public DataReader
{
public:
  DataReader_cnpy(int batchSize, bool shuffle = true);
  DataReader_cnpy(const DataReader_cnpy& source);
  ~DataReader_cnpy();

  DataReader_cnpy& operator=(const DataReader_cnpy& source);


  int fetch_data(Mat& X);
  void load();

  int get_linearized_data_size() { return m_num_features; }

private:
  int m_num_features;
  int m_num_samples;
  cnpy::NpyArray m_data;
  void load(double use_percentage, bool firstN);
  void load(size_t max_sample_count, bool firstN);
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_CNPY_HPP
