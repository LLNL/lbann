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
// lbann_data_generator .hpp .cpp - Synthetic Data Generator
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_GENERATOR_HPP
#define LBANN_DATA_GENERATOR_HPP

#include "data_reader.hpp"
#include "image_preprocessor.hpp"

namespace lbann
{
class DataGenerator : public DataReader
{
public:
  DataGenerator(Int num_samples, Int width, Int height, Int batchSize);
  DataGenerator(const DataGenerator& source);
  ~DataGenerator();

  int fetch_data(Mat& X);
  int fetch_label(Mat& Y) { return 0; }

  int getDataWidth() { return m_data_width; }
  int getDataHeight() { return m_data_height; }
  int get_linearized_data_size() { return m_data_width * m_data_height; }
  int get_linearized_label_size() { return 0; }

  void load();

  DataGenerator& operator=(const DataGenerator& source);

private:

  Int m_num_samples;
  Int m_data_width;
  Int m_data_height;
  StarMat m_data;
};

}  // namespace lbann

#endif  // LBANN_DATA_GENERATOR_HPP
