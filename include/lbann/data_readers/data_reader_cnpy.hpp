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
// lbann_data_reader_cnpy .hpp .cpp - generic_data_reader class for numpy dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_CNPY_HPP
#define LBANN_DATA_READER_CNPY_HPP

#include "data_reader.hpp"
#include <cnpy.h>

namespace lbann {
class cnpy_reader : public generic_data_reader {
 public:
  cnpy_reader(int batchSize, bool shuffle = true);
  cnpy_reader(const cnpy_reader& source);
  ~cnpy_reader();

  cnpy_reader& operator=(const cnpy_reader& source);

  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);
  void load();

  int get_linearized_data_size() const {
    return m_num_features;
  }

 private:
  int m_num_features;
  int m_num_samples;
  cnpy::NpyArray m_data;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_CNPY_HPP
