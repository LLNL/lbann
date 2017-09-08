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
// lbann_data_reader_numpy .hpp .cpp - generic_data_reader class for numpy dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_NUMPY_HPP
#define LBANN_DATA_READER_NUMPY_HPP

#include "data_reader.hpp"
#include <cnpy.h>

namespace lbann {

/**
 * Data reader for data stored in numpy (.npy) files.
 * This assumes that the zero'th axis is the sample axis and that all subsequent
 * axes can be flattened to form a sample.
 * This does not support fetching labels from the same file; labels must be
 * provided by some other means and composed with this data reader.
 */
class numpy_reader : public generic_data_reader {
 public:
  numpy_reader(int batch_size, bool shuffle = true);
  numpy_reader(const numpy_reader& source);
  numpy_reader& operator=(const numpy_reader& source);
  ~numpy_reader();

  numpy_reader* copy() const { return new numpy_reader(*this); }

  void load();

  int get_num_labels() const {
    throw lbann_exception("numpy_reader: labels not supported");
  }
  int get_linearized_data_size() const { return m_num_features; }
  int get_linearized_label_size() const {
    throw lbann_exception("numpy_reader: labels not supported");
  }
  const std::vector<int> get_data_dims() const {
    return std::vector<int>(m_data.shape.begin() + 1,
                            m_data.shape.end());
  }

 protected:
  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);

  /// Number of samples.
  int m_num_samples;
  /// Number of features in each sample.
  int m_num_features;
  /// Underlying numpy data.
  cnpy::NpyArray m_data;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_NUMPY_HPP
