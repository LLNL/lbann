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
// data_reader_pilot2_molecular .hpp .cpp - data reader for Pilot 2 molecular data
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_PILOT2_MOLECULAR_HPP
#define LBANN_DATA_READER_PILOT2_MOLECULAR_HPP

#include "data_reader.hpp"
#include <cnpy.h>

namespace lbann {

/**
 * Data reader for loading Pilot 2 molecular data.
 */
class pilot2_molecular_reader : public generic_data_reader {
 public:
  pilot2_molecular_reader(int num_neighbors, int batch_size,
                          bool shuffle = true);
  pilot2_molecular_reader(const pilot2_molecular_reader&);
  pilot2_molecular_reader& operator=(const pilot2_molecular_reader&);
  ~pilot2_molecular_reader() {}
  pilot2_molecular_reader* copy() const {
    return new pilot2_molecular_reader(*this);
  }

  void load();

  int get_linearized_data_size() const {
    return m_num_features * (m_num_neighbors + 1);
  }
  const std::vector<int> get_data_dims() const {
    return {m_num_neighbors, (int) m_features.shape[2],
        (int) m_features.shape[3]};
  }
 protected:
  /// Fetch a molecule and its neighbors.
  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);
  /// Fetch molecule data_id into X at molecule offset idx.
  void fetch_molecule(Mat& X, int data_id, int idx, int mb_idx);

  /// Return the frame data_id is in.
  int get_frame(int data_id) const {
    return data_id / m_num_samples_per_frame;
  }

  /// Number of samples.
  int m_num_samples = 0;
  /// Number of features in each sample.
  int m_num_features = 0;
  /// Number of samples in each frame (assume constant across all frames).
  int m_num_samples_per_frame = 0;
  // Number of neighbors to fetch for each molecule.
  int m_num_neighbors;
  /// Molecular features.
  cnpy::NpyArray m_features;
  /// Neighbor information (adjacency matrix).
  cnpy::NpyArray m_neighbors;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_PILOT2_MOLECULAR_HPP
