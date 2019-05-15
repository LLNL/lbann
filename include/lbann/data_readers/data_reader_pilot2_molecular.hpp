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
  pilot2_molecular_reader(int num_neighbors,
                          int max_neighborhood,
                          bool shuffle = true);
  pilot2_molecular_reader(const pilot2_molecular_reader&) = default;
  pilot2_molecular_reader& operator=(const pilot2_molecular_reader&) = default;
  ~pilot2_molecular_reader() override {}
  pilot2_molecular_reader* copy() const override {
    return new pilot2_molecular_reader(*this);
  }
  std::string get_type() const override {
    return "pilot2_molecular_reader";
  }

  void load() override;

  int get_linearized_data_size() const override {
    return m_num_features * (m_num_neighbors + 1);
  }
  const std::vector<int> get_data_dims() const override {
    return m_shape;
    //return {m_num_neighbors + 1, (int) m_features.shape[2],
    //    (int) m_features.shape[3]};
  }

  /// Data format is:
  /// [Frames (2900), Molecules (3040), Beads (12), ['x', 'y', 'z',
  /// 'CHOL', 'DPPC', 'DIPC', 'Head', 'Tail', 'BL1', 'BL2', 'BL3',
  /// 'BL4', 'BL5', 'BL6', 'BL7', 'BL8', 'BL9', 'BL10', 'BL11',
  /// 'BL12'] (20)]
  template <class T>
  T scale_data(int idx, T datum) {
    idx = idx % 20;
    T scaled_datum = datum;
    if(idx >= 0 && idx <= 2) { /// x,y,z
      scaled_datum /= position_scale_factor;
    }
    if(idx >= 8 && idx <= 19) {
      scaled_datum /= bond_len_scale_factor;
    }
    return scaled_datum;
  }

  /// support for data_store_pilot2_molecular
  float * get_features_4() {
    return m_features.data<float>();
  }
  double * get_features_8() {
    return m_features.data<double>();
  }
  float * get_neighbors_4() {
    return m_neighbors.data<float>();
  }
  double * get_neighbors_8() {
    return m_neighbors.data<double>();
  }


  /// support for data_store_pilot2_molecular
  int get_word_size() const {
    return m_word_size;
  }

  /// support for data_store_pilot2_molecular
  int get_num_neighbors() const {
    return m_num_neighbors;
  }

  /// Return the frame data_id is in.
  /// (made public to support data_store_pilot2_molecular)
  int get_frame(int data_id) const {
    return data_id / m_num_samples_per_frame;
  }

  /// support for data_store_pilot2_molecular
  int get_num_samples_per_frame() const {
    return m_num_samples_per_frame;
  }

  /// support for data_store_pilot2_molecular
  int get_max_neighborhood() const {
    return m_max_neighborhood;
  }

  /// support for data_store_pilot2_molecular
  int get_num_features() const {
    return m_num_features;
  }

  /// support for data_store_pilot2_molecular
  int get_neighbors_data_size() {
    return m_neighbors_data_size;
  }

 protected:
  /// Fetch a molecule and its neighbors.
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  /// Fetch molecule data_id into X at molecule offset idx.
  void fetch_molecule(CPUMat& X, int data_id, int idx, int mb_idx);

  /// Number of samples.
  int m_num_samples = 0;
  /// Number of features in each sample.
  int m_num_features = 0;
  /// Number of samples in each frame (assume constant across all frames).
  int m_num_samples_per_frame = 0;
  // Number of neighbors to fetch for each molecule.
  int m_num_neighbors;
  // Size of the neighborhood in the data set
  int m_max_neighborhood;
  /// Molecular features.
  cnpy::NpyArray m_features;
  /// Neighbor information (adjacency matrix).
  cnpy::NpyArray m_neighbors;

  DataType position_scale_factor = 320.0;
  DataType bond_len_scale_factor = 10.0;

  /// support for data_store_pilot2_molecular
  std::vector<int> m_shape;

  /// support for data_store_pilot2_molecular
  int m_word_size;
  /// support for data_store_pilot2_molecular
  int m_owner;
  /// support for data_store_pilot2_molecular
  int m_neighbors_data_size;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_PILOT2_MOLECULAR_HPP
