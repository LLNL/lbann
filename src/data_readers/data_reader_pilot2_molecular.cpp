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

#include "lbann/data_readers/data_reader_pilot2_molecular.hpp"
#include "lbann/utils/options.hpp"

namespace lbann {

pilot2_molecular_reader::pilot2_molecular_reader(
  int num_neighbors, int max_neighborhood, bool shuffle) :
  generic_data_reader(shuffle), m_num_neighbors(num_neighbors), m_max_neighborhood(max_neighborhood) {}

void pilot2_molecular_reader::load() {
  // support for data store functionality: when not using data store, all procs
  // load the data; when using data store, only one does so
  bool is_mine = true;
  // note: when support for merge_samples is in place, the condition
  //       "get_role() == "test" will go away. For now we need it, else
  //       merge_samples will break

  if (is_mine) {
    std::string infile = get_file_dir() + get_data_filename();
    // Ensure the file exists.
    std::ifstream ifs(infile);
    if (!ifs) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " pilot2_molecular::load() - can't open file : " + infile);
    }
    ifs.close();

    // Load the dictionary.
    cnpy::npz_t dict = cnpy::npz_load(infile);
    // Verify we have features and neighbors.
    if (dict.count("features") != 1) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " pilot2_molecular::load() - no features");
    }
    if (dict.count("neighbors") != 1) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " pilot2_molecular::load() - no neighbors");
    }
    m_features = dict["features"];
    m_neighbors = dict["neighbors"];

    // Ensure we understand the word size.
    if (!(m_features.word_size == 4 || m_features.word_size == 8)) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " pilot2_molecular::load() - feature word size " +
        std::to_string(m_features.word_size) + " not supported");
    }
    if (!(m_neighbors.word_size == 4 || m_neighbors.word_size == 8)) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " pilot2_molecular::load() - neighbor word size " +
        std::to_string(m_neighbors.word_size) + " not supported");
    }
    // Fortran data order not supported.
    if (m_features.fortran_order) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " pilot2_molecular::load() - feature fortran order not supported");
    }
    if (m_neighbors.fortran_order) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " pilot2_molecular::load() - neighbor fortran order not supported");
    }

    // Assume we collapse samples from every frame into one set.
    m_num_samples = m_features.shape[0] * m_features.shape[1];
    m_num_samples_per_frame = m_features.shape[1];
    // The first two dimensions are the frame and the sample, so skip.
    m_num_features = std::accumulate(
      m_features.shape.begin() + 2, m_features.shape.end(), (unsigned) 1,
      std::multiplies<unsigned>());

    m_word_size = m_neighbors.word_size;

    m_shape.resize(3);
    m_shape[0] = m_num_neighbors + 1;
    m_shape[1] = m_features.shape[2];
    m_shape[2] = m_features.shape[3];
  }

  // Reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

bool pilot2_molecular_reader::fetch_datum(
  CPUMat& X, int data_id, int mb_idx) {

  const int frame = get_frame(data_id);
  // Fetch the actual molecule.
  fetch_molecule(X, data_id, 0, mb_idx);
  // Fetch the neighbors - note that the offset is 2x the max
  // neighborhood size to accommodate the top and bottom of the
  // bilayer
  const int neighbor_frame_offset =
    frame * m_num_samples_per_frame * (2 * m_max_neighborhood);
  const int intra_frame_data_id = data_id - frame * m_num_samples_per_frame;
  if (m_neighbors.word_size == 4) {
    float *neighbor_data = m_neighbors.data<float>() +
      neighbor_frame_offset + intra_frame_data_id * (2 * m_max_neighborhood);
    // Start at 1 to skip self.
    for (int i = 1; i < m_num_neighbors + 1; ++i) {
      int neighbor_id = neighbor_data[i];
      if (neighbor_id != -1) {
        fetch_molecule(X, neighbor_id + frame * m_num_samples_per_frame, i,
                       mb_idx);
      }
    }
  } else if (m_neighbors.word_size == 8) {
    double *neighbor_data = m_neighbors.data<double>() +
      neighbor_frame_offset + intra_frame_data_id * (2 * m_max_neighborhood);
    // Start at 1 to skip self.
    for (int i = 1; i < m_num_neighbors + 1; ++i) {
      int neighbor_id = neighbor_data[i];
      if (neighbor_id != -1) {
        fetch_molecule(X, neighbor_id + frame * m_num_samples_per_frame, i,
                       mb_idx);
      }
    }
  }
  return true;
}

void pilot2_molecular_reader::fetch_molecule(CPUMat& X, int data_id, int idx,
                                             int mb_idx) {
  const int frame = get_frame(data_id);
  // Compute the offset in features for this frame.
  const int frame_offset = frame * m_num_features * m_num_samples_per_frame;
  const int intra_frame_data_id = data_id - frame * m_num_samples_per_frame;
  if (m_features.word_size == 4) {
    float *data = m_features.data<float>() + frame_offset +
      intra_frame_data_id * m_num_features;
    for (int i = 0; i < m_num_features; ++i) {
      X(m_num_features * idx + i, mb_idx) = scale_data<float>(i, data[i]);
    }
  } else if (m_features.word_size == 8) {
    double *data = m_features.data<double>() + frame_offset +
      intra_frame_data_id * m_num_features;
    for (int i = 0; i < m_num_features; ++i) {
      X(m_num_features * idx + i, mb_idx) = scale_data<double>(i, data[i]);
    }
  }
}

}  // namespace lbann
