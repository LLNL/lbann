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
////////////////////////////////////////////////////////////////////////////////


#ifndef LBANN_DATA_READER_NPZ_RAS_LIPID_HPP
#define LBANN_DATA_READER_NPZ_RAS_LIPID_HPP

#include "lbann/data_readers/data_reader.hpp"
#include "conduit/conduit.hpp"
#include <cnpy.h>
#include <memory>

namespace lbann {
  /**
   * Data reader for data stored in numpy (.npz) files that are encapsulated
   * in conduit::Nodes
   */
class ras_lipid_conduit_data_reader : public generic_data_reader {

public:
  ras_lipid_conduit_data_reader(const bool shuffle);
  ras_lipid_conduit_data_reader(const ras_lipid_conduit_data_reader&);
  ras_lipid_conduit_data_reader& operator=(const ras_lipid_conduit_data_reader&);
  ~ras_lipid_conduit_data_reader() override {}

  ras_lipid_conduit_data_reader* copy() const override { return new ras_lipid_conduit_data_reader(*this); }

  std::string get_type() const override {
    return "ras_lipid_conduit_data_reader";
  }

  void load() override;

  int get_linearized_data_size() const override { return m_num_features; }
  int get_linearized_label_size() const override { return m_num_labels; }
  int get_linearized_response_size() const override { return m_num_response_features; }
  const std::vector<int> get_data_dims() const override { return m_data_dims; }

  int get_num_labels() const override { return m_num_labels; }

private:
  int m_num_features = 0;
  int m_num_labels = 3;
  int m_num_response_features = 0;
  std::vector<int> m_data_dims;

  /** @brief List of input npz filenames */
  std::vector<std::string> m_filenames;

  /** @brief The global number of samples */
  int m_num_samples = 0;

  /** @brief m_samples_per_file[j] contains the number of samples in the j-th file */
  std::vector<int> m_samples_per_file;

  /** @brief Maps a data_id to the file index (in m_filenames) that
   * contains the sample, and the offset in that file's npy array */
  std::unordered_map<int, std::pair<int, int>> m_data_id_map;

  /** @brief Maps a field name to the data's shape
   *
   * Example: "bbs" -> {184, 3}
   */
  std::unordered_map<std::string, std::vector<size_t>> m_datum_shapes;

  /** @brief Maps a field name to the word size */
  std::unordered_map<std::string, size_t> m_datum_word_sizes;

  /** @brief Maps a field name to the number of bytes in the datum
   *
   * Example: "bbs" -> 184*3*word_size
   */
  std::unordered_map<std::string, size_t> m_datum_num_bytes;

  /** @brief Maps a field name to the number of words in the datum */
  std::unordered_map<std::string, size_t> m_datum_num_words;

  //=====================================================================
  // private methods follow
  //=====================================================================

  /** @brief Contains common code for operator= and copy ctor */
  void copy_members(const ras_lipid_conduit_data_reader& rhs);

  void do_preload_data_store() override;

  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;

  /** @brief Populates in m_datum_shapes, m_datum_num_bytes, m_datum_word_sizes */
  void fill_in_metadata();

  /** @brief Collect the sample_ids that belong to this rank and
   *         rebuild the data store's owner map
   *
   * my_samples maps a filename (index in m_filenames) to the pair:
   * (data_id, local index of the sample wrt the samples in the file).
   */
  void get_my_indices(std::unordered_map<int, std::vector<std::pair<int,int>>> &my_samples);

  /** @brief Re-build the data store's owner map
   *
   * This one-off, wouldn't need to do this if we were using sample lists.
   */
  void rebuild_data_store_owner_map();

  /** @brief Fills in m_samples_per_file */
  void get_samples_per_file();

  /** @brief Write file sizes to disk
   *
   * Each line of the output file contains: filename num_samples
   */
  void write_file_sizes();

  /** @brief Read file that contains: filename num_samples
   *
   * see: write_file_sizes()
   */
  void read_file_sizes();
};

}  // namespace lbann

#endif //LBANN_DATA_READER_NPZ_RAS_LIPID_HPP
