////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "conduit/conduit.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/utils/options.hpp"
#include <cnpy.h>
#include <memory>

namespace lbann {
/**
 * Data reader for data stored in numpy (.npz) files that are encapsulated
 * in conduit::Nodes
 */
class ras_lipid_conduit_data_reader : public generic_data_reader
{

public:
  ras_lipid_conduit_data_reader(const bool shuffle);
  ras_lipid_conduit_data_reader(const ras_lipid_conduit_data_reader&);
  ras_lipid_conduit_data_reader&
  operator=(const ras_lipid_conduit_data_reader&);
  ~ras_lipid_conduit_data_reader() override {}

  ras_lipid_conduit_data_reader* copy() const override
  {
    return new ras_lipid_conduit_data_reader(*this);
  }

  std::string get_type() const override
  {
    return "ras_lipid_conduit_data_reader";
  }

  void load() override;

  void set_num_labels(int n) { m_num_labels = n; }

  int get_linearized_data_size() const override
  {
    return m_seq_len * m_num_features;
  }
  int get_linearized_label_size() const override
  {
    return m_seq_len * m_num_labels;
  }
  int get_linearized_response_size() const override
  {
    return m_num_response_features;
  }
  // const std::vector<int> get_data_dims() const override {  return
  // m_data_dims; }
  const std::vector<El::Int> get_data_dims() const override
  {
    return {get_linearized_data_size()};
  }
  int get_num_labels() const override { return m_seq_len * m_num_labels; }

private:
  int m_num_features = 0;
  int m_num_labels = 3;
  int m_num_response_features = 0;
  std::vector<int> m_data_dims;

  /** @brief Total of train + validate samples */
  size_t m_num_global_samples;
  size_t m_num_train_samples;
  size_t m_num_validate_samples;

  /** the number of sequential samples that are combined into a multi-sample */
  int m_seq_len = 1;

  // owner map for multi-samples
  std::unordered_map<int, int> m_multi_sample_to_owner;

  std::unordered_map<std::string, std::set<int>> m_filename_to_multi_sample;
  // std::unordered_map<std::string, std::unordered_set<int>>
  // m_filename_to_multi_sample;

  std::unordered_map<int, int> m_multi_sample_id_to_first_sample;

  //  sample_list_t m_sample_list;

  /** @brief List of input npz filenames */
  std::vector<std::string> m_filenames;

  /** @brief m_samples_per_file[j] contains the number of samples in the j-th
   * file */
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

  std::vector<double> m_min;
  std::vector<double> m_max_min;
  std::vector<double> m_mean;
  std::vector<double> m_std_dev;
  bool m_use_min_max;
  bool m_use_z_score;

  //=====================================================================
  // private methods follow
  //=====================================================================

  /** @brief Contains common code for operator= and copy ctor */
  void copy_members(const ras_lipid_conduit_data_reader& rhs);

  void do_preload_data_store() override;

  bool fetch_datum(CPUMat& X, uint64_t data_id, uint64_t mb_idx) override;
  bool fetch_label(CPUMat& Y, uint64_t data_id, uint64_t mb_idx) override;
  bool fetch_response(CPUMat& Y, uint64_t data_id, uint64_t mb_idx) override;

  /** @brief Populates in m_datum_shapes, m_datum_num_bytes, m_datum_word_sizes
   */
  void fill_in_metadata();

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

  void read_normalization_data();

  /** Print some statistics to cout */
  void print_shapes_etc();

  void load_the_next_sample(conduit::Node& node,
                            int sample_index,
                            std::map<std::string, cnpy::NpyArray>& data);

  void construct_multi_sample(std::vector<conduit::Node>& work,
                              uint64_t data_id,
                              conduit::Node& node);
};

} // namespace lbann

#endif // LBANN_DATA_READER_NPZ_RAS_LIPID_HPP
