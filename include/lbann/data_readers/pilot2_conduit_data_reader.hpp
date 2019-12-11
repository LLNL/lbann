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

#ifndef LBANN_DATA_READER_PILOT2_CONDUIT
#define LBANN_DATA_READER_PILOT2_CONDUIT

#include "lbann/data_readers/data_reader.hpp"
#include "lbann_config.hpp"
#include "data_reader.hpp"
#include "conduit/conduit.hpp"
#include "hdf5.h"

//#define _USE_IO_HANDLE_ //assume we're reading data that was saved in conduit-bin format
#ifdef _USE_IO_HANDLE_
#include "lbann/data_readers/sample_list_conduit_io_handle.hpp"
#else
#include "lbann/data_readers/sample_list_hdf5.hpp"
#endif

namespace lbann {
  /**
   * Data reader for npz data that has previously been converted to conduit
   */
class pilot2_conduit_data_reader : public generic_data_reader {

public:

  using sample_name_t = std::string;
#ifdef _USE_IO_HANDLE_
  using sample_list_t = sample_list_conduit_io_handle<sample_name_t>;
#else
  using sample_list_t = sample_list_hdf5<sample_name_t>;
#endif

  using file_handle_t = sample_list_t::file_handle_t;
  using sample_file_id_t = sample_list_t::sample_file_id_t;
  using sample_t = std::pair<sample_file_id_t, sample_name_t>;

  pilot2_conduit_data_reader(const bool shuffle);
  pilot2_conduit_data_reader(const pilot2_conduit_data_reader&);
  pilot2_conduit_data_reader& operator=(const pilot2_conduit_data_reader&);
  ~pilot2_conduit_data_reader() override {}

  pilot2_conduit_data_reader* copy() const override { return new pilot2_conduit_data_reader(*this); }

  std::string get_type() const override {
    return "pilot2_conduit_data_reader";
  }

  void load() override;

  void set_num_labels(int n) { m_num_labels = n; }

  int get_linearized_data_size() const override { return m_num_features; }
  int get_linearized_label_size() const override { return m_num_labels; }
  int get_linearized_response_size() const override { return m_num_response_features; }
  const std::vector<int> get_data_dims() const override {  return m_data_dims; }
  int get_num_labels() const override { return m_num_labels; }

private:

  /**
   * The leading data reader among the local readers, which actually does the
   * file IO and data shuffling.
   */
   pilot2_conduit_data_reader* m_leading_reader = nullptr;

  int m_num_features = 0;
  int m_num_labels = 0;
  int m_num_response_features = 0;
  std::vector<int> m_data_dims;

  sample_list_t m_sample_list;

  /** @brief List of input npz filenames */
  std::vector<std::string> m_filenames;

  /** @brief The global number of samples */
  int m_num_samples = 0;

  /** @brief Maps a data_id to the file index (in m_filenames) that
   * contains the sample, and the offset in that file's npy array */
  std::unordered_map<int, std::pair<int, int>> m_data_id_map;

  /** @brief Maps a field name to the data's shape
   *
   * Example: "bbs" -> {184, 3}
   */
  std::unordered_map<std::string, std::vector<size_t>> m_datum_shapes;

  /** @brief Maps a field name to the number of words in the datum */
  std::unordered_map<std::string, size_t> m_datum_num_words;

  ///@{
  /** Structure for holding normalization values */
  std::vector<double> m_min;
  std::vector<double> m_max_min;
  std::vector<double> m_mean;
  std::vector<double> m_std_dev;
  ///@}

  enum normalization_type { none, min_max, std_dev };

  normalization_type m_normalize = std_dev;

  //=====================================================================
  // private methods follow
  //=====================================================================

  void load_conduit_node(const size_t index, conduit::Node &node, bool pre_open_fd = false); 

  /** @brief Fills in m_min, m_max, m_mean, m_stdev with values from file */
  void read_normalization_data();

  void normalize(conduit::Node &node);

  bool has_path(const file_handle_t& h, const std::string& path) const;
  void read_node(const file_handle_t& h, const std::string& path, conduit::Node& n) const;

  /** @brief Contains common code for operator= and copy ctor */
  void copy_members(const pilot2_conduit_data_reader& rhs);

  void do_preload_data_store() override;

  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;

  /** @brief Populates in m_datum_shapes, etc. */
  void fill_in_metadata();
};

}  // namespace lbann

#endif //LBANN_DATA_READER_PILOT2_CONDUIT
