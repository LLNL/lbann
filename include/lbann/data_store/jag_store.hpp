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
////////////////////////////////////////////////////////////////////////////////

#ifndef _JAG_STORE_HPP__
#define _JAG_STORE_HPP__

#include "lbann_config.hpp" 

#ifdef LBANN_HAS_CONDUIT

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "lbann/data_readers/data_reader_jag_conduit_hdf5.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "lbann/comm.hpp"

namespace lbann {

class data_reader_jag_conduit_hdf5;

/**
 * Loads the pairs of JAG simulation inputs and results from a conduit-wrapped hdf5 file
 */
class jag_store {
 public:

  #define METADATA_FN "metadata.txt"
  #define IMAGE_SIZE_PER_CHANNEL 4096
  #define NUM_IMAGE_CHANNELS 4
  #define MAX_SAMPLES_PER_BINARY_FILE 10000
  #define BINARY_FILE_BASENAME "converted"

  jag_store();

  jag_store(const jag_store&) = default;

  jag_store& operator=(const jag_store&) = default;

  ~jag_store() {}

  void set_comm(lbann_comm *comm) {
    m_comm = comm;
    m_master = comm->get_rank_in_world() == 0 ? true : false;
  }

  /// Returns the requested inputs
  const std::vector<data_reader_jag_conduit_hdf5::input_t> & fetch_inputs(size_t sample_id, size_t tid) const {
    check_sample_id(sample_id);
    return m_data_inputs[tid];
  }

  /// Returns the requested scalars
  const std::vector<data_reader_jag_conduit_hdf5::scalar_t> & fetch_scalars (size_t sample_id, size_t tid) const {
    check_sample_id(sample_id);
    return m_data_scalars[tid];
  }

  /// Returns the requested images
  const std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>> & fetch_views(size_t sample_id, size_t tid) {
    check_sample_id(sample_id);
    return m_data_images[tid];
  }

  /**
   * Loads data using the hdf5 conduit API from one or more conduit files.
   * "num_stores" and "my_rank" are used to determine which of the files
   * (in the conduit_filenames list) will be used. This functionality is 
   * needed when the jag_store is used in conjunction with 
   * data_store_jag_conduit
   */
  void setup(const std::vector<std::string> conduit_filenames,
             data_reader_jag_conduit_hdf5 *reader,
             bool num_stores = 1,
             int my_rank = 0);

  void set_image_size(size_t n) { m_image_size = n; }

  size_t get_linearized_data_size() const;
  size_t get_linearized_image_size() const { return m_image_size; }
  size_t get_num_channels() const { return NUM_IMAGE_CHANNELS; }
  size_t get_linearized_channel_size() const { return IMAGE_SIZE_PER_CHANNEL; }
  size_t get_linearized_scalar_size() const { return m_scalars_to_use.size(); }
  size_t get_linearized_input_size() const { return m_inputs_to_use.size(); }
  size_t get_num_img_srcs() const { return m_image_views_to_use.size(); }
  size_t get_num_channels_per_view() const { return m_image_channels_to_use.size(); }
  size_t get_total_num_channels() const { return get_num_img_srcs() * get_num_channels_per_view(); }

  const std::vector<size_t> & get_linearized_data_sizes() const { return m_data_sizes; }

  bool check_sample_id(const size_t sample_id) const { return sample_id < m_num_samples; }

  size_t get_num_samples() const { return m_num_samples; }

  void load_data(int data_id, int tid);

 private:

  bool m_is_setup;

  size_t m_image_size;

  size_t m_num_samples;

  bool m_run_tests;

  lbann_comm *m_comm;

  bool m_master;

  bool m_use_conduit;

  size_t m_sample_len;

  data_reader_jag_conduit_hdf5 *m_reader;

  std::unordered_set<std::string> m_valid_samples;

  std::unordered_map<size_t, std::string> m_id_to_name;

  std::vector<std::string> m_inputs_to_use;
  std::vector<std::string> m_scalars_to_use;
  std::vector<std::string> m_image_views_to_use;
  std::vector<int> m_image_channels_to_use;

  void load_inputs(const std::string &keys);
  void load_scalars(const std::string &keys);
  void load_image_views(const std::string &keys);
  void load_image_channels(const std::string &keys);

  std::vector<std::vector<data_reader_jag_conduit_hdf5::input_t>> m_data_inputs;
  std::vector<std::vector<data_reader_jag_conduit_hdf5::scalar_t>> m_data_scalars;
  std::vector<std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>>> m_data_images;

  lbann_comm *m_comm;

  bool m_master;

  void get_default_keys(std::string &filename, std::string &sample_id, std::string key1);
  // given a data_id, the corresponding sample is in the file
  // m_conduit_filenames[data_id], and the sample's name (key)
  // is m_data_id_to_string_id[data_id]
  std::vector<std::string> m_conduit_filenames;
  std::vector<int> m_data_id_to_filename_idx;
  std::vector<std::string> m_data_id_to_string_id;

  std::vector<size_t> m_data_sizes;

  void build_data_sizes();

  void run_tests(const std::vector<std::string> &conduit_filenames); 

  void load_variable_names();
  void report_linearized_sizes();
  void allocate_memory(); 

  void convert_conduit(const std::vector<std::string> &conduit_filenames);
  void open_output_files(const std::string &dir); 
  void write_binary_metadata(std::string dir); 
  void write_binary(const std::string &input, const std::string &dir); 

  std::unordered_map<std::string, size_t> m_key_map;
  void read_key_map(const std::string &filename); 

  // maps a shuffled index to <m_stream[idx], int>
  std::unordered_map<int, std::pair<int, int>> m_sample_map;
  std::unordered_map<int, std::string> m_sample_id_map;
  std::vector<std::vector<unsigned char>> m_scratch;

  std::vector<std::vector<std::ifstream*>> m_stream;

  //normalization scalars for image channels
  std::vector<double> m_normalize;

  void check_entry(std::string &e) {
    if (m_key_map.find(e) == m_key_map.end()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: m_key_map is missing entry: " + e);
    }
  }

  std::ofstream m_name_file;
  std::ofstream m_binary_file;
  int m_cur_bin_count;
  int m_bin_file_count;
};

} // end of namespace lbann
#endif //ifdef LBANN_HAS_CONDUIT

#endif // _JAG_STORE_HPP__
