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
   #define IMAGE_CHANNELS 4
   #define MAX_SAMPLES_PER_BINARY_FILE 10000
   #define BINARY_FILE_BASENAME "converted"

  jag_store();

  jag_store(const jag_store&) = default;

  jag_store& operator=(const jag_store&) = default;

  ~jag_store() {}

  void set_comm(lbann_comm *comm) {
    m_comm = comm;
  }

  /// load data from disk into RAM buffer for the specified thread
  void load_data(int data_id, int tid);

  /// Returns the requested inputs
  const std::vector<data_reader_jag_conduit_hdf5::input_t> & fetch_inputs(int tid) const {
    return m_data_inputs[tid];
  }

  /// Returns the requested scalars
  const std::vector<data_reader_jag_conduit_hdf5::scalar_t> & fetch_scalars (int tid) const {
    return m_data_scalars[tid];
  }

  /// Returns the requested views (a.k.a. images)
  const std::vector<std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>>> & fetch_views(int tid) {
    return m_data_images[tid];
  }

  /// methods for converting conduit bundle to our binary format
  void write_binary(const std::string &input_fn, const std::string &output_dir);
  void write_binary_metadata(std::string dir);

  void load_inputs(const std::string &keys);
  void load_scalars(const std::string &keys);
  void load_image_views(const std::string &keys);
  void load_image_channels(const std::string &keys);

  /**
   * Loads data using the hdf5 conduit API from one or more conduit files.
   * "num_stores" and "my_rank" are used to determine which of the files
   * (in the conduit_filenames list) will be used when each processor
   * is loading a subset of the dataset.
   */
  void setup(const std::vector<std::string> conduit_filenames,
             data_reader_jag_conduit_hdf5 *reader,
             bool num_stores = 1,
             int my_rank = 0);

  void set_image_size(size_t n) { 
    //m_image_size = n; 
  }

  size_t get_linearized_data_size() const;

  // this is the number of pixels in one channel from a view;
  size_t get_linearized_image_size() const { 
    return IMAGE_SIZE_PER_CHANNEL * m_image_channels_to_use.size();
  }
  size_t get_linearized_scalar_size() const { return m_scalars_to_use.size(); }
  size_t get_linearized_input_size() const { return m_inputs_to_use.size(); }
  size_t get_num_img_srcs() const { return m_image_views_to_use.size(); }

  const std::vector<size_t> & get_linearized_data_sizes() const { return m_data_sizes; }

  void check_sample_id(const size_t sample_id) const { 
    if (sample_id >= m_num_samples) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: sample_id >= m_num_samples");
    }
    if (m_sample_map.find(sample_id) == m_sample_map.end()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: sample_id: " + std::to_string(sample_id) + " missing from m_sample_map");
    }
  }

  size_t get_num_samples() const { return m_num_samples; }

 private:

  bool m_is_setup;

  size_t m_image_size;

  size_t m_num_samples;

  // list of keys in the conduit files
  std::vector<std::string> m_inputs_to_use;
  std::vector<std::string> m_scalars_to_use;
  std::vector<std::string> m_image_views_to_use;
  std::vector<int> m_image_channels_to_use;

  // the actual data. The outer vector has size: omp_get_max_threads()
  std::vector<std::vector<data_reader_jag_conduit_hdf5::input_t>> m_data_inputs;
  std::vector<std::vector<data_reader_jag_conduit_hdf5::scalar_t>> m_data_scalars;
  // outer vector: 
  //    |thread    |views (3)  |channels (4)  |the data
  std::vector<std::vector<std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>>>> m_data_images;
    //std::vector<std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>>> m_data_images_2;

  lbann_comm *m_comm;

  std::vector<size_t> m_data_sizes;
  void build_data_sizes();

  // given a data_id, the corresponding sample is in the file
  // m_conduit_filenames[data_id], and the sample's name (key)
  // is m_data_id_to_string_id[data_id]
  std::vector<std::string> m_conduit_filenames;
  std::vector<int> m_data_id_to_filename_idx;
  std::vector<std::string> m_data_id_to_string_id;

  data_reader_jag_conduit_hdf5 *m_reader;

  bool m_master;

  void convert_conduit(const std::vector<std::string> &conduit_filenames);
  void test_conversion(const std::string &input_fn, const std::string &output_dir);

  std::unordered_map<std::string, size_t> m_key_map;
  void read_key_map(const std::string &filename);

  std::ofstream m_name_file;
  std::ofstream m_binary_file;
  int m_cur_bin_count;
  int m_bin_file_count;
  void open_output_files(const std::string &base);
  size_t m_sample_len;

  std::vector<std::vector<std::ifstream*>> m_stream;

  // maps a shuffled index to <m_stream[idx], int>
  std::unordered_map<int, std::pair<int, int>> m_sample_map;

  //normalization scalars for image channels
  std::vector<double> m_normalize;

  std::vector<std::vector<unsigned char>> m_scratch;
  //std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>> m_scratch;

  void check_entry(std::string &e) {
    if (m_key_map.find(e) == m_key_map.end()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: m_key_map is missing entry: " + e);
    }
  }

  bool m_use_conduit;

  void load_data_from_conduit(int data_id, int tid);
  void setup_conduit(double tm1);
  void load_variable_names();
  void allocate_memory();
  void report_linearized_sizes();

  std::unordered_set<std::string> m_valid_samples;
  std::unordered_map<size_t, std::string> m_id_to_name;
  void get_default_keys(std::string &filename, std::string &sample_id, std::string key1, bool master);
  std::vector<std::string> m_inputs_to_load;
  std::vector<std::string> m_scalars_to_load;
  std::vector<std::string> m_images_to_load;



};

} // end of namespace lbann
#endif //ifdef LBANN_HAS_CONDUIT

#endif // _JAG_STORE_HPP__
