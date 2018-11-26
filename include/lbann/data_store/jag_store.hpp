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

#include "lbann/utils/timer.hpp"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "lbann/data_readers/data_reader_jag_conduit_hdf5.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "lbann/comm.hpp"
#include "hdf5.h"

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
  #define MAX_SAMPLES_PER_BINARY_FILE 1000
  //#define MAX_SAMPLES_PER_BINARY_FILE 10000
  #define BINARY_FILE_BASENAME "converted"
  #define FILES_PER_DIR 1000

  jag_store();

  jag_store(const jag_store&) = default;

  jag_store& operator=(const jag_store&) = default;

  ~jag_store() {}

  void set_comm(lbann_comm *comm) {
    m_comm = comm;
    m_num_procs_in_world = m_comm->get_procs_in_world();
    m_rank_in_world = m_comm->get_rank_in_world();
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

  void setup(data_reader_jag_conduit_hdf5 *reader,
             bool num_stores = 1,
             int my_rank = 0);

  void set_image_size(size_t n) { m_image_size = n; }

  size_t get_linearized_data_size() const;
  size_t get_linearized_image_size() const { return 4096*4; }
  //size_t get_linearized_image_size() const { return m_image_size; }
  size_t get_linearized_channel_size() const { return IMAGE_SIZE_PER_CHANNEL; }

  /// returns the total number of channels in a view (image)
  /// Note: probably should be deleted, since we can chose which
  ///       channels to use
  //size_t get_num_channels() const { return NUM_IMAGE_CHANNELS; }
  size_t get_linearized_scalar_size() const { return m_scalars_to_use.size(); }
  size_t get_linearized_input_size() const { return m_inputs_to_use.size(); }

  /// returns the number of views (images) that we're actually using
  /// (so currently may be 0, 1, 2, or 3)
  size_t get_num_img_srcs() const { return m_image_views_to_use.size(); }

  /// returns the number of channels that we're actually using per view,
  /// i.e, may be 1, 2, 3, or 4
  size_t get_num_channels_per_view() const { return m_image_channels_to_use.size(); }

  /// returns the number channels that we're actually using, * num_views
  size_t get_total_num_channels() const { return get_num_img_srcs() * get_num_channels_per_view(); }

  const std::vector<size_t> & get_linearized_data_sizes() const { return m_data_sizes; }

  bool check_sample_id(const size_t sample_id) const { return sample_id < m_num_samples; }

  size_t get_num_samples() const { return m_num_samples; }

  void load_data(int data_id, int tid) {
    check_sample_id(data_id);
    if (m_mode == 1) {
      load_data_conduit(data_id, tid);
    } else if (m_mode == 2) {
      load_data_binary(data_id, tid);
    }
  }

 private:

  /// one of these is called by load_data()
  void load_data_conduit(int data_id, int tid);
  void load_data_binary(int data_id, int tid);

  size_t m_image_size;

  size_t m_num_samples;

  lbann_comm *m_comm;

  int m_num_procs_in_world;

  int m_rank_in_world;

  bool m_master;

  data_reader_jag_conduit_hdf5 *m_reader;

  /// next three will contain the actual sample data;
  /// they are filled in by one of the load_data_XX methods;
  /// each thread has a separate set of buffers
  std::vector<std::vector<data_reader_jag_conduit_hdf5::input_t>> m_data_inputs;
  std::vector<std::vector<data_reader_jag_conduit_hdf5::scalar_t>> m_data_scalars;
  std::vector<std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>>> m_data_images;

  /// next four are called by setup()
  void build_data_sizes();
  void load_variable_names();
  void report_linearized_sizes();
  void allocate_memory(); 

  /// these hold the names of the dependent and independant variables
  /// that we're using
  std::vector<std::string> m_inputs_to_use;
  std::vector<std::string> m_scalars_to_use;
  std::vector<std::string> m_image_views_to_use;
  std::vector<int> m_image_channels_to_use;

  /// these fill in the above four variables;
  /// they are called by load_variable_names()
  void load_inputs_to_use(const std::string &keys);
  void load_scalars_to_use(const std::string &keys);
  void load_image_views_to_use(const std::string &keys);
  void load_image_channels_to_use(const std::string &keys);

  std::vector<size_t> m_data_sizes;

  void check_entry(std::string &e) {
    if (m_key_map.find(e) == m_key_map.end()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: m_key_map is missing entry: " + e);
    }
  }

  /// one of the next three methods is called by setup(), depending 
  /// on the value of --mode=<int>
  int m_mode;
  void setup_conduit();  // mode = 1
  void setup_binary();   // mode = 2
  void setup_testing();  // mode = 3

  size_t m_max_samples;

  /// next three are used when reading samples from conduit files
  std::vector<std::string> m_conduit_filenames;
  std::vector<int> m_data_id_to_conduit_filename_idx;
  std::vector<std::string> m_data_id_to_sample_id;


  // these are used when reading samples from binary formatted files
  std::vector<std::vector<unsigned char>> m_scratch;
  std::unordered_map<std::string, size_t> m_key_map;
  // maps a shuffled index to <file_idx, local_idx>
  std::unordered_map<int, std::pair<int, int>> m_sample_map;
  std::unordered_map<std::string, int> m_sample_id_to_global_idx;
  std::vector<std::string> m_binary_filenames;
  // maps global idx (i.e: shuffled indices subscript) to sample ID 
  // (e.g: 0.9.99.57:1)
  std::unordered_map<int, std::string> m_sample_id_map;
  size_t m_sample_len;
  std::vector<std::vector<std::ifstream*>> m_streams;
  void read_key_map(const std::string &filename); 

  /// methods and variables for dealing with normalization
  void load_normalization_values();
  void load_normalization_values_impl(
      std::vector<std::pair<double, double>> &values,
      const std::vector<std::string> &variables); 

  std::vector<std::pair<double, double>> m_normalize_inputs;
  std::vector<std::pair<double, double>> m_normalize_scalars;
  std::vector<std::pair<double, double>> m_normalize_views;

  // magic numbers (from Rushil); these are for normalizing the images
  // 0.035550589898738466
  // 0.0012234476453273034
  // 1.0744965260584181e-05
  // 2.29319120949361e-07

  // testing and other special methods: if these are invoked something 
  // special happens, the the code exits; in the case a model is not run
  void compute_min_max();
  void compute_bandwidth();
  void build_conduit_index(const std::vector<std::string> &filenames);
  void compute_bandwidth_binary();
  void convert_conduit_to_binary(const std::vector<std::string> &filenames);
  void test_converted_files();

  /// functions and variables for converting conduit files to a binary format;
  /// these are used by convert_conduit_to_binary
  void write_binary_metadata(std::string dir); 
  void write_binary(const std::vector<std::string> &input, const std::string &dir); 
  std::ofstream m_name_file;
  size_t m_global_file_idx;
  size_t m_num_converted_samples;
  void open_binary_file_for_output(const std::string &dir);
  std::ofstream m_binary_output_file;
  std::ofstream m_binary_output_file_names;
  std::string m_binary_output_filename;
};

} // end of namespace lbann
#endif //ifdef LBANN_HAS_CONDUIT

#endif // _JAG_STORE_HPP__
