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

  jag_store();

  jag_store(const jag_store&) = default;

  jag_store& operator=(const jag_store&) = default;

  ~jag_store() {}

  void set_comm(lbann_comm *comm) {
    m_comm = comm;
  }

  void load_data(int data_id, int tid);

  /// Returns the requested inputs
  const std::vector<data_reader_jag_conduit_hdf5::input_t> & fetch_inputs(int tid) const {
    return m_data_inputs[tid];
  }

  /// Returns the requested scalars
  const std::vector<data_reader_jag_conduit_hdf5::scalar_t> & fetch_scalars (int tid) const {
    return m_data_scalars[tid];
  }

  /// Returns the requested images
  const std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>> & fetch_images(int tid) {
    return m_data_images[tid];
  }

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

  void set_image_size(size_t n) { m_image_size = n; }

  size_t get_linearized_data_size() const;

  size_t get_linearized_image_size() const { 
    return m_image_size * m_image_channels_to_use.size();
  }
  size_t get_linearized_scalar_size() const { return m_scalars_to_use.size(); }
  size_t get_linearized_input_size() const { return m_inputs_to_use.size(); }
  size_t get_num_img_srcs() const { return m_image_views_to_use.size(); }

  const std::vector<size_t> & get_linearized_data_sizes() const { return m_data_sizes; }

  void check_sample_id(const size_t sample_id) const { 
    if (sample_id >= m_num_samples) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: sample_id >= m_num_samples");
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
  std::vector<std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>>> m_data_images;

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
};

} // end of namespace lbann
#endif //ifdef LBANN_HAS_CONDUIT

#endif // _JAG_STORE_HPP__
