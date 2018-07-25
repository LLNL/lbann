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

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "lbann/data_readers/cv_process.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "lbann/comm.hpp"

namespace lbann {

/**
 * Loads the pairs of JAG simulation inputs and results from a conduit-wrapped hdf5 file
 */
class jag_store {
 public:

  using ch_t = float; ///< jag output image channel type
  using scalar_t = double; ///< jag scalar output type
  using input_t = double; ///< jag input parameter type

  jag_store();

  jag_store(const jag_store&) = default;

  jag_store& operator=(const jag_store&) = default;

  ~jag_store() {}

  void set_comm(lbann_comm *comm) {
    m_comm = comm;
  }

  /// Returns the requested input
  input_t fetch_input(size_t sample_id, const std::string &key) const;

  /// Returns the requested scalar
  scalar_t fetch_scalar(size_t sample_id, const std::string &key) const;

  /// Returns a pointer to the requested scalar
  const ch_t * fetch_image(size_t sample_id, const std::string &key) const;

  /**
   * Must be called before setup(), else has no effect. Test mode
   * incurs considerable overhead in memory, and may increase setup
   * time by over an order of magnitude.
   */
  void set_test_mode() { m_test_mode_is_set = true; }

  /**
   * Has no effect if set_test_mode() was not called. Also,
   * must be called after setup(), else has no effect.
   * In addition to normal loading (where we load only the keys of
   * interest using the hdf5 api of conduit), we also load the
   * the entire conduit::Node, and test that our loaded char* data
   * matches that from the conduit::node
   */
  void run_test();

  /**
   * Load all keys from the "input" section of the bundle.
   * This must be called before calling setup()
   */
  void load_inputs();

  /**
   * Load all keys from the "scalars" section of the bundle.
   * This must be called before calling setup()
   */
  void load_scalars();

  /**
   * Load the requested images.
   * This must be called before calling setup()
   */
  void load_images(const std::vector<std::string> &keys);
  
  /**
   * Loads data using the hdf5 conduit API from one or more conduit files.
   * "num_stores" and "my_rank" are used to determine which of the files
   * (in the conduit_filenames list) will be used. This functionality is 
   * needed when the jag_store is used in conjunction with 
   * data_store_jag_conduit
   */
  void setup(const std::vector<std::string> conduit_filenames,
             bool num_stores = 1,
             int my_rank = 0);

  size_t get_linearized_data_size() const;
  size_t get_linearized_image_size() const;
  size_t get_linearized_scalar_size() const;
  size_t get_linearized_input_size() const;
  size_t get_num_img_srcs() const;

  std::vector<size_t> get_linearized_data_sizes() const;

  bool check_sample_id(const size_t sample_id) const;

  size_t get_num_samples() const { return m_num_samples; }

 private:

  bool m_test_mode_is_set;

  bool m_is_setup;

  bool m_load_inputs;

  bool m_load_scalars;

  size_t m_image_size;

  size_t m_num_samples;

  std::unordered_set<std::string> m_inputs_to_load;
  std::unordered_set<std::string> m_scalars_to_load;
  std::unordered_set<std::string> m_images_to_load;

  std::unordered_map<std::string, std::vector<input_t>> m_data_inputs;
  std::unordered_map<std::string, std::vector<scalar_t>> m_data_scalars;
  std::unordered_map<std::string, std::vector<std::vector<ch_t>>> m_data_images;

  lbann_comm *m_comm;

  void get_default_keys(std::string &filename, std::string &sample_id, std::string key1, bool master);


};

} // end of namespace lbann

#endif // _JAG_STORE_HPP__
