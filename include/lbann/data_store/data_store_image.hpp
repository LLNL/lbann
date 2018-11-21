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
//
////////////////////////////////////////////////////////////////////////////////

#ifndef __DATA_STORE_IMAGE_HPP__
#define __DATA_STORE_IMAGE_HPP__

#include "lbann/data_store/generic_data_store.hpp"
#include <unordered_map>

namespace lbann {

/**
 * todo
 */

class data_store_image : public generic_data_store {
 public:

  //! ctor
  data_store_image(generic_data_reader *reader, model *m) :
    generic_data_store(reader, m),
    m_num_img_srcs(1) {}

  //! copy ctor
  data_store_image(const data_store_image&) = default;

  //! operator=
  data_store_image& operator=(const data_store_image&) = default;

  generic_data_store * copy() const override = 0;

  //! dtor
  ~data_store_image() override;

  void setup() override;

  using generic_data_store::get_data_buf;

  /// data readers call this method
  void get_data_buf(int data_id, std::vector<unsigned char> *&buf, int multi_idx = 0) override;

 protected :

  void exchange_data() override;

  /// maps a global index (wrt image_list) to number of bytes in the file
  std::unordered_map<size_t, size_t> m_file_sizes;
  /// fills in m_file_sizes
  virtual void get_file_sizes() = 0;

  /// fills in m_file_sizes; this is called when we're using files
  /// from a tarball
  virtual void read_file_sizes();

  /// called by get_file_sizes
  void exchange_file_sizes(
    std::vector<int> &global_indices,
    std::vector<int> &num_bytes);

  /// buffers that will be passed to reader::fetch_datum
  std::unordered_map<int, std::vector<unsigned char> > m_my_minibatch_data;
   
  /// loads file from disk into *p; checks that bytes read = sz
  void load_file(const std::string &dir, const std::string &fn, unsigned char *p, size_t sz); 

  /// reads all files assigned to this processor into memory (m_data)
  /// version for in-memory mode
  virtual void read_files() = 0; 
  /// version for out-of-memory mode
  virtual void read_files(const std::unordered_set<int> &indices) = 0; 

  /// in multi-image scenarios, the number of images in each sample
  unsigned int m_num_img_srcs;

  /// the actual data store!
  std::unordered_map<int, std::vector<unsigned char>> m_data;

  /// returns memory required to hold p's files in memory
  size_t get_my_num_file_bytes();

  /// returns number of bytes in the data set
  size_t get_global_num_file_bytes();

  /// parses /proc/meminfo to determine available memory; returned 
  /// value is memory in kB
  size_t get_available_memory();

  /// attempts to determine if there is sufficient RAM for
  /// in-memory data store; may call MPI_Abort
  void report_memory_constraints();

  /// for out-of-memory mode: read files from, e.g, lscratchX, and write
  /// to local store, e.g, /l/ssd
  void stage_files();

  /// for out-of-memory mode: unpack files from a previously created tarball
  void stage_tarball();

  /// called by data_reader::fetch_data; supports out-of-memory mode
  void fetch_data() override;

  /// creates a tarball of files written to local disk, then
  /// copies the tarball to, e.g, lscratchX. Activated by the cmd line
  /// options: --create_tarball <name> where <name> is the directory
  /// to which to copy the tarball. 
  void create_tarball();

  /// returns the string that will be passed to a system call to
  /// create the tarball on local store (/l/ssd), and string for copying
  /// to remote store (lscratchX)
  std::pair<std::string, std::string> get_tarball_exe();

  /// called by create_tarball
  void write_file_sizes();
  
  /// called by create_tarball
  void write_datastore_indices();

  void read_datastore_indices();

  /// called by create_tarball
  void write_data_filepaths();

  void read_data_filepaths();

  /// returns true if option: --create_tarball is in use;
  /// print info to screen, and performs error checking
  bool are_we_creating_tarballs();
};

}  // namespace lbann

#endif  // __DATA_STORE_IMAGE_HPP__
