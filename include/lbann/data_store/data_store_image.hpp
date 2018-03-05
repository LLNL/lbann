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

namespace lbann {

/**
 * todo
 */

class data_store_image : public generic_data_store {
 public:

  //! ctor
  data_store_image(lbann_comm *comm, generic_data_reader *reader, model *m) :
    generic_data_store(comm, reader, m),
    m_num_img_srcs(1) {}

  //! copy ctor
  data_store_image(const data_store_image&) = default;

  //! operator=
  data_store_image& operator=(const data_store_image&) = default;

  generic_data_store * copy() const override = 0;

  //! dtor
  ~data_store_image() override;

  void setup() override;

  /// data readers call this method
  void get_data_buf(int data_id, int tid, std::vector<unsigned char> *&buf, int multi_idx = 0) override;

 protected :

  void exchange_data() override;

  struct Triple {
    int global_index;
    int num_bytes;
    size_t offset;
    int rank;
  };

  /// maps a global index (wrt image_list) to number of bytes in the file
  std::unordered_map<size_t, size_t> m_file_sizes;

  /// maps a global index (wrt image_list) to the file's data location 
  /// wrt m_data
  std::map<size_t, size_t> m_offsets;

  /// fills in m_file_sizes
  virtual void get_file_sizes() = 0;

  /// called by get_file_sizes
  virtual void exchange_file_sizes(std::vector<Triple> &my_file_sizes, int num_global_indices);

  /// when running in in-memory mode, this buffer will contain
  /// the concatenated data
  std::vector<unsigned char> m_data;

  /// allocate mem for m_data
  void allocate_memory(); 

  /// loads file from disk into *p; checks that bytes read = sz
  void load_file(const std::string &dir, const std::string &fn, unsigned char *p, size_t sz); 

  /// reads all files assigned to this processor into memory (m_data)
  virtual void read_files() = 0; 

  /// will contain data to be passed to the data_reader
  std::vector<std::vector<unsigned char> > m_my_minibatch_data;

  /// maps indices wrt shuffled indices to indices in m_my_minibatch_data
  std::unordered_map<size_t, size_t> m_my_data_hash;

  /// this contains a concatenation of the indices in m_minibatch_indices
  /// (see: generic_data_reader.hpp)
  std::vector<size_t> m_my_minibatch_indices;

  /// m_num_images[j] contains the number of images "owned" by P_j
  std::vector<int> m_num_samples;

  /// fills in m_num_images
  void compute_num_samples();

  /// in multi-image scenarios, the number of images in each sample
  unsigned int m_num_img_srcs;

  /// used for extended testing
  std::unordered_map<size_t, std::string> m_test_filenames;
  /// used for extended testing
  std::unordered_map<size_t, size_t> m_test_filesizes;

  /// fills in m_test_filenames and m_test_filesizes; these are
  /// used by run_extended_testing, which is called durring exchange_data,
  /// *if* the "extended_testing" cmd line option is present.
  virtual void setup_extended_testing() {}

  MPI_Win m_win;
};

}  // namespace lbann

#endif  // __DATA_STORE_IMAGE_HPP__
