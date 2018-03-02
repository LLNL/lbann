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

#ifndef __GENERIC_DATA_STORE_HPP__
#define __GENERIC_DATA_STORE_HPP__


#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include <vector>
#include <unordered_map>

namespace lbann {

class generic_data_reader;
class lbann_comm;
class model;

/**
 * todo
 */

class generic_data_store {
 public:

  //! ctor
  generic_data_store(lbann_comm *comm, generic_data_reader *reader, model *m); 

  //! copy ctor
  generic_data_store(const generic_data_store&) = default;

  //! operator=
  generic_data_store& operator=(const generic_data_store&) = default;

  //! dtor
  virtual ~generic_data_store() {}

  virtual generic_data_store * copy() const = 0;

  /// called by generic_data_reader::setup_data_store
  virtual void setup();

  /// called by generic_data_reader::update
  void set_shuffled_indices(const std::vector<int> *indices);

  /// called by various image data readers 
  virtual void get_data_buf(int data_id, std::vector<unsigned char> *&buf, int multi_idx = 0) = 0;

 protected :

  virtual void exchange_data() = 0;

  /// returns the number of bytes in dir/fn; it's OK if dir = ""
  size_t get_file_size(std::string dir, std::string fn);

  /// number of indices that m_reader owns (in a global sense);
  /// equal to m_shuffled_indices->size()
  size_t m_num_global_indices;

  virtual void set_num_global_indices() = 0;

  /// the indices that will be used locally; the inner j-th vector
  /// contains indices referenced during the j-th call to
  /// genreic_data_reader::fetch_data(...)
  const std::vector<std::vector<int> > *m_minibatch_indices;

  /// the indices that this processor owns; these are in the
  /// range [0..m_num_global_indices]
  std::vector<size_t> m_my_datastore_indices;

  ///m_my_global_indices[i] = m_shuffled_indices[ m_my_datastore_indices[i]];
  /// this is wrt the initial shuffled index vector
  std::vector<size_t> m_my_global_indices;

  /// fills in m_my_datastore_indices and m_my_global_indices
  virtual void get_my_datastore_indices() = 0;

  size_t m_num_readers;

  size_t m_rank;

  size_t m_epoch;

  bool m_in_memory;

  lbann_comm *m_comm;

  bool m_master;

  generic_data_reader *m_reader;

  const std::vector<int> *m_shuffled_indices;

  /// maps global indices (wrt shuffled_indices) to owning processor
  std::unordered_map<size_t, size_t> m_owner_mapping;

  model *m_model;

  /// base directory for data
  std::string m_dir;

  /// conduct extensive testing
  bool m_extended_testing;
};

}  // namespace lbann

#endif  // __GENERIC_DATA_STORE_HPP__
