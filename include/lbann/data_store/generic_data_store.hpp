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
#include <unordered_set>
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
  generic_data_store(generic_data_reader *reader, model *m); 

  //! copy ctor
  generic_data_store(const generic_data_store&) = default;

  //! operator=
  generic_data_store& operator=(const generic_data_store&) = default;

  //! dtor
  virtual ~generic_data_store() {}

  virtual generic_data_store * copy() const = 0;

  /// called by generic_data_reader::setup_data_store
  virtual void setup();

  /// called by generic_data_reader::update;
  /// this method call exchange_data if m_epoch > 1
  virtual void set_shuffled_indices(const std::vector<int> *indices, bool exchange_indices = true);

  /// called by various image data readers 
  virtual void get_data_buf(int data_id, std::vector<unsigned char> *&buf, int multi_idx = 0) {}
  virtual void get_data_buf(int data_id, int tid, std::vector<double> *&buf) {}

  virtual void get_data_buf_DataType(int data_id, std::vector<DataType> *&buf) {}

  const std::string & get_name() const {
    return m_name;
  }

  void set_name(std::string name) {
    m_name = name;
  }

  void set_is_subsidiary_store() {
    m_is_subsidiary_store = true;
  }

  bool is_subsidiary_store() const {
    return m_is_subsidiary_store;
  }

  const std::vector<std::vector<int> > * get_minibatch_indices() const {
    return m_my_minibatch_indices;
  }

  void set_minibatch_indices(const std::vector<std::vector<int> > *indices) {
    m_my_minibatch_indices = indices;
  }

  //@todo: for optimization, change m_my_minibatch_indices_v to a pointer,
  //       and properly handle ownership and destruction; this is needed
  //       to reduce memory requirements in, e.g, data_store_merge_features
  const std::vector<int>  & get_minibatch_indices_v() const {
    return m_my_minibatch_indices_v;
  }

  void set_minibatch_indices_v(const std::vector<int > &indices) {
    m_my_minibatch_indices_v = indices;
  }

  //@todo: for optimization, change m_my_minibatch_indices_v to a pointer,
  //       and properly handle ownership and destruction; this is needed
  //       to reduce memory requirements in, e.g, data_store_merge_features
  const std::unordered_set<int> & get_datastore_indices() const {
    return m_my_datastore_indices;
  }

  void set_datastore_indices(const std::unordered_set<int> &indices) {
    m_my_datastore_indices = indices;
  }

  const std::vector<std::vector<int>> & get_all_minibatch_indices() const {
    return m_all_minibatch_indices;
  }

  //@todo: for optimization, change m_all_minibatch_indices to a pointer,
  //       and properly handle ownership and destruction; this is needed
  //       to reduce memory requirements in, e.g, data_store_merge_features
  void set_all_minibatch_indices(const std::vector<std::vector<int>> &indices) {
    m_all_minibatch_indices = indices;
  }

  /// supports out-of-memory-mode
  virtual void fetch_data() {}

  void init_minibatch();

protected :

  virtual void exchange_data() = 0;

  generic_data_reader *m_reader;

  lbann_comm *m_comm;

  std::string m_name;

  /// returns the number of bytes in dir/fn; it's OK if dir = ""
  size_t get_file_size(std::string dir, std::string fn);

  /// number of indices that m_reader owns (in a global sense);
  /// equal to m_shuffled_indices->size()
  size_t m_num_global_indices;

  void set_num_global_indices() {
    m_num_global_indices = m_shuffled_indices->size();
  }

  /// the indices that will be used locally; the inner j-th vector
  /// contains indices referenced during the j-th call to
  /// generic_data_reader::fetch_data(...)
  const std::vector<std::vector<int> > *m_my_minibatch_indices;
  /// contains a concatenation of the indices in m_my_minibatch_indices
  ///@todo: for optimization, this should be a pointer -- as it is now,
  ///       in merge_features the vector must be copied to the subsidiary
  ///       data_store_cvs
  std::vector<int> m_my_minibatch_indices_v;
  /// fills in m_my_minibatch_indices_v
  void get_minibatch_index_vector();

  /// m_mb_counts[j] contains the number of indices
  /// passed to data_reader::fetch_data in one epoch
  std::vector<int> m_mb_counts;
  /// fills in m_mb_counts
  void exchange_mb_counts();

  /// m_all_minibatch_indices[j] will contain all indices that
  /// will be passed to data_reader::fetch_data in one epoch,
  /// for all processors
  std::vector<std::vector<int>> m_all_minibatch_indices;
  /// fills in m_all_minibatch_indices
  void  exchange_mb_indices();

  /// the indices that this processor owns;
  std::unordered_set<int> m_my_datastore_indices;
  /// fills in m_my_datastore_indices 
  void get_my_datastore_indices();
  /// fills in m_my_datastore_indices; this call is used when creating tarballs
  /// for pre-staging data
  void get_my_tarball_indices();

  size_t m_num_readers;

  /// this processor's rank
  int  m_rank;

  /// number of procs in the model
  int  m_np;

  size_t m_epoch;

  bool m_in_memory;

  bool m_master;

  const std::vector<int> *m_shuffled_indices;

  model *m_model;

  /// base directory for data
  std::string m_dir;

  /// conduct extensive testing
  bool m_extended_testing;

  /// returns the processor that owns the data associated
  /// with the index
  int get_index_owner(int idx) {
    return idx % m_np;
  }

  virtual void extended_testing() {}

  MPI_Comm m_mpi_comm;

  /// as of now, only applicable to merge_features and merge_samples
  bool m_is_subsidiary_store;

  /// maps and index from m_my_datastore_indices to a filepath
  /// for use in out-of-memory mode
  std::unordered_map<int, std::string> m_data_filepaths;
  /// fills in m_data_filepaths
  virtual void build_data_filepaths() {std::cerr << "shouldn't be here!\n";}

  /// outer vector size is m_np; m_all_partitioned_indices[i]
  /// contains m_my_minibatch_indices from P_i
  std::vector<std::vector<std::vector<int>>> m_all_partitioned_indices;
  /// supports out-of-memory-mode;
  /// all-to-all exchange of m_my_minibatch_indices;
  /// fills in m_all_partitioned_indices
  void exchange_partitioned_indices();
  /// size of the largest middle vector in m_all_partitioned_indices;
  /// this should be the number of minibatches in an epoch
  size_t m_num_minibatches;

  /// for debugging during development
  void print_partitioned_indices();

  /// supports out-of-memory mode; this is the current
  /// minibatch that is read into memory
  size_t m_cur_minibatch;

  bool m_is_setup;
  bool m_verbose;

  /// given a pathname: someplace/[someplace_else/...]/prefix
  /// returns <prefix,pathname>' throws exception if 's' does not contain
  /// at least one directory name, or 's' end with //'
  std::pair<std::string, std::string> get_pathname_and_prefix(std::string s);

  /// created the directory structure specified in 's', if it doesn't exist;
  /// 's' may optionally end in '/'
  void create_dirs(std::string s);

  /// runs a system command, and returns the output;
  /// if exit_on_error=true, and the value returned by the system
  /// call is other than the empty string, then an exception is thrown
  std::string run_cmd(std::string s, bool exit_on_error = true);
};

}  // namespace lbann

#endif  // __GENERIC_DATA_STORE_HPP__
