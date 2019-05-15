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

#ifndef __DATA_STORE_CONDUIT_HPP__
#define __DATA_STORE_CONDUIT_HPP__

#include "lbann_config.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "conduit/conduit_node.hpp"
#include <unordered_map>
#include <unordered_set>


namespace lbann {

// support for encoding data_id in conduit::Node, used by
// conduit_data_store and associated code
#define LBANN_SAMPLE_ID_PAD 9
#define LBANN_DATA_ID_STR(data_id) pad(std::to_string(data_id), LBANN_SAMPLE_ID_PAD, '0')

class generic_data_reader;

class data_store_conduit {

 public:

  //! ctor
  data_store_conduit(generic_data_reader *reader);

  //! copy ctor
  data_store_conduit(const data_store_conduit&);

  //! copy / split ctor
  data_store_conduit(const data_store_conduit&, const std::vector<int>&);

  //! operator=
  data_store_conduit& operator=(const data_store_conduit&);

  data_store_conduit * copy() const { return new data_store_conduit(*this); }

  //! dtor
  ~data_store_conduit();

  /// normally not needed, since reader is passed to ctor. But may
  /// be useful in some cases
  void set_data_reader_ptr(generic_data_reader *reader) { m_reader = reader; }

  //! convenience handle
  void set_shuffled_indices(const std::vector<int> *indices) { m_shuffled_indices = indices; }

  void setup(int mini_batch_size);

  /*
   * dah - may be needed in the future, but not needed for bare-bones squashing
  void set_is_subsidiary_store() {
    m_is_subsidiary_store = true;
  }

  bool is_subsidiary_store() const {
    return m_is_subsidiary_store;
  }
  */

  void check_mem_capacity(lbann_comm *comm, const std::string sample_list_file, size_t stride, size_t offset);

  /// returns the conduit node
  const conduit::Node & get_conduit_node(int data_id) const;

  /// if 'already_have = true' then the passed 'node' was obtained by a call to
  /// get_empty_node(). In some operating modes this saves us from copying the node
  void set_conduit_node(int data_id, conduit::Node &node, bool already_have = false);

  void set_preloaded_conduit_node(int data_id, conduit::Node &node);

  const conduit::Node & get_random_node() const;

  const conduit::Node & get_random_node(const std::string &field) const;

  /// returns an empty node
  conduit::Node & get_empty_node(int data_id);

  /// As of this writing, will be called if cmd line includes: --preload_data_store
  /// This may change in the future; TODO revisit
  void set_preload() { m_preload = true; }

  bool is_preloaded() { return m_preload; }

  void set_explicit_loading(bool flag) { m_explicit_loading = flag; }

  bool is_explicitly_loading() { return m_explicit_loading; }

  /// fills in m_owner, which maps index -> owning processor
  void build_preloaded_owner_map(const std::vector<int>& per_rank_list_sizes);

  /// Removed nodes corresponding from the indices vector from the data store
  void purge_unused_samples(const std::vector<int>& indices);

  /// Recompact the nodes because they are not copied properly when instantiating
  /// using the copy constructor
  void compact_nodes();

  /// returns the processor that owns the data associated
  /// with the index
  int get_index_owner(int idx);

  bool is_local_cache() const { return m_is_local_cache; }

  void exchange_mini_batch_data(size_t current_pos, size_t mb_size) {
    if (is_local_cache()) {
      return;
    }
    if (m_super_node) {
      exchange_data_by_super_node(current_pos, mb_size);
    } else {
      exchange_data_by_sample(current_pos, mb_size);
    }
    ++m_n;
  }

  bool has_conduit_node(int data_id) const;

protected :

  /// records the number of times exchange_mini_batch_data has been called
  int m_n;

  bool m_is_setup;

  void copy_members(const data_store_conduit& rhs, const std::vector<int>& = std::vector<int>());
  generic_data_reader *m_reader;

  lbann_comm *m_comm;

  /// rank in the trainer; convenience handle
  int  m_rank_in_trainer;

  /// number of procs in the trainer; convenience handle
  int  m_np_in_trainer;

  /// convenience handle
  bool m_world_master;

  /// convenience handle
  bool m_trainer_master;

  /// set to true if data_store is preloaded
  bool m_preload;

  /// set to true if data_store is being explicitly loaded
  bool m_explicit_loading;

  /// maps an index to the processor that owns the associated data
  mutable std::unordered_map<int, int> m_owner;

  /// convenience handle
  const std::vector<int> *m_shuffled_indices;

  /// The size of the mini-batch that was used to calculate ownership
  /// of samples when building the owner map.  This size has to be
  /// used consistently when computing the indices that will be sent
  /// and received.
  int m_owner_map_mb_size;

  /// if true, use exchange_data_by_super_node, else use
  /// exchange_data_by_sample; default if false
  bool m_super_node;

  void exchange_data_by_super_node(size_t current_pos, size_t mb_size);
  void exchange_data_by_sample(size_t current_pos, size_t mb_size);

  /// Contains the list of data IDs that will be received
  std::vector<int> m_recv_data_ids;

  /// contains the Nodes that this processor owns;
  /// maps data_id to conduit::Node
  mutable std::unordered_map<int, conduit::Node> m_data;

  /// This vector contains Nodes that this processor needs for
  /// the current minibatch; this is filled in by exchange_data()
  std::unordered_map<int, conduit::Node> m_minibatch_data;

  /// work space; used in exchange_data
  std::vector<conduit::Node> m_send_buffer;
  std::vector<conduit::Node> m_send_buffer_2;
  std::vector<El::mpi::Request<El::byte>> m_send_requests;
  std::vector<El::mpi::Request<El::byte>> m_recv_requests;
  std::vector<conduit::Node> m_recv_buffer;
  std::vector<int> m_outgoing_msg_sizes;
  std::vector<int> m_incoming_msg_sizes;

  /// size of a compacted conduit::Node that contains a single sample
  int m_compacted_sample_size;

  /// used in exchange_data_by_super_node(); contains the super_nodes,
  /// after they have been converted from compacted format
  std::vector<conduit::Node> m_reconstituted;

  void setup_data_store_buffers();

  /// called by exchange_data
  static void build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out);

  /// fills in m_owner, which maps index -> owning processor
  void build_owner_map(int mini_batch_size);

  /// maps processor id -> set of indices (whose associated samples)
  /// this proc needs to send. (formerly called "proc_to_indices)
  std::vector<std::unordered_set<int>> m_indices_to_send;

  /// fills in m_indices_to_send and returns the number of samples
  /// that will be sent
  int build_indices_i_will_send(int current_pos, int mb_size);

  /// maps processor id -> set of indices (whose associated samples)
  /// this proc needs to recv from others. (formerly called "needed")
  std::vector<std::unordered_set<int>> m_indices_to_recv;

  /// fills in m_indices_to_recv and returns the number of samples
  /// that will be received
  int build_indices_i_will_recv(int current_pos, int mb_size);

  void error_check_compacted_node(const conduit::Node &nd, int data_id);

  bool m_is_local_cache;
};

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT

#endif  // __DATA_STORE_JAG_HPP__
