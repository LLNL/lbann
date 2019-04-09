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

#ifndef __DATA_STORE_JAG_HPP__
#define __DATA_STORE_JAG_HPP__

#include "lbann_config.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "conduit/conduit_relay_io.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "conduit/conduit_relay_mpi.hpp"
#include <unordered_map>

namespace lbann {

class data_store_jag {
 public:

  //! ctor
  data_store_jag(generic_data_reader *reader);

  //! copy ctor
  data_store_jag(const data_store_jag&);

  //! copy / split ctor
  data_store_jag(const data_store_jag&, const std::vector<int>&);

  //! operator=
  data_store_jag& operator=(const data_store_jag&);

  data_store_jag * copy() const override { return new data_store_jag(*this); }

  //! dtor
  ~data_store_jag() override;

  void setup(int mini_batch_size) override;

  void copy_members(const data_store_jag& rhs);

  /// normally not needed, since reader is passed to ctor. But may
  /// be useful in some cases
  void set_data_reader_ptr(generic_data_reader *reader) { m_reader = reader; }

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

  /// returns the conduit node
  const conduit::Node & get_conduit_node(int data_id) const;

  void set_conduit_node(int data_id, conduit::Node &node);

  void set_preloaded_conduit_node(int data_id, conduit::Node &node);

  const conduit::Node & get_random_node() const;

  const conduit::Node & get_random_node(const std::string &field) const;

  /// returns an empty node
  conduit::Node & get_empty_node(int data_id);

  /// As of this writing, will be called if cmd line includes: --preload_data_store
  /// This may change in the future; TODO revisit
  void set_preload() { m_preload = true; }
  bool is_preloaded() { return m_preload; }

  /// fills in m_owner, which maps index -> owning processor
  void build_preloaded_owner_map(const std::vector<int>& per_rank_list_sizes);

  /// Removed nodes corresponding from the indices vector from the data store
  void purge_unused_samples(const std::vector<int>& indices);

  /// Recompact the nodes because they are not copied properly when instantiating
  /// a validation reader using the train reader copy constructor
  void compact_nodes();

protected :

  void copy_members(const data_store_jag& rhs, const std::vector<int>& = std::vector<int>());

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

  void exchange_mini_batch_data(size_t current_pos, size_t mb_size) override {
    if (m_super_node) {
      exchange_data_by_super_node(current_pos, mb_size);
    } else {
      exchange_data_by_sample(current_pos, mb_size);
    }
    ++m_n;
  }
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
  void build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out);

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
};

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT

#endif  // __DATA_STORE_JAG_HPP__
