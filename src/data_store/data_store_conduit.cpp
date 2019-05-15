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

#include "lbann/data_store/data_store_conduit.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include <unordered_set>

namespace lbann {

data_store_conduit::data_store_conduit(
  generic_data_reader *reader) :
  m_n(0),
  m_is_setup(false),
  m_reader(reader),
  m_preload(false),
  m_explicit_loading(false),
  m_owner_map_mb_size(0),
  m_super_node(false),
  m_compacted_sample_size(0),
  m_is_local_cache(false) {
  m_comm = m_reader->get_comm();
  if (m_comm == nullptr) {
    LBANN_ERROR(" m_comm is nullptr");
  }

  m_world_master = m_comm->am_world_master();
  m_trainer_master = m_comm->am_trainer_master();
  m_rank_in_trainer = m_comm->get_rank_in_trainer();
  m_np_in_trainer = m_comm->get_procs_per_trainer();

  options *opts = options::get();
  m_super_node = opts->get_bool("super_node");

  m_is_local_cache = opts->get_bool("data_store_cache");
  if (m_is_local_cache && opts->get_bool("preload_data_store")) {
    LBANN_ERROR("you cannot use both of these options: --data_store_cache --preload_data_store");
  }
}

data_store_conduit::~data_store_conduit() {}

data_store_conduit::data_store_conduit(const data_store_conduit& rhs) {
  copy_members(rhs);
}

data_store_conduit::data_store_conduit(const data_store_conduit& rhs, const std::vector<int>& ds_sample_move_list) {

  copy_members(rhs, ds_sample_move_list);
}

data_store_conduit& data_store_conduit::operator=(const data_store_conduit& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  copy_members(rhs);
  return (*this);
}

void data_store_conduit::copy_members(const data_store_conduit& rhs, const std::vector<int>& ds_sample_move_list) {
  m_n = rhs.m_n;
  m_is_setup = rhs.m_is_setup;
  m_reader = rhs.m_reader;
  m_comm = rhs.m_comm;
  m_rank_in_trainer = rhs.m_rank_in_trainer;
  m_np_in_trainer = rhs.m_np_in_trainer;
  m_world_master = rhs.m_world_master;
  m_trainer_master = rhs.m_trainer_master;
  m_preload = rhs.m_preload;
  m_explicit_loading = rhs.m_explicit_loading;
  m_owner = rhs.m_owner;
  m_shuffled_indices = rhs.m_shuffled_indices;
  m_owner_map_mb_size = rhs.m_owner_map_mb_size;
  m_super_node = rhs.m_super_node;
  m_compacted_sample_size = rhs.m_compacted_sample_size;
  m_is_local_cache = rhs.m_is_local_cache;

  if(ds_sample_move_list.size() == 0) {
    m_data = rhs.m_data;
  } else {
    /// Move indices on the list from the data and owner maps in the RHS data store to the new data store
    for(auto&& i : ds_sample_move_list) {
      if(rhs.m_data.find(i) != rhs.m_data.end()){
        conduit::Node node = rhs.m_data[i]["data"];
        rhs.m_data.erase(i);
        /// Repack the nodes because they don't seem to copy correctly
        build_node_for_sending(node, m_data[i]);
      }
      /// Removed migrated nodes from the original data store's owner list
      if(rhs.m_owner.find(i) != rhs.m_owner.end()) {
        m_owner[i] = rhs.m_owner[i];
        rhs.m_owner.erase(i);
      }
    }
  }


  /// Clear the pointer to the data reader, this cannot be copied
  m_reader = nullptr;
  m_shuffled_indices = nullptr;

  //these will probably zero-length, but I don't want to make assumptions
  //as to state when copy_member is called
  m_minibatch_data = rhs.m_minibatch_data;
  m_send_buffer = rhs.m_send_buffer;
  m_send_buffer_2 = rhs.m_send_buffer_2;
  m_send_requests = rhs.m_send_requests;
  m_recv_requests = rhs.m_recv_requests;
  m_recv_buffer = rhs.m_recv_buffer;
  m_outgoing_msg_sizes = rhs.m_outgoing_msg_sizes;
  m_incoming_msg_sizes = rhs.m_incoming_msg_sizes;
  m_compacted_sample_size = rhs.m_compacted_sample_size;
  m_reconstituted = rhs.m_reconstituted;
  m_indices_to_send = rhs.m_indices_to_send;
  m_indices_to_recv = rhs.m_indices_to_recv;
}

void data_store_conduit::setup(int mini_batch_size) {

  if (m_world_master) {
    if (m_super_node) {
      std::cout << "data store mode: exchange_data via super nodes\n";
    } else {
      std::cout << "data store mode: exchange_data via individual samples\n";
    }
  }

  double tm1 = get_time();
  if (m_world_master && !m_preload) {
    std::cout << "starting data_store_conduit::setup() for role: " << m_reader->get_role() << "\n";
  }

  if (!m_preload) {
    // generic_data_store::setup(mini_batch_size);
    build_owner_map(mini_batch_size);
  } else {
    m_owner_map_mb_size = mini_batch_size;
  }

  m_is_setup = true;

  if (m_world_master && !m_preload) {
    std::cout << "TIME for data_store_conduit setup: " << get_time() - tm1 << "\n";
  }
}

void data_store_conduit::setup_data_store_buffers() {
  // allocate buffers that are used in exchange_data()
  m_send_buffer.resize(m_np_in_trainer);
  m_send_buffer_2.resize(m_np_in_trainer);
  m_send_requests.resize(m_np_in_trainer);
  m_recv_requests.resize(m_np_in_trainer);
  m_outgoing_msg_sizes.resize(m_np_in_trainer);
  m_incoming_msg_sizes.resize(m_np_in_trainer);
  m_recv_buffer.resize(m_np_in_trainer);
  m_reconstituted.resize(m_np_in_trainer);
}

// Note: conduit has a very nice interface for communicating nodes
//       in blocking scenarios. Unf, for non-blocking we need to
//       handle things ourselves. TODO: possibly modify conduit to
//       handle non-blocking comms
void data_store_conduit::exchange_data_by_super_node(size_t current_pos, size_t mb_size) {

  if (! m_is_setup) {
    LBANN_ERROR("setup(mb_size) has not been called");
  }

  if (m_n == 0) {
    setup_data_store_buffers();
  }

  //========================================================================
  //part 1: construct the super_nodes

  build_indices_i_will_send(current_pos, mb_size);
  build_indices_i_will_recv(current_pos, mb_size);

  // construct a super node for each processor; the super node
  // contains all samples this proc owns that other procs need
  for (int p=0; p<m_np_in_trainer; p++) {
    m_send_buffer[p].reset();
    for (auto idx : m_indices_to_send[p]) {
      m_send_buffer[p].update_external(m_data[idx]);
    }
    build_node_for_sending(m_send_buffer[p], m_send_buffer_2[p]);
  }

  //========================================================================
  //part 1.5: exchange super_node sizes

  for (int p=0; p<m_np_in_trainer; p++) {
    m_outgoing_msg_sizes[p] = m_send_buffer_2[p].total_bytes_compact();
    El::byte *s = reinterpret_cast<El::byte*>(&m_outgoing_msg_sizes[p]);
    m_comm->nb_send<El::byte>(s, sizeof(int), m_comm->get_trainer_rank(), p, m_send_requests[p]);
  }

  for (int p=0; p<m_np_in_trainer; p++) {
    El::byte *s = reinterpret_cast<El::byte*>(&m_incoming_msg_sizes[p]);
    m_comm->nb_recv<El::byte>(s, sizeof(int), m_comm->get_trainer_rank(), p, m_recv_requests[p]);
  }
  m_comm->wait_all<El::byte>(m_send_requests);
  m_comm->wait_all<El::byte>(m_recv_requests);

  //========================================================================
  //part 2: exchange the actual data

  // start sends for outgoing data
  for (int p=0; p<m_np_in_trainer; p++) {
    const El::byte *s = reinterpret_cast<El::byte*>(m_send_buffer_2[p].data_ptr());
    m_comm->nb_send<El::byte>(s, m_outgoing_msg_sizes[p], m_comm->get_trainer_rank(), p, m_send_requests[p]);
  }

  // start recvs for incoming data
  for (int p=0; p<m_np_in_trainer; p++) {
    m_recv_buffer[p].set(conduit::DataType::uint8(m_incoming_msg_sizes[p]));
    m_comm->nb_recv<El::byte>((El::byte*)m_recv_buffer[p].data_ptr(), m_incoming_msg_sizes[p], m_comm->get_trainer_rank(), p, m_recv_requests[p]);
  }

  // wait for all msgs to complete
  m_comm->wait_all<El::byte>(m_send_requests);
  m_comm->wait_all<El::byte>(m_recv_requests);

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

  m_minibatch_data.clear();
  for (int p=0; p<m_np_in_trainer; p++) {
    conduit::uint8 *n_buff_ptr = (conduit::uint8*)m_recv_buffer[p].data_ptr();
    conduit::Node n_msg;
    n_msg["schema_len"].set_external((conduit::int64*)n_buff_ptr);
    n_buff_ptr +=8;
    n_msg["schema"].set_external_char8_str((char*)(n_buff_ptr));
    conduit::Schema rcv_schema;
    conduit::Generator gen(n_msg["schema"].as_char8_str());
    gen.walk(rcv_schema);
    n_buff_ptr += n_msg["schema"].total_bytes_compact();
    n_msg["data"].set_external(rcv_schema,n_buff_ptr);
    m_reconstituted[p].reset();

    // I'm unsure what happens here: m_reconstituted is persistent, but
    // we're updating from n_msg, which is transitory. Best guess,
    // when n_msg goes out of scope a deep copy is made. Possibly
    // there's room for optimization here.
    m_reconstituted[p].update_external(n_msg["data"]);
    const std::vector<std::string> &names = m_reconstituted[p].child_names();

    for (auto &t : names) {
      m_minibatch_data[atoi(t.c_str())][t].update_external(m_reconstituted[p][t]);
    }
  }
}

void data_store_conduit::set_preloaded_conduit_node(int data_id, conduit::Node &node) {
  // note: at this point m_data[data_id] = node
  // note: if running in super_node mode, nothing to do
  if (!m_super_node) {
    conduit::Node n2 = node;
    build_node_for_sending(n2, m_data[data_id]);
    error_check_compacted_node(m_data[data_id], data_id);
  }
}

void data_store_conduit::error_check_compacted_node(const conduit::Node &nd, int data_id) {
  if(m_compacted_sample_size == 0) {
    m_compacted_sample_size = nd.total_bytes_compact();
  } else if(m_compacted_sample_size != nd.total_bytes_compact()) {
    LBANN_ERROR("Conduit node being added data_id: " + std::to_string(data_id)
                + " is not the same size as existing nodes in the data_store "
                + std::to_string(m_compacted_sample_size) + " != "
                + std::to_string(nd.total_bytes_compact())
                + " role: " + m_reader->get_role());
  }
  if(!nd.is_contiguous()) {
    LBANN_ERROR("m_data[" + std::to_string(data_id) + "] does not have a contiguous layout");
  }
  if(nd.data_ptr() == nullptr) {
    LBANN_ERROR("m_data[" + std::to_string(data_id) + "] does not have a valid data pointer");
  }
  if(nd.contiguous_data_ptr() == nullptr) {
    LBANN_ERROR("m_data[" + std::to_string(data_id) + "] does not have a valid contiguous data pointer");
  }
}


void data_store_conduit::set_conduit_node(int data_id, conduit::Node &node, bool already_have) {
  if (already_have == false && m_data.find(data_id) != m_data.end()) {
    LBANN_ERROR("duplicate data_id: " + std::to_string(data_id) + " in data_store_conduit::set_conduit_node");
  }

  if (already_have && is_local_cache()) {
    if (m_data.find(data_id) == m_data.end()) {
      LBANN_ERROR("you claim the passed node was obtained from this data_store, but the data_id (" + std::to_string(data_id) + ") doesn't exist in m_data");
    }
    return;
  }

  if (m_owner[data_id] != m_rank_in_trainer) {
    std::stringstream s;
    s << "set_conduit_node error for data id: "<<data_id<< " m_owner: " << m_owner[data_id] << " me: " << m_rank_in_trainer << "; data reader role: " << m_reader->get_role() << "\n";
    LBANN_ERROR(s.str());
  }

  if (is_local_cache()) {
    m_data[data_id] = node;
  }

  else if (! m_super_node) {
    build_node_for_sending(node, m_data[data_id]);
    error_check_compacted_node(m_data[data_id], data_id);
  }

  else {
    m_data[data_id] = node;
    // @TODO would like to do: m_data[data_id].set_external(node); but since
    // (as of now) 'node' is a local variable in a data_reader+jag_conduit,
    // we need to do a deep copy. If the data_store furnishes a node to the
    // data_reader during the first epoch, this copy can be avoided
  }
}

const conduit::Node & data_store_conduit::get_conduit_node(int data_id) const {
  /**
   * dah: commenting this out since it gives a false positive for test
   *      case with unshuffled indices. Since we currently send samples
   *      to ourselves, they should be in m_minibatch_data. The following
   *      block is only useful if, at some future time, we do not send
   *      indices to ourself
  std::unordered_map<int, conduit::Node>::const_iterator t = m_data.find(data_id);
  if (t != m_data.end()) {
    if(m_super_node) {
      return t->second;
    } else {
      return t->second["data"];
    }
  }
  */
  if (is_local_cache()) {
    std::unordered_map<int, conduit::Node>::const_iterator t3 = m_data.find(data_id);
    if (t3 == m_data.end()) {
      LBANN_ERROR("failed to find data_id: " + std::to_string(data_id) + " in m_data; m_data.size: " + std::to_string(m_data.size()));
    }
    return t3->second;
  }

  std::unordered_map<int, conduit::Node>::const_iterator t2 = m_minibatch_data.find(data_id);
  // if not preloaded, and get_label() or get_response() is called,
  // we need to check m_data
  if (t2 == m_minibatch_data.end()) {
    std::unordered_map<int, conduit::Node>::const_iterator t3 = m_data.find(data_id);
    if (t3 != m_data.end()) {
      return t3->second["data"];
    }
    LBANN_ERROR("failed to find data_id: " + std::to_string(data_id) + " in m_minibatch_data; m_minibatch_data.size: " + std::to_string(m_minibatch_data.size())+ " and also failed to find it in m_data; m_data.size: " + std::to_string(m_data.size()) + "; role: " + m_reader->get_role());
  }

  return t2->second;
}

// code in the following method is a modification of code from
// conduit/src/libs/relay/conduit_relay_mpi.cpp
void data_store_conduit::build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out) {

  node_out.reset();
  conduit::Schema s_data_compact;
  if( node_in.is_compact() && node_in.is_contiguous()) {
    s_data_compact = node_in.schema();
  } else {
    node_in.schema().compact_to(s_data_compact);
  }

  std::string snd_schema_json = s_data_compact.to_json();

  conduit::Schema s_msg;
  s_msg["schema_len"].set(conduit::DataType::int64());
  s_msg["schema"].set(conduit::DataType::char8_str(snd_schema_json.size()+1));
  s_msg["data"].set(s_data_compact);

  conduit::Schema s_msg_compact;
  s_msg.compact_to(s_msg_compact);
  node_out.reset();
  node_out.set(s_msg_compact);
  node_out["schema"].set(snd_schema_json);
  node_out["data"].update(node_in);

  if(!node_out.is_contiguous()) {
    LBANN_ERROR("node_out does not have a contiguous layout");
  }
  if(node_out.data_ptr() == nullptr) {
    LBANN_ERROR("node_out does not have a valid data pointer");
  }
  if(node_out.contiguous_data_ptr() == nullptr) {
    LBANN_ERROR("node_out does not have a valid contiguous data pointer");
  }
}

void data_store_conduit::exchange_data_by_sample(size_t current_pos, size_t mb_size) {
  if (! m_is_setup) {
    LBANN_ERROR("setup(mb_size) has not been called");
  }

  int num_send_req = build_indices_i_will_send(current_pos, mb_size);
  int num_recv_req = build_indices_i_will_recv(current_pos, mb_size);

  m_send_requests.resize(num_send_req);
  m_recv_requests.resize(num_recv_req);
  m_recv_buffer.resize(num_recv_req);
  m_recv_data_ids.resize(num_recv_req);

  //========================================================================
  //part 2: exchange the actual data

  // start sends for outgoing data
  size_t ss = 0;
  for (int p=0; p<m_np_in_trainer; p++) {
    const std::unordered_set<int> &indices = m_indices_to_send[p];
    for (auto index : indices) {
      if (m_data.find(index) == m_data.end()) {
        LBANN_ERROR("failed to find data_id: " + std::to_string(index) + " to be sent to " + std::to_string(p) + " in m_data");
      }
      const conduit::Node& n = m_data[index];
      const El::byte *s = reinterpret_cast<const El::byte*>(n.data_ptr());
      if(!n.is_contiguous()) {
        LBANN_ERROR("data_id: " + std::to_string(index) + " does not have a contiguous layout");
      }
      if(n.data_ptr() == nullptr) {
        LBANN_ERROR("data_id: " + std::to_string(index) + " does not have a valid data pointer");
      }
      if(n.contiguous_data_ptr() == nullptr) {
        LBANN_ERROR("data_id: " + std::to_string(index) + " does not have a valid contiguous data pointer");
      }
      m_comm->nb_tagged_send(s, m_compacted_sample_size, p, index, m_send_requests[ss++], m_comm->get_trainer_comm());
    }
  }

  // sanity checks
  if (ss != m_send_requests.size()) {
    LBANN_ERROR("ss != m_send_requests.size; ss: " + std::to_string(ss) + " m_send_requests.size: " + std::to_string(m_send_requests.size()));
  }

  // start recvs for incoming data
  ss = 0;
  for (int p=0; p<m_np_in_trainer; p++) {
    const std::unordered_set<int> &indices = m_indices_to_recv[p];
    for (auto index : indices) {
      m_recv_buffer[ss].set(conduit::DataType::uint8(m_compacted_sample_size));
      El::byte *r = reinterpret_cast<El::byte*>(m_recv_buffer[ss].data_ptr());
      m_comm->nb_tagged_recv<El::byte>(r, m_compacted_sample_size, p, index, m_recv_requests[ss], m_comm->get_trainer_comm());
      m_recv_data_ids[ss] = index;
      ++ss;
    }
  }

  // sanity checks
  if (ss != m_recv_buffer.size()) {
    LBANN_ERROR("ss != m_recv_buffer.size; ss: " + std::to_string(ss) + " m_recv_buffer.size: " + std::to_string(m_recv_buffer.size()));
  }
  if (m_recv_requests.size() != m_recv_buffer.size()) {
    LBANN_ERROR("m_recv_requests.size != m_recv_buffer.size; m_recv_requests: " + std::to_string(m_recv_requests.size()) + " m_recv_buffer.size: " + std::to_string(m_recv_buffer.size()));
  }

  // wait for all msgs to complete
  m_comm->wait_all(m_send_requests);
  m_comm->wait_all(m_recv_requests);

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

  conduit::Node nd;
  m_minibatch_data.clear();
  for (size_t j=0; j < m_recv_buffer.size(); j++) {
    conduit::uint8 *n_buff_ptr = (conduit::uint8*)m_recv_buffer[j].data_ptr();
    conduit::Node n_msg;
    n_msg["schema_len"].set_external((conduit::int64*)n_buff_ptr);
    n_buff_ptr +=8;
    n_msg["schema"].set_external_char8_str((char*)(n_buff_ptr));
    conduit::Schema rcv_schema;
    conduit::Generator gen(n_msg["schema"].as_char8_str());
    gen.walk(rcv_schema);
    n_buff_ptr += n_msg["schema"].total_bytes_compact();
    n_msg["data"].set_external(rcv_schema,n_buff_ptr);

    int data_id = m_recv_data_ids[j];
    m_minibatch_data[data_id].set_external(n_msg["data"]);
  }
}

int data_store_conduit::build_indices_i_will_recv(int current_pos, int mb_size) {
  m_indices_to_recv.clear();
  m_indices_to_recv.resize(m_np_in_trainer);
  int k = 0;
  for (int i=current_pos; i< current_pos + mb_size; ++i) {
    auto index = (*m_shuffled_indices)[i];
    if ((i % m_owner_map_mb_size) % m_np_in_trainer == m_rank_in_trainer) {
      int owner = m_owner[index];
      m_indices_to_recv[owner].insert(index);
      k++;
    }
  }
  return k;
}

int data_store_conduit::build_indices_i_will_send(int current_pos, int mb_size) {
  m_indices_to_send.clear();
  m_indices_to_send.resize(m_np_in_trainer);
  int k = 0;
  for (int i = current_pos; i < current_pos + mb_size; i++) {
    auto index = (*m_shuffled_indices)[i];
    /// If this rank owns the index send it to the (i%m_np)'th rank
    if (m_data.find(index) != m_data.end()) {
      m_indices_to_send[(i % m_owner_map_mb_size) % m_np_in_trainer].insert(index);

      // Sanity check
      if (m_owner[index] != m_rank_in_trainer) {
        std::stringstream s;
        s << "error for i: "<<i<<" index: "<<index<< " m_owner: " << m_owner[index] << " me: " << m_rank_in_trainer;
        LBANN_ERROR(s.str());
      }
      k++;
    }
  }
  return k;
}

void data_store_conduit::build_preloaded_owner_map(const std::vector<int>& per_rank_list_sizes) {
  m_owner.clear();
  int owning_rank = 0;
  size_t per_rank_list_range_start = 0;
  for (size_t i = 0; i < m_shuffled_indices->size(); i++) {
    const auto per_rank_list_size = per_rank_list_sizes[owning_rank];
    if(i == (per_rank_list_range_start + per_rank_list_size)) {
      ++owning_rank;
      per_rank_list_range_start += per_rank_list_size;
    }
    m_owner[i] = owning_rank;
  }
}

void data_store_conduit::build_owner_map(int mini_batch_size) {
  if (m_world_master) std::cout << "starting data_store_conduit::build_owner_map for role: " << m_reader->get_role() << " with mini_batch_size: " << mini_batch_size << "\n";
  if (mini_batch_size == 0) {
    LBANN_ERROR("mini_batch_size == 0; can't build owner_map");
  }
  m_owner.clear();
  m_owner_map_mb_size = mini_batch_size;
  for (size_t i = 0; i < m_shuffled_indices->size(); i++) {
    auto index = (*m_shuffled_indices)[i];
    /// To compute the owner index first find its position inside of
    /// the mini-batch (mod mini-batch size) and then find how it is
    /// striped across the ranks in the trainer
    m_owner[index] = (i % m_owner_map_mb_size) % m_np_in_trainer;
  }
}

const conduit::Node & data_store_conduit::get_random_node() const {
  size_t sz = m_data.size();

  // Deal with edge case
  if (sz == 0) {
    LBANN_ERROR("can't return random node since we have no data (set_conduit_node has never been called)");
  }

  int offset = random() % sz;
  auto it = std::next(m_data.begin(), offset);
  return it->second;
}

const conduit::Node & data_store_conduit::get_random_node(const std::string &field) const {
  auto node = get_random_node();
  //return node;
  return node[field];
}

conduit::Node & data_store_conduit::get_empty_node(int data_id) {
  if (m_data.find(data_id) != m_data.end()) {
    LBANN_ERROR("we already have a node with data_id= " + std::to_string(data_id));
  }
  return m_data[data_id];
}

void data_store_conduit::purge_unused_samples(const std::vector<int>& indices) {
  /// Remove unused indices from the data and owner maps
  for(auto&& i : indices) {
    if(m_data.find(i) != m_data.end()){
      m_data.erase(i);
    }
    if(m_owner.find(i) != m_owner.end()) {
      m_owner.erase(i);
    }
  }
}

void data_store_conduit::compact_nodes() {
  for(auto&& j : *m_shuffled_indices) {
    if(m_data.find(j) != m_data.end()){
      if(!m_data[j].is_contiguous()) {
        /// Repack the nodes because they don't seem to copy correctly
        conduit::Node node = m_data[j]["data"];
        m_data.erase(j);
        build_node_for_sending(node, m_data[j]);
      }
    }
  }
}

int data_store_conduit::get_index_owner(int idx) {
  if (m_owner.find(idx) == m_owner.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << " idx: " << idx << " was not found in the m_owner map;"
        << " map size: " << m_owner.size();
    throw lbann_exception(err.str());
  }
  return m_owner[idx];
}

void data_store_conduit::check_mem_capacity(lbann_comm *comm, const std::string sample_list_file, size_t stride, size_t offset) {
  if (comm->am_world_master()) {
    // note: we only estimate memory required by the data reader/store

    // get avaliable memory
    std::ifstream in("/proc/meminfo");
    std::string line;
    std::string units;
    double a_mem = 0;
    while (getline(in, line)) {
      if (line.find("MemAvailable:")) {
        std::stringstream s3(line);
        s3 >> line >> a_mem >> units;
        if (units != "kB") {
          LBANN_ERROR("units is " + units + " but we only know how to handle kB; please contact Dave Hysom");
        }
        break;
      }
    }
    in.close();
    if (a_mem == 0) {
      LBANN_ERROR("failed to find MemAvailable field in /proc/meminfo");
    }

    // a lot of the following is cut-n-paste from the sample list class;
    // would like to use the sample list class directly, but this
    // is quicker than figuring out how to modify the sample_list.
    // Actually there are at least three calls, starting from
    // data_reader_jag_conduit, before getting to the code that
    // loads the sample list file names

    // get list of conduit files that I own, and compute my num_samples
    std::ifstream istr(sample_list_file);
    if (!istr.good()) {
      LBANN_ERROR("failed to open " + sample_list_file + " for reading");
    }

    std::string base_dir;
    std::getline(istr, line);  //exclusiveness; discard

    std::getline(istr, line);
    std::stringstream s5(line);
    int included_samples;
    int excluded_samples;
    size_t num_files;
    s5 >> included_samples >> excluded_samples >> num_files;

    std::getline(istr, base_dir); // base dir; discard

    const std::string whitespaces(" \t\f\v\n\r");
    size_t cnt_files = 0u;
    int my_sample_count = 0;

    conduit::Node useme;
    bool got_one = false;

    // loop over conduit filenames
    while (std::getline(istr, line)) {
      const size_t end_of_str = line.find_last_not_of(whitespaces);
      if (end_of_str == std::string::npos) { // empty line
        continue;
      }
      if (cnt_files++ >= num_files) {
        break;
      }
      if ((cnt_files-1)%stride != offset) {
        continue;
      }
      std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
      std::string filename;
      sstr >> filename >> included_samples >> excluded_samples;
      my_sample_count += included_samples;

      // attempt to load a JAG sample
      if (!got_one) {
        hid_t hdf5_file_hnd;
        try {
          hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read(base_dir + '/' + filename);
        } catch (conduit::Error const& e) {
          LBANN_ERROR(" failed to open " + base_dir + '/' + filename + " for reading");
        }
        std::vector<std::string> sample_names;
        try {
          conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", sample_names);
        } catch (conduit::Error const& e) {
          LBANN_ERROR("hdf5_group_list_child_names() failed");
        }

        for (auto t : sample_names) {
          std::string key = "/" + t + "/performance/success";
          try {
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, useme);
          } catch (conduit::Error const& e) {
            LBANN_ERROR("failed to read success flag for " + key);
          }
          if (useme.to_int64() == 1) {
            got_one = true;
            try {
              key = "/" + t;
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, useme);
            } catch (conduit::Error const& e) {
              LBANN_ERROR("failed to load JAG sample: " + key);
            }
            break;
          }
        } // end: for (auto t : sample_names)

        conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
      } // end: attempt to load a JAG sample
    } // end: loop over conduit filenames
    istr.close();
    // end: get list of conduit files that I own, and compute my num_samples

    if (! got_one) {
      LBANN_ERROR("failed to find any successful JAG samples");
    }

    // compute memory for the compacted nodes this processor owns
    double bytes_per_sample = useme.total_bytes_compact() / 1024;
    double  procs_per_node = comm->get_procs_per_node();
    double mem_this_proc = bytes_per_sample * my_sample_count;
    double mem_this_node = mem_this_proc * procs_per_node;

    std::cout
      << "\n"
      << "==============================================================\n"
      << "Estimated memory requirements for JAG samples:\n"
      << "Memory for one sample:             " <<  bytes_per_sample << " kB\n"
      << "Total mem for a single rank:       " << mem_this_proc << " kB\n"
      << "Samples per proc:                  " << my_sample_count << "\n"
      << "Procs per node:                    " << procs_per_node << "\n"
      << "Total mem for all ranks on a node: " << mem_this_node << " kB\n"
      << "Available memory: " << a_mem << " kB (RAM only; not virtual)\n";
    if (mem_this_node > static_cast<double>(a_mem)) {
      std::cout << "\nYOU DO NOT HAVE ENOUGH MEMORY\n"
        << "==============================================================\n\n";
      LBANN_ERROR("insufficient memory to load data\n");
    } else {
      double m = 100 * mem_this_node / a_mem;
      std::cout << "Estimate that data will consume at least " << m << " % of memory\n"
        << "==============================================================\n\n";
    }
  }

  comm->trainer_barrier();
}

bool data_store_conduit::has_conduit_node(int data_id) const {
  std::unordered_map<int, conduit::Node>::const_iterator t = m_data.find(data_id);
  return t == m_data.end();
}


}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT
