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

#include "lbann/data_store/data_store_jag.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"
#include <unordered_set>

namespace lbann {

data_store_jag::data_store_jag(
  generic_data_reader *reader, model *m) :
  generic_data_store(reader, m),
  m_super_node(false),
  m_super_node_overhead(0),
  m_compacted_sample_size(0) {
  set_name("data_store_jag");
}

data_store_jag::~data_store_jag() {}

void data_store_jag::setup(int mini_batch_size) {
  double tm1 = get_time();
  std::stringstream err;

  if (m_master) {
    std::cout << "starting data_store_jag::setup() for role: " << m_reader->get_role() << "\n";
  }

  // I suspect we'll never go out-of-memory ...
  if (! m_in_memory) {
    LBANN_ERROR("out-of-memory mode for data_store_jag has not been implemented");
  }

  generic_data_store::setup(mini_batch_size);
  build_owner_map(mini_batch_size);

  m_super_node = options::get()->get_bool("super_node");
  if (m_master) {
    if (m_super_node) {
      std::cerr << "mode: exchange_data via super nodes\n";
    } else {
      std::cerr << "mode: exchange_data via individual samples\n";
    }
  }

  if (m_master) {
    std::cout << "num shuffled_indices: " << m_shuffled_indices->size() << "\n";
  }

  data_reader_jag_conduit *jag_reader = dynamic_cast<data_reader_jag_conduit*>(m_reader);
  if (jag_reader == nullptr) {
    LBANN_ERROR(" dynamic_cast<data_reader_jag_conduit*>(m_reader) failed");
  }

  if (m_master) {
    std::cout << "TIME for data_store_jag setup: " << get_time() - tm1 << "\n";
  }
}

void data_store_jag::setup_data_store_buffers() {
  // allocate buffers that are used in exchange_data()
  m_send_buffer.resize(m_np);
  m_send_buffer_2.resize(m_np);
  m_send_requests.resize(m_np);
  m_recv_requests.resize(m_np);
  m_outgoing_msg_sizes.resize(m_np);
  m_incoming_msg_sizes.resize(m_np);
  m_recv_buffer.resize(m_np);
  m_reconstituted.resize(m_np);
}

// Note: conduit has a very nice interface for communicating nodes
//       in blocking scenarios. Unf, for non-blocking we need to
//       handle things ourselves. TODO: possibly modify conduit to
//       handle non-blocking comms
void data_store_jag::exchange_data_by_super_node(size_t current_pos, size_t mb_size) {

  if (m_n == 0) {
    setup_data_store_buffers();
  }

  //========================================================================
  //part 1: construct the super_nodes

  build_indices_i_will_send(current_pos, mb_size);
  build_indices_i_will_recv(current_pos, mb_size);

  // construct a super node for each processor; the super node
  // contains all samples this proc owns that other procs need
  for (int p=0; p<m_np; p++) {
    m_send_buffer[p].reset();
    for (auto idx : m_indices_to_send[p]) {
      m_send_buffer[p].update_external(m_data[idx]);
    }
    build_node_for_sending(m_send_buffer[p], m_send_buffer_2[p]);
  }

  //========================================================================
  //part 1.5: exchange super_node sizes

  for (int p=0; p<m_np; p++) {
    m_outgoing_msg_sizes[p] = m_send_buffer_2[p].total_bytes_compact();
    El::byte *s = reinterpret_cast<El::byte*>(&m_outgoing_msg_sizes[p]);
    m_comm->nb_send<El::byte>(s, sizeof(int), m_comm->get_trainer_rank(), p, m_send_requests[p]);
  }

  for (int p=0; p<m_np; p++) {
    El::byte *s = reinterpret_cast<El::byte*>(&m_incoming_msg_sizes[p]);
    m_comm->nb_recv<El::byte>(s, sizeof(int), m_comm->get_trainer_rank(), p, m_recv_requests[p]);
  }
  m_comm->wait_all<El::byte>(m_send_requests);
  m_comm->wait_all<El::byte>(m_recv_requests);

  //========================================================================
  //part 2: exchange the actual data

  // start sends for outgoing data
  for (int p=0; p<m_np; p++) {
    const El::byte *s = reinterpret_cast<El::byte*>(m_send_buffer_2[p].data_ptr());
    m_comm->nb_send<El::byte>(s, m_outgoing_msg_sizes[p], m_comm->get_trainer_rank(), p, m_send_requests[p]);
  }

  // start recvs for incoming data
  for (int p=0; p<m_np; p++) {
    m_recv_buffer[p].set(conduit::DataType::uint8(m_incoming_msg_sizes[p]));
    m_comm->nb_recv<El::byte>((El::byte*)m_recv_buffer[p].data_ptr(), m_incoming_msg_sizes[p], m_comm->get_trainer_rank(), p, m_recv_requests[p]);
  }

  // wait for all msgs to complete
  m_comm->wait_all<El::byte>(m_send_requests);
  m_comm->wait_all<El::byte>(m_recv_requests);

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

  m_minibatch_data.clear();
  for (int p=0; p<m_np; p++) {
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

void data_store_jag::set_conduit_node(int data_id, conduit::Node &node) {
  if (m_data.find(data_id) != m_data.end()) {
    LBANN_ERROR("duplicate data_id: " + std::to_string(data_id) + " in data_store_jag::set_conduit_node");
  }

  if (m_owner[data_id] != m_rank) {
    std::stringstream s;
    s << "set_conduit_node error for data id: "<<data_id<< " m_owner: " << m_owner[data_id] << " me: " << m_rank;
    LBANN_ERROR(s.str());
  }

  if (! m_super_node) {
    build_node_for_sending(node, m_data[data_id]);
    const conduit::Node& n2 = m_data[data_id];
    if(m_compacted_sample_size == 0) {
      m_compacted_sample_size = n2.total_bytes_compact();
    }else if(m_compacted_sample_size != n2.total_bytes_compact()) {
      LBANN_ERROR("Conduit node being added data_id: " + std::to_string(data_id)
                  + " is not the same size as existing nodes in the data_store "
                  + std::to_string(m_compacted_sample_size) + " != "
                  + std::to_string(n2.total_bytes_compact()));
    }
    if(!m_data[data_id].is_contiguous()) {
      LBANN_ERROR("m_data[" + std::to_string(data_id) + "] does not have a contiguous layout");
    }
    if(m_data[data_id].data_ptr() == nullptr) {
      LBANN_ERROR("m_data[" + std::to_string(data_id) + "] does not have a valid data pointer");
    }
    if(m_data[data_id].contiguous_data_ptr() == nullptr) {
      LBANN_ERROR("m_data[" + std::to_string(data_id) + "] does not have a valid contiguous data pointer");
    }
  }

  else {
    m_data[data_id] = node;
    // @TODO would like to do: m_data[data_id].set_external(node); but since
    // (as of now) 'node' is a local variable in a data_reader+jag_conduit,
    // we need to do a deep copy. If the data_store furnishes a node to the
    // data_reader during the first epoch, this copy can be avoided
  }
}

const conduit::Node & data_store_jag::get_conduit_node(int data_id) const {
  std::unordered_map<int, conduit::Node>::const_iterator t = m_data.find(data_id);
  if (t != m_data.end()) {
    if(m_super_node) {
      return t->second;
    } else {
      return t->second["data"];
    }
  }

  std::unordered_map<int, conduit::Node>::const_iterator t2 = m_minibatch_data.find(data_id);
  if (t2 == m_minibatch_data.end()) {
    LBANN_ERROR("failed to find data_id: " + std::to_string(data_id) + " in m_minibatch_data; m_minibatch_data.size: " + std::to_string(m_minibatch_data.size()) + "; epoch:"  + std::to_string(m_model->get_epoch()));
  }

  return t2->second;
}

// code in the following method is a modification of code from
// conduit/src/libs/relay/conduit_relay_mpi.cpp
void data_store_jag::build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out) {
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

void data_store_jag::exchange_data_by_sample(size_t current_pos, size_t mb_size) {
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
  for (int p=0; p<m_np; p++) {
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
  for (int p=0; p<m_np; p++) {
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

int data_store_jag::build_indices_i_will_recv(int current_pos, int mb_size) {
  m_indices_to_recv.clear();
  m_indices_to_recv.resize(m_np);
  int k = 0;
  for (int i=current_pos; i< current_pos + mb_size; ++i) {
    auto index = (*m_shuffled_indices)[i];
    if ((i % mb_size) % m_np == m_rank) {
      int owner = m_owner[index];
      m_indices_to_recv[owner].insert(index);
      k++;
    }
  }
  return k;
}

int data_store_jag::build_indices_i_will_send(int current_pos, int mb_size) {
  m_indices_to_send.clear();
  m_indices_to_send.resize(m_np);
  int k = 0;
  for (int i = current_pos; i < current_pos + mb_size; i++) {
    auto index = (*m_shuffled_indices)[i];
    /// If this rank owns the index send it to the (i%m_np)'th rank
    if (m_data.find(index) != m_data.end()) {
      m_indices_to_send[(i % mb_size) % m_np].insert(index);

      // Sanity check
      if (m_owner[index] != m_rank) {
        std::stringstream s;
        s << "error for i: "<<i<<" index: "<<index<< " m_owner: " << m_owner[index] << " me: " << m_rank;
        LBANN_ERROR(s.str());
      }
      k++;
    }
  }
  return k;
}

void data_store_jag::build_owner_map(int mini_batch_size) {
  m_owner.clear();
  for (size_t i = 0; i < m_shuffled_indices->size(); i++) {
    auto index = (*m_shuffled_indices)[i];
    /// To compute the owner index first find its position inside of
    /// the mini-batch (mod mini-batch size) and then find how it is
    /// striped across the ranks in the trainer
    m_owner[index] = (i % mini_batch_size) % m_np;
  }
}

void data_store_jag::compute_super_node_overhead() {
  if (m_super_node_overhead != 0) {
    return;
  }
  if (m_data.size() < 2) {
    LBANN_ERROR("m_data must contain at least two sample nodes");
  }
  conduit::Node n2;
  conduit::Node n3;
  int first = 0;
  for (auto &t : m_data) {
    n2.update_external(t.second);
    build_node_for_sending(n2, n3);
    if (first == 0) {
      first = n3.total_bytes_compact();
    } else {
      m_super_node_overhead = 2*first - n3.total_bytes_compact();
      m_compacted_sample_size = first - m_super_node_overhead;
      if (m_master) {
        std::cerr << "m_super_node_overhead: " << m_super_node_overhead
                  << " m_compacted_sample_size: " << m_compacted_sample_size << "\n";
      }
      return;
    }
  }
}

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT
