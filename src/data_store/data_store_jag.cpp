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

std::ofstream debug;
char b[1024];

data_store_jag::data_store_jag(
  generic_data_reader *reader, model *m) :
  generic_data_store(reader, m),
  m_super_node(false) {
  set_name("data_store_jag");
}

data_store_jag::~data_store_jag() {
  debug.close();
}

void data_store_jag::setup() {
  double tm1 = get_time();
  std::stringstream err;

  if (m_master) {
    std::cout << "starting data_store_jag::setup() for role: " << m_reader->get_role() << "\n";
  }

  // I suspect we'll never go out-of-memory ...
  if (! m_in_memory) {
    LBANN_ERROR("out-of-memory mode for data_store_jag has not been implemented");
  }

  generic_data_store::setup();
  build_owner_map();

  m_super_node = options::get()->get_bool("super_node");
  if (m_master) {
    if (m_super_node) {
      std::cerr << "mode: exchange_data via super nodes\n";
    } else {
      std::cerr << "mode: exchange_data via individual samples\n";
    }
  }

  sprintf(b, "debug.%d", m_rank);
  debug.open(b);

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
  m_status.resize(m_np);
  m_outgoing_msg_sizes.resize(m_np);
  m_incoming_msg_sizes.resize(m_np);
  m_recv_buffer.resize(m_np);

  m_reconstituted.resize(m_data.size());
}

// this gets called at the beginning of each epoch (except for epoch 0)
//
// Note: conduit has a very nice interface for communicating nodes
//       in non-blocking scenarios. Unf, for blocking we need to
//       handle things ourselves. TODO: possible modify conduit to
//       handle non-blocking comms
void data_store_jag::exchange_data_by_super_node(size_t current_pos, size_t mb_size) {
  double tm1 = get_time();


  //========================================================================
  //part 1: exchange the sizes of the data
  // m_send_buffer[j] is a conduit::Node that contains
  // all samples that this proc will send to P_j

  double tma = get_time();
  build_indices_i_will_send(current_pos, mb_size);

  // construct a super node for each processor; the super node
  // contains all samples this proc owns that other procs need
  for (int p=0; p<m_np; p++) {
    m_send_buffer[p].reset();
    for (auto idx : m_indices_to_send[p]) {
      m_send_buffer[p].update_external(m_data[idx]);
    }
    build_node_for_sending(m_send_buffer[p], m_send_buffer_2[p]);

    // start sends for sizes of the data (sizes of the super_node)
    // @TODO: if all sample nodes are the same size, this may not be
    // necessary: revisit
    m_outgoing_msg_sizes[p] = m_send_buffer_2[p].total_bytes_compact();
    MPI_Isend((void*)&m_outgoing_msg_sizes[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &m_send_requests[p]);
  }

  //start receives for sizes of the data (sizes of the super_nodes)
  for (int p=0; p<m_np; p++) {
    MPI_Irecv((void*)&m_incoming_msg_sizes[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &m_recv_requests[p]);

  }

  // wait for all msgs to complete
  MPI_Waitall(m_np, m_send_requests.data(), m_status.data());
  MPI_Waitall(m_np, m_recv_requests.data(), m_status.data());

  debug << "TOTAL Time to exchange data sizes: " << get_time() -  tma << "\n\n";

  //========================================================================
  //part 2: exchange the actual data
  tma = get_time();

  // start sends for outgoing data
  for (int p=0; p<m_np; p++) {
    const void *s = m_send_buffer_2[p].data_ptr();
    MPI_Isend(s, m_outgoing_msg_sizes[p], MPI_BYTE, p, 1, MPI_COMM_WORLD, &m_send_requests[p]);
  }

  // start recvs for incoming data
  for (int p=0; p<m_np; p++) {
    m_recv_buffer[p].set(conduit::DataType::uint8(m_incoming_msg_sizes[p]));
    MPI_Irecv(m_recv_buffer[p].data_ptr(), m_incoming_msg_sizes[p], MPI_BYTE, p, 1, MPI_COMM_WORLD, &m_recv_requests[p]);
  }

  // wait for all msgs to complete
  MPI_Waitall(m_np, m_send_requests.data(), m_status.data());
  MPI_Waitall(m_np, m_recv_requests.data(), m_status.data());

   debug << "TOTAL Time to exchange the actual data: " << get_time() -  tma << "\n";

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

  double tmw = get_time();

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
    //nd.reset();
    //nd.update_external(n_msg["data"]);
    m_reconstituted[p].reset();
    m_reconstituted[p].update_external(n_msg["data"]);
    const std::vector<std::string> &names = m_reconstituted[p].child_names();

    for (auto t : names) {
      m_minibatch_data[atoi(t.c_str())][t].update_external(m_reconstituted[p][t]);
    }
  }

  debug << "TOTAL Time to unpack and break up all incoming data: " << get_time() - tmw << "\n";

  if (m_master) std::cout << "data_store_jag::exchange_data Time: " << get_time() - tm1 << "\n";

  debug << "TOTAL exchange_data Time: " << get_time() - tm1 << "\n";
}

void data_store_jag::set_conduit_node(int data_id, conduit::Node &node) {
  if (m_data.find(data_id) != m_data.end()) {
    LBANN_ERROR("duplicate data_id: " + std::to_string(data_id) + " in data_store_jag::set_conduit_node");
  }

  if (! m_super_node) {
    //@TODO fix, so we don't need to do a deep copy
    conduit::Node n2;
    build_node_for_sending(node, n2);
    m_data[data_id] = n2;
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
    return t->second;
  }

  std::unordered_map<int, conduit::Node>::const_iterator t2 = m_minibatch_data.find(data_id);
  if (t2 == m_minibatch_data.end()) {
    LBANN_ERROR("failed to find data_id: " + std::to_string(data_id) + " in m_minibatch_data; m_minibatch_data.size: " + std::to_string(m_minibatch_data.size()) + "; epoch:"  + std::to_string(m_model->get_cur_epoch()));
  }

  return t2->second;
}

// code in the following method is a modification of code from
// conduit/src/libs/relay/conduit_relay_mpi.cpp
void data_store_jag::build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out) {
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
}


void data_store_jag::exchange_data_by_sample(size_t current_pos, size_t mb_size) {
#if 0
  double tm1 = get_time();

  debug.open(b, std::ios::app);
  debug << "\n============================================================\n"
  <<"starting exchange_data_by_sample; epoch: "<<m_model->get_cur_epoch()<< " data size: "<<m_data.size()<<"  m_n: " << m_n << "  send_buffer size: " << m_send_buffer.size() << "\n";
  debug.close();

  if (m_n == 1) {
    if (m_master) std::cerr << "allocating storage\n";
    int sz = m_data.size();
    m_send_buffer.resize(sz);
    m_send_requests.resize(sz);
    m_recv_requests.resize(sz);
    m_recv_buffer.resize(sz);
    m_status.resize(sz);

    // sanity check
    /*
    int n = 0;
    for (auto t : m_data) {
      if (t.second.total_bytes_compact() != n) {
        LBANN_ERROR("t.total_bytes_compact() != n; " + std::to_string(n) + " " + std::to_string(t.second.total_bytes_compact()));
      }
    }
    */
  }

  //========================================================================
  // build map: proc -> global indices that P_x needs for this epoch, and
  //                    which I own
  // build map: owner -> set of indices I need that owner has

double tma = get_time();

  std::vector<std::unordered_set<int>> proc_to_indices(m_np);
  // get indices that I need for this epoch; these correspond to
  // samples that this proc receives from others
  std::unordered_map<int, std::unordered_set<int>> needed;
  {
  size_t j = 0;
  for (auto i = current_pos; i < current_pos + mb_size; i++) {
    auto index = (*m_shuffled_indices)[i];
    /// If this rank owns the index send it to the j'th rank
    if (m_data.find(index) != m_data.end()) {
      proc_to_indices[j].insert(index);
    }
    if(j == static_cast<size_t>(m_rank)) {
      int owner = m_owner[index];
      needed[owner].insert(index);
    }
    j = (j + 1) % m_np;
  }
  }

  int sample_size = 0;
  for (auto t : m_data) {
    if(sample_size == 0) {
      sample_size = t.second.total_bytes_compact();
    } else {
      if(sample_size != t.second.total_bytes_compact()) {
        debug << "bad sample size: " << t.second.total_bytes_compact() << " num samples: " << m_data.size() << "\n";
      }
    }
  }
  debug << "sample size: " << sample_size << " num samples: " << m_data.size() << "\n";
  debug.close();
  debug.open(b, std::ios::app);


  //========================================================================
  //part 2: exchange the actual data

tma = get_time();

  // start sends for outgoing data
  size_t ss = 0;
  for (int p=0; p<m_np; p++) {
    const std::unordered_set<int> &indices = proc_to_indices[p];
    for (auto index : indices) {
      if (m_data.find(index) == m_data.end()) {
        LBANN_ERROR("failed to find data_id: " + std::to_string(index) + " to be sent to " + std::to_string(p) + " in m_data");
      }

      //const void *s = m_send_buffer[ss].data_ptr();
      const void *s = m_data[index].data_ptr();
      MPI_Isend(s, sample_size, MPI_BYTE, p, index, MPI_COMM_WORLD, &m_send_requests[ss++]);
      //MPI_Isend(s, m_outgoing_msg_sizes[p], MPI_BYTE, p, 1, MPI_COMM_WORLD, &m_send_requests[p]);
    }
  }
  LBANN_ERROR("Stopping");

  // sanity checks
  if (ss != m_send_requests.size()) {
    LBANN_ERROR("ss != m_send_requests.size; ss: " + std::to_string(ss) + " m_send_requests`.size: " + std::to_string(m_send_requests.size()));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (m_master) std::cerr << "\nSENDS STARTED\n\n";
  debug << "\nSENDS STARTED\n\n";
  MPI_Barrier(MPI_COMM_WORLD);


  // start recvs for incoming data
  ss = 0;
  for (int p=0; p<m_np; p++) {
    const std::unordered_set<int> &indices = needed[p];
debug << "starting " << indices.size() << " recvs from " << p << "\n";
    for (auto index : indices) {
      m_recv_buffer[ss].set(conduit::DataType::uint8(sample_size));
      MPI_Irecv(m_recv_buffer[ss].data_ptr(), sample_size, MPI_BYTE, p, index, MPI_COMM_WORLD, &m_recv_requests[ss]);
      m_index_to_data_id[index] = ss;
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
  MPI_Waitall(m_send_requests.size(), m_send_requests.data(), m_status.data());
  MPI_Waitall(m_recv_requests.size(), m_recv_requests.data(), m_status.data());

debug << "TOTAL Time to exchange the actual data: " << get_time() -  tma << "\n";
debug.close();
debug.open(b, std::ios::app);

tma = get_time();

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

double tmw = get_time();

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

    // this is inefficent @TODO
    nd.reset();
    nd.update(n_msg["data"]);
    m_minibatch_data[nd["id"].value()] = nd;
  }
for (auto t : m_minibatch_data) {
  debug << t.first << " ";
}
debug << "\n";

debug << "TOTAL Time to unpack incoming data: " << get_time() - tmw << "\n";

  if (m_master) std::cout << "data_store_jag::exchange_data Time: " << get_time() - tm1 << "\n";

  debug << "TOTAL exchange_data Time: " << get_time() - tm1 << "\n";
debug.close(); debug.open(b, std::ios::app);
#endif
}

#if 0
// fills in m_ds_indices and m_owner
void data_store_jag::build_ds_indices() {
  m_owner.clear();
  m_ds_indices.clear();
  m_ds_indices.resize(m_np);

  std::vector<std::unordered_set<int>> proc_to_indices(m_np);
  size_t j = 0;
  for (size_t i = 0; i < m_shuffled_indices->size(); i++) {
    auto index = (*m_shuffled_indices)[i];
    m_ds_indices[j].insert(index);
    m_owner[index] = j;
    j = (j + 1) % m_np;
  }
}
#endif

void data_store_jag::build_indices_i_will_recv(int current_pos, int mb_size) {
  for (size_t j=m_rank; j<m_shuffled_indices->size(); j += m_np) {
    auto index = (*m_shuffled_indices)[j];
    int owner = m_owner[index];
    m_indices_to_recv[owner].insert(index);
    j = (j + 1) % m_np;
  }
}

void data_store_jag::build_indices_i_will_send(int current_pos, int mb_size) {
  m_indices_to_send.clear();
  m_indices_to_send.resize(m_np);
  size_t j = 0;
  for (auto i = current_pos; i < current_pos + mb_size; i++) {
    auto index = (*m_shuffled_indices)[i];
    /// If this rank owns the index send it to the j'th rank
    if (m_data.find(index) != m_data.end()) {
      m_indices_to_send[j].insert(index);
    }
    j = (j + 1) % m_np;
  }
}

void data_store_jag::build_owner_map() {
  m_owner.clear();
  size_t j = 0;
  for (size_t i = 0; i < m_shuffled_indices->size(); i++) {
    auto index = (*m_shuffled_indices)[i];
    m_owner[index] = j;
    j = (j + 1) % m_np;
  }
}

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT
