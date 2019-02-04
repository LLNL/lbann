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
  generic_data_store(reader, m) {
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

  sprintf(b, "debug.%d", m_rank);
  debug.open(b);

  if (m_master) {
    std::cout << "num shuffled_indices: " << m_shuffled_indices->size() << "\n";
  }

  for (size_t j=0; j<m_shuffled_indices->size(); j++) {
    m_unshuffle[(*m_shuffled_indices)[j]] = j;
  }

  data_reader_jag_conduit *jag_reader = dynamic_cast<data_reader_jag_conduit*>(m_reader);
  if (jag_reader == nullptr) {
    LBANN_ERROR(" dynamic_cast<data_reader_jag_conduit*>(m_reader) failed");
  }

  /// m_all_minibatch_indices[j] will contain all indices that
  //  will be passed to data_reader::fetch_datum in one epoch
  build_all_minibatch_indices();

  // allocate buffers that are used in exchange_data()
  m_send_buffer.resize(m_np);
  m_send_buffer_2.resize(m_np);
  m_send_requests.resize(m_np);
  m_recv_requests.resize(m_np);
  m_status.resize(m_np);
  m_outgoing_msg_sizes.resize(m_np);
  m_incoming_msg_sizes.resize(m_np);
  m_recv_buffer.resize(m_np);

  if (m_master) {
    std::cout << "TIME for data_store_jag setup: " << get_time() - tm1 << "\n";
  }
}

// this gets called at the beginning of each epoch (except for epoch 0)
//
// Note: conduit has a very nice interface for communicating nodes
//       in non-blocking scenarios. Unf, for blocking we need to
//       handle things ourselves. TODO: possible modify conduit to
//       handle non-blocking comms
void data_store_jag::exchange_data() {
  double tm1 = get_time();

  debug << "\n============================================================\n"
  <<"starting exchange_data; epoch: "<<m_model->get_cur_epoch()<< " data size: "<<m_data.size()<<"\n";

  //========================================================================
  //build map: proc -> global indices that P_x needs for this epoch, and
  //                   which I own

  //@TODO: change m_all_minibatch_indices from vector<vector<int>> to
  //vector<unordered_set<int>>; then: 
  //  const std::unordered_set<int>> &my_datastore_indices;m_rank]
  //
  //  Hm ... I think m_all_minibatch_indices is identical to ds indices

double tma = get_time();

  std::unordered_set<int> my_ds_indices;
  for (auto t : m_all_minibatch_indices[m_rank]) {
    my_ds_indices.insert(t);
  }

  std::vector<std::unordered_set<int>> proc_to_indices(m_np);
  for (size_t j=0; j<m_all_minibatch_indices.size(); j++) {
    for (auto idx : m_all_minibatch_indices[j]) {
      int index = (*m_shuffled_indices)[idx];
      // P_j needs the sample that corresponds to 'index' in order
      // to complete the next epoch
      if (my_ds_indices.find(index) != my_ds_indices.end()) {
        proc_to_indices[j].insert(index);
      }
    }
  }

  debug << "exchange_data; built map\n";

debug << "exchange_data: Time to build map: " << get_time() -  tma << "\n";

  //========================================================================
  //part 1: exchange the sizes of the data
  // m_send_buffer[j] is a conduit::Node that contains
  // all samples that this proc will send to P_j

tma = get_time();

  for (int p=0; p<m_np; p++) {

double tmy = get_time();

    m_send_buffer[p].reset();
    for (auto idx : proc_to_indices[p]) {
      m_send_buffer[p][std::to_string(idx)] = m_data[idx];
    }

debug << "\nassemble send_buffer -> P_" << p <<"; num samples: " << proc_to_indices[p].size() << " Time: " << get_time() -  tmy << "\n";
tmy = get_time();

    // code in the following method is a modification of code from
    // conduit/src/libs/relay/conduit_relay_mpi.cpp
    build_node_for_sending(m_send_buffer[p], m_send_buffer_2[p]);

debug << "  build_node for sending; Time: " << get_time() -  tmy << "\n";
tmy = get_time();

    m_outgoing_msg_sizes[p] = m_send_buffer_2[p].total_bytes_compact();
    MPI_Isend((void*)&m_outgoing_msg_sizes[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &m_send_requests[p]);

debug << "  start Isend Time: " << get_time() -  tmy << "\n";
  }
 double tmy = get_time();

  //start receives for sizes of the data
  for (int p=0; p<m_np; p++) {
    MPI_Irecv((void*)&m_incoming_msg_sizes[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &m_recv_requests[p]);

  }

debug << "\nTime to start Irecvs: " << get_time() -  tmy << "\n";
  
double tmz = get_time();

  // wait for all msgs to complete
  MPI_Waitall(m_np, m_send_requests.data(), m_status.data());
  MPI_Waitall(m_np, m_recv_requests.data(), m_status.data());

debug << "Time for waitalls" << get_time() -  tmz << "\n";

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
tma = get_time();

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

double tmw = get_time();

  conduit::Node nd;
  m_minibatch_data.clear();
  for (int p=0; p<m_np; p++) {
double tmx = get_time();
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
    nd.reset();
    nd.update(n_msg["data"]);
    const std::vector<std::string> &names = nd.child_names();
debug << "exchange_data: Time to unpack data from P_"<<p<<" Time: " << get_time() - tmx << "\n";
tmx = get_time();
    for (auto t : names) {
      conduit::Node n3 = nd[t];
      m_minibatch_data[atoi(t.c_str())] = n3;
    }
debug << "exchange_data: Time break up  samples from P_" << p  << ": " << " num samples: " << m_minibatch_data.size() << "  Time: " << get_time() -  tmx << "\n";
  }

debug << "TOTAL Time to unpack and break up all incoming data: " << get_time() - tmw << "\n";

  if (m_master) std::cout << "data_store_jag::exchange_data Time: " << get_time() - tm1 << "\n";

  debug << "TOTAL exchange_data Time: " << get_time() - tm1 << "\n";
}

void data_store_jag::set_conduit_node(int data_id, conduit::Node &node) {
  if (m_unshuffle.find(data_id) == m_unshuffle.end()) {
    LBANN_ERROR("failed to find data_id: " + std::to_string(data_id) + " in m_unshuffle");
  }
  int idx = m_unshuffle[data_id];
  if (m_data.find(idx) != m_data.end()) {
    LBANN_ERROR("duplicate data_id: " + std::to_string(idx) + " in data_store_jag::set_conduit_node");
  }
  m_data[idx] = node;
}

const conduit::Node & data_store_jag::get_conduit_node(int data_id, bool any_node) const {
  if (any_node) {
    LBANN_ERROR("data_store_jag::get_conduit_node called with any_node = true; this is not yet functional; please contact Dave Hysom");
  }

  std::unordered_map<int, conduit::Node>::const_iterator t = m_minibatch_data.find(data_id);
  if (t == m_minibatch_data.end()) {
    LBANN_ERROR("failed to find data_id: " + std::to_string(data_id) + " in m_minibatch_data; m_minibatch_data.size: " + std::to_string(m_minibatch_data.size()) + "; epoch:"  + std::to_string(m_model->get_cur_epoch()));
  }

  return t->second;
}

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


void data_store_jag::build_all_minibatch_indices() {
  m_all_minibatch_indices.clear();
  m_owner.clear();
  m_all_minibatch_indices.resize(m_np);
  for (size_t idx=0; idx<m_shuffled_indices->size(); ++idx) {
    int owner = idx % m_np;
    m_owner[idx] = owner;
    m_all_minibatch_indices[owner].push_back(idx);
  }
}

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT
