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

// this gets called at the beginning of each epoch (except for epoch 0)
//
// Note: conduit has a very nice interface for communicating nodes
//       in non-blocking scenarios. Unf, for blocking we need to
//       handle things ourselves. TODO: possible modify conduit to
//       handle non-blocking comms
void data_store_jag::exchange_data_by_super_node() {
  double tm1 = get_time();

  if (m_n == 1) {
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

    exchange_ds_indices();
  }

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
  /// Within a trainer the shuffled indices are distributed round
  /// robin across ranks
  size_t j = 0;
  for (auto index : (*m_shuffled_indices)) {
    /// If this rank owns the index send it to the j'th rank
    if (my_ds_indices.find(index) != my_ds_indices.end()) {
      proc_to_indices[j].insert(index);
    }
    j = (j + 1) % m_np;
  }

  //========================================================================
  //part 1: exchange the sizes of the data
  // m_send_buffer[j] is a conduit::Node that contains
  // all samples that this proc will send to P_j

tma = get_time();
//double t1 = 0;
//double t2 = 0;

  for (int p=0; p<m_np; p++) {
//tmb = get_time;
    m_send_buffer[p].reset();
    for (auto idx : proc_to_indices[p]) {
      m_send_buffer[p].update_external(m_data[idx]);
    }
      //if (m_master) m_send_buffer[p].print();

    // code in the following method is a modification of code from
    // conduit/src/libs/relay/conduit_relay_mpi.cpp
    build_node_for_sending(m_send_buffer[p], m_send_buffer_2[p]);

    m_outgoing_msg_sizes[p] = m_send_buffer_2[p].total_bytes_compact();
    MPI_Isend((void*)&m_outgoing_msg_sizes[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &m_send_requests[p]);
  }

  //start receives for sizes of the data
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
tma = get_time();

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
      m_minibatch_data[atoi(t.c_str())] = m_reconstituted[p][t].parent();
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
    node["id"] = data_id;
    conduit::Node n2;
    build_node_for_sending(node, n2);
    m_data[data_id] = n2;
  }

  else {
    m_data[data_id] = node;
    /* debug block, to test if idx matches the id in the conduit node;
     * if these don't match up exceptions will be thrown in get_conduit_node
     *
    if (m_master) {
      std::cerr<<"data id:" <<data_id<< "\n";
      node.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, -1);
    */
  }
}

const conduit::Node & data_store_jag::get_conduit_node(int data_id, bool any_node) const {
  if (any_node) {
    LBANN_ERROR("data_store_jag::get_conduit_node called with any_node = true; this is not yet functional; please contact Dave Hysom");
  }

  std::unordered_map<int, const conduit::Node*>::const_iterator t = m_minibatch_data.find(data_id);
  if (t == m_minibatch_data.end()) {
    debug << "failed to find data_id: " << data_id <<  " in m_minibatch_data; m_minibatch_data.size: " << m_minibatch_data.size() << "\n";
    debug << "data IDs that we know about (these are the keys in the m_minibatch_data map): ";
    std::set<int> s3;
    for (auto t3 :  m_minibatch_data) {
      s3.insert(t3.first);
    }
    for (auto t3 : s3) debug << t3 << " ";
    debug << "\n";
    debug.close();
    debug.open(b, std::ios::app);

    LBANN_ERROR("failed to find data_id: " + std::to_string(data_id) + " in m_minibatch_data; m_minibatch_data.size: " + std::to_string(m_minibatch_data.size()) + "; epoch:"  + std::to_string(m_model->get_cur_epoch()));
  }

  return *(t->second);
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


#if 0
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
#endif

void data_store_jag::exchange_data_by_sample() {
#if 0
  double tm1 = get_time();

  debug << "\n============================================================\n"
  <<"starting exchange_data_by_sample; epoch: "<<m_model->get_cur_epoch()<< " data size: "<<m_data.size()<<"  m_n: " << m_n << "  send_buffer size: " << m_send_buffer.size() << "\n";

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

  // get indices that I need for this epoch; these correspond to
  // samples that this proc receives from others
  std::unordered_map<int, std::unordered_set<int>> needed;
  for (auto t :  my_ds_indices) {
    int index = (*m_shuffled_indices)[t];
    int owner = index % m_np;
    needed[owner].insert(index);
  }

  //debug block
  int tot = 0;
  for (auto t : needed) {
    debug << "I need " << t.second.size() << " samples from P_" << t.first << " :: ";
    for (auto tt : t.second) debug << tt << " ";
    debug << "\n";
    tot += t.second.size();
  }
  debug << "total incoming samples: " << tot << "\n";
  debug << "exchange_data: Time to build maps: " << get_time() -  tma << "\n";
  debug.close();
  debug.open(b, std::ios::app);

  int sample_size = 0;
  for (auto t : m_data) {
    sample_size = t.second.total_bytes_compact();
    break;
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

  debug << "sending " << index << " to " << p << " &m_send_requests size: " << m_send_requests.size() <<  " bytes: " << m_data[index].total_bytes_compact() << " ss: " << ss << "\n";
  debug.close();
  debug.open(b, std::ios::app);

      //const void *s = m_send_buffer[ss].data_ptr();
      const void *s = m_data[index].data_ptr();
      MPI_Isend(s, sample_size, MPI_BYTE, p, index, MPI_COMM_WORLD, &m_send_requests[ss++]);
      //MPI_Isend(s, m_outgoing_msg_sizes[p], MPI_BYTE, p, 1, MPI_COMM_WORLD, &m_send_requests[p]);

  debug << "    DONE!\n";
  debug.close();
  debug.open(b, std::ios::app);

    }
  }

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

debug << "FINISHED! starting " << indices.size() << " recvs from " << p << "\n";
debug.close();
debug.open(b, std::ios::app);

  }
  debug << "\nALL RECVS STARTED\n\n";
debug.close();
debug.open(b, std::ios::app);

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
  m_mininatch_data.clear();
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
//    int data_id = m_index_to_data_id[j];
//    m_data[data_id] = nd;
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

void data_store_jag::exchange_ds_indices() {
  std::vector<int> counts(m_np);
  int my_num_indices = m_data.size();
  m_comm->trainer_all_gather<int>(my_num_indices, counts);

  //setup data structures to exchange minibatch indices with all processors
  //displacement vector
  std::vector<int> displ(m_np);
  displ[0] = 0;
  for (size_t j=1; j<counts.size(); j++) {
    displ[j] = displ[j-1] + counts[j-1];
  }

  //recv vector
  int n = std::accumulate(counts.begin(), counts.end(), 0);
  std::vector<int> all_indices(n);

  //receive the indices
  std::vector<int> v;
  v.reserve(m_data.size());
  for (auto t : m_data) {
    v.push_back(t.first);
  }
  m_comm->all_gather<int>(v, all_indices, counts, displ, m_comm->get_trainer_comm());

  //fill in the final data structure
  m_all_minibatch_indices.clear();
  m_owner.clear();
  m_all_minibatch_indices.resize(m_np);
  for (int p=0; p<m_np; p++) {
    m_all_minibatch_indices[p].reserve(counts[p]);
    for (int i=displ[p]; i<displ[p]+counts[p]; i++) {
      m_all_minibatch_indices[p].push_back(all_indices[i]);
      m_owner[all_indices[i]] = p;
    }
  }
}

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT
