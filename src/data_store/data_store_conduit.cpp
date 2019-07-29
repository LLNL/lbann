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

#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/data_readers/data_reader_image.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include <unordered_set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <unistd.h>
#include <sys/statvfs.h>

namespace lbann {

// Macro to throw an LBANN exception
#undef LBANN_ERROR
#define LBANN_ERROR(message)                                    \
  do {                                                          \
    std::stringstream ss_LBANN_ERROR;                           \
    ss_LBANN_ERROR << "LBANN error ";                           \
    const int rank_LBANN_ERROR = lbann::get_rank_in_world();    \
    if (rank_LBANN_ERROR >= 0) {                                \
      ss_LBANN_ERROR << "on rank " << rank_LBANN_ERROR << " ";  \
    }                                                           \
    ss_LBANN_ERROR << "(" << __FILE__ << ":" << __LINE__ << ")" \
                     << ": " << (message);                      \
    if (errno) {                                                \
      ss_LBANN_ERROR << "\nerrno: " << errno << " msg: "        \
                     << strerror(errno);                        \
    }                                                           \
    if (m_output) {                                             \
      m_output << "ERROR: " << ss_LBANN_ERROR.str()             \
               << std::endl;                                    \
      m_output.close();                                         \
    }                                                           \
    throw lbann::exception(ss_LBANN_ERROR.str());               \
  } while (0)

data_store_conduit::data_store_conduit(
  generic_data_reader *reader) :
  m_reader(reader) {

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

  if (opts->get_bool("debug")) {
    std::stringstream ss;
    ss << "debug_" << m_reader->get_role() << "." << m_comm->get_rank_in_world();
    m_output.open(ss.str().c_str());
    if (m_world_master) {
      std::cerr << "opened " << ss.str() << " for writing\n";
    }
  }

  m_is_local_cache = opts->get_bool("data_store_cache");
  m_preload = opts->get_bool("preload_data_store");
  if (m_is_local_cache && !m_preload) {
    LBANN_ERROR("data_store_cache is currently only implemented for preload mode; this will change in the future. For now, pleas pass both flags: data_store_cache and --preload_data_store");
  }

  if (m_world_master) {
    if (m_is_local_cache) {
      std::cerr << "data_store_conduit is running in local_cache mode\n";
    } else if (m_super_node) {
      std::cerr << "data_store_conduit is running in super_node mode\n";
    } else {
      std::cerr << "data_store_conduit is running in multi-message mode\n";
    }
  }
}

data_store_conduit::~data_store_conduit() {
  if (m_output) {
    m_output.close();
  }
  if (m_is_local_cache && m_mem_seg) {
    int sanity = shm_unlink(m_seg_name.c_str());
    if (sanity != 0) {
      std::cerr << "\nWARNING: shm_unlink failed in data_store_conduit::~data_store_conduit()\n";
    }
    sanity = munmap(reinterpret_cast<void*>(m_mem_seg), m_mem_seg_length);
    if (sanity != 0) {
      std::cerr << "\nWARNING: munmap failed in data_store_conduit::~data_store_conduit()\n";
    }
  }
}

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

void data_store_conduit::set_role(const std::string role) {
  if (options::get()->get_bool("debug")) {
    std::stringstream ss;
    ss << "debug_" << m_reader->get_role() << "." << m_comm->get_rank_in_world();
    m_output.open(ss.str().c_str());
  }
}

void data_store_conduit::copy_members(const data_store_conduit& rhs, const std::vector<int>& ds_sample_move_list) {
  m_n = rhs.m_n;
  m_is_setup = rhs.m_is_setup;
  m_preload = rhs.m_preload;
  m_explicit_loading = rhs.m_explicit_loading;
  m_owner_map_mb_size = rhs.m_owner_map_mb_size;
  m_super_node = rhs.m_super_node;
  m_compacted_sample_size = rhs.m_compacted_sample_size;
  m_is_local_cache = rhs.m_is_local_cache;
  m_node_sizes_vary = rhs.m_node_sizes_vary;
  m_have_sample_sizes = rhs.m_have_sample_sizes;
  m_reader = rhs.m_reader;
  m_comm = rhs.m_comm;
  m_world_master = rhs.m_world_master;
  m_trainer_master = rhs.m_trainer_master;
  m_rank_in_trainer = rhs.m_rank_in_trainer;
  m_np_in_trainer = rhs.m_np_in_trainer;
  m_owner = rhs.m_owner;
  m_shuffled_indices = rhs.m_shuffled_indices;
  m_sample_sizes = rhs.m_sample_sizes;
  m_mem_seg = rhs.m_mem_seg;
  m_mem_seg_length = rhs.m_mem_seg_length;
  m_seg_name = rhs.m_seg_name;
  m_image_offsets = rhs.m_image_offsets;

  /// This block needed when carving a validation set from the training set
  if (options::get()->get_bool("debug") && !m_output) {
    std::stringstream ss;
    ss << "debug_" << m_reader->get_role() << "." << m_comm->get_rank_in_world();
  }

  if(ds_sample_move_list.size() == 0) {
    if (m_trainer_master) {
      std::cout << "data_store_conduit::copy_members; ds_sample_move_list.size = 0; copying all entries in m_data\n";
    }
    m_data = rhs.m_data;
  } else {
    /// Move indices on the list from the data and owner maps in the RHS data store to the new data store
    if (m_trainer_master) {
      std::cout << "data_store_conduit::copy_members; ds_sample_move_list.size != 0; copying ONLY SOME entries in m_data\n";
    }
    for(auto&& i : ds_sample_move_list) {

      if(rhs.m_data.find(i) != rhs.m_data.end()){
        if (m_output) {
          rhs.m_output << "moving index: " << i << " from other to myself\n";
        }

        if (!m_super_node) {
          /// Repack the nodes because they don't seem to copy correctly
          build_node_for_sending(rhs.m_data[i]["data"], m_data[i]);
        } else {
          m_data[i] = rhs.m_data[i];
        }
        rhs.m_data.erase(i);
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
    std::cerr << "starting data_store_conduit::setup() for role: " << m_reader->get_role() << "\n";
    if (m_is_local_cache) {
      std::cerr << "data store mode: local cache\n";
    } else if (m_super_node) {
      std::cerr << "data store mode: exchange_data via super nodes\n";
    } else {
      std::cerr << "data store mode: exchange_data via individual samples\n";
    }
  }

  double tm1 = get_time();
  if (!m_preload) {
    if (m_world_master) std::cout << "calling build_owner_map\n";
    build_owner_map(mini_batch_size);
    if (m_world_master) std::cout << "  build_owner_map time: " << (get_time()-tm1) << "\n";
  } else {
    m_owner_map_mb_size = mini_batch_size;
  }

  m_is_setup = true;

  if (m_is_local_cache && m_preload) {
    preload_local_cache();
  }

  if (m_world_master && !m_preload) {
    std::cerr << "TIME for data_store_conduit setup: " << get_time() - tm1 << "\n";
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

  if (m_output) {
    m_output << "starting data_store_conduit::exchange_data_by_super_node; mb_size: " << mb_size << std::endl;
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
    if (m_output) {
      m_output << "unpacking nodes from " << p << std::endl;
    }
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
      if (m_output) {
        m_output << "next name: " << t << std::endl;
      }
      m_minibatch_data[atoi(t.c_str())][t].update_external(m_reconstituted[p][t]);
    }
  }

  if (m_output) {
    m_output << "m_minibatch_data.size(): " << m_minibatch_data.size() << "; indices: ";
    for (auto t : m_minibatch_data) {
      m_output << t.first << " ";
    }  
    m_output << std::endl;
  }  
}

void data_store_conduit::set_preloaded_conduit_node(int data_id, conduit::Node &node) {
  // note: at this point m_data[data_id] = node
  // note: if running in super_node mode, nothing to do
  // note2: this may depend on the particular data reader
  if (!m_super_node) {
    if (m_output) {
      m_output << "set_preloaded_conduit_node: " << data_id << " for non-super_node mode\n";
    }
    conduit::Node n2 = node;
    build_node_for_sending(n2, m_data[data_id]);
    if (!m_node_sizes_vary) {
      error_check_compacted_node(m_data[data_id], data_id);
    } else {
      m_sample_sizes[data_id] = m_data[data_id].total_bytes_compact();
    }
  } else {
    if (m_data.find(data_id) == m_data.end()) {
      m_data[data_id] = node;
      if (m_output) {
        m_output << "set_preloaded_conduit_node: " << data_id << " for super_node mode\n";
      }
    } else {  
      if (m_output) {
        m_output << "set_preloaded_conduit_node: " << data_id << " is already in m_data\n";
      }
    }
  }  
}

void data_store_conduit::error_check_compacted_node(const conduit::Node &nd, int data_id) {
  if (m_compacted_sample_size == 0) {
    m_compacted_sample_size = nd.total_bytes_compact();
  } else if (m_compacted_sample_size != nd.total_bytes_compact() && !m_node_sizes_vary) {
    LBANN_ERROR("Conduit node being added data_id: " + std::to_string(data_id)
                + " is not the same size as existing nodes in the data_store "
                + std::to_string(m_compacted_sample_size) + " != "
                + std::to_string(nd.total_bytes_compact())
                + " role: " + m_reader->get_role());
  }
  if (!nd.is_contiguous()) {
    LBANN_ERROR("m_data[" + std::to_string(data_id) + "] does not have a contiguous layout");
  }
  if (nd.data_ptr() == nullptr) {
    LBANN_ERROR("m_data[" + std::to_string(data_id) + "] does not have a valid data pointer");
  }
  if (nd.contiguous_data_ptr() == nullptr) {
    LBANN_ERROR("m_data[" + std::to_string(data_id) + "] does not have a valid contiguous data pointer");
  }
}


void data_store_conduit::set_conduit_node(int data_id, conduit::Node &node, bool already_have) {
  if (m_is_local_cache && m_preload) {
    LBANN_ERROR("you called data_store_conduit::set_conduit_node, but you're running in local cache mode with preloading; something is broken; please contact Dave Hysom");
  }
  m_mutex.lock();
  if (already_have == false && m_data.find(data_id) != m_data.end()) {
    LBANN_ERROR("duplicate data_id: " + std::to_string(data_id) + " in data_store_conduit::set_conduit_node");
  }

  if (m_output) {
    m_output << "set_conduit_node: " << data_id << std::endl;
  }

  if (already_have && is_local_cache()) {
    if (m_data.find(data_id) == m_data.end()) {
      LBANN_ERROR("you claim the passed node was obtained from this data_store, but the data_id (" + std::to_string(data_id) + ") doesn't exist in m_data");
    }
    m_mutex.unlock();
    return;
  }

  if (is_local_cache()) {
    m_data[data_id] = node;
    m_mutex.unlock();
  }

  else if (m_owner[data_id] != m_rank_in_trainer) {
    std::stringstream s;
    s << "set_conduit_node error for data id: "<<data_id<< " m_owner: " << m_owner[data_id] << " me: " << m_rank_in_trainer << "; data reader role: " << m_reader->get_role() << "\n";
    LBANN_ERROR(s.str());
  }

  else if (! m_super_node) {
    build_node_for_sending(node, m_data[data_id]);
    error_check_compacted_node(m_data[data_id], data_id);
    m_sample_sizes[data_id] = m_data[data_id].total_bytes_compact();
    m_mutex.unlock();
  }

  else {
    m_data[data_id] = node;
    m_mutex.unlock();
    // @TODO would like to do: m_data[data_id].set_external(node); but since
    // (as of now) 'node' is a local variable in a data_reader+jag_conduit,
    // we need to do a deep copy. If the data_store furnishes a node to the
    // data_reader during the first epoch, this copy can be avoided
  }
}

const conduit::Node & data_store_conduit::get_conduit_node(int data_id) const {
  if (m_output) {
    m_output << "get_conduit_node: " << data_id << std::endl;
  }
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
      LBANN_ERROR("(local cache) failed to find data_id: " + std::to_string(data_id) + " in m_data; m_data.size: " + std::to_string(m_data.size()));
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
    if (m_output) {
      m_output << "failed to find data_id: " << data_id << " in m_minibatch_data; my m_minibatch_data indices: ";
      for (auto t : m_minibatch_data) {
        m_output << t.first << " ";
      }  
      m_output << std::endl;
    }
  }

  return t2->second;
}

// code in the following method is a modification of code from
// conduit/src/libs/relay/conduit_relay_mpi.cpp
void data_store_conduit::build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out) {
  if (m_output) {
    m_output << "starting build_node_for_sending\n";
  }

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

  /// exchange sample sizes if they are non-uniform (imagenet);
  /// this will only be called once, during the first call to 
  /// exchange_data_by_sample at the beginning of the 2nd epoch,
  /// or during the first call th exchange_data_by_sample() during
  /// the first epoch if preloading
  if (m_node_sizes_vary && !m_have_sample_sizes) {
    exchange_sample_sizes();
  }

  if (m_output) {
    m_output << "starting data_store_conduit::exchange_data_by_sample; mb_size: " << mb_size << std::endl;
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

      int sz = m_compacted_sample_size;

      if (m_node_sizes_vary) {
        if (m_sample_sizes.find(index) == m_sample_sizes.end()) {
          LBANN_ERROR("m_sample_sizes.find(index) == m_sample_sizes.end() for index: " + std::to_string(index) + "; m_sample_sizes.size: " + std::to_string(m_sample_sizes.size()));
        }
        sz = m_sample_sizes[index];
      }

      if (m_output) {
        m_output << "sending " << index << " size: " << sz << " to " << p << std::endl;
      }

      m_comm->nb_tagged_send<El::byte>(s, sz, p, index, m_send_requests[ss++], m_comm->get_trainer_comm());
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
    int sanity = 0;
    for (auto index : indices) {
      ++sanity;
      int sz = m_compacted_sample_size;
      if (m_node_sizes_vary) {
        if (m_sample_sizes.find(index) == m_sample_sizes.end()) {
          LBANN_ERROR("m_sample_sizes.find(index) == m_sample_sizes.end() for index: " + std::to_string(index) + "; m_sample_sizes.size(): " + std::to_string(m_sample_sizes.size()) + " role: " + m_reader->get_role() + " for index: " + std::to_string(sanity) + " of " + std::to_string(indices.size()));
        }
        sz = m_sample_sizes[index];
      }

      m_recv_buffer[ss].set(conduit::DataType::uint8(sz));
      El::byte *r = reinterpret_cast<El::byte*>(m_recv_buffer[ss].data_ptr());
      m_comm->nb_tagged_recv<El::byte>(r, sz, p, index, m_recv_requests[ss], m_comm->get_trainer_comm());
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
  if (m_output) {
    m_output << "build_indices_i_will_send; cur pos: " << current_pos << " mb_size: " << mb_size << " m_data.size: " << m_data.size() << "\n";
  }
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
  if (m_world_master) std::cerr << "starting data_store_conduit::build_owner_map for role: " << m_reader->get_role() << " with mini_batch_size: " << mini_batch_size << " num indices: " << m_shuffled_indices->size() << "\n";
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
  if (m_output) {
    m_output << " starting purge_unused_samples; indices.size(): " << indices.size() << " data.size(): " << m_data.size() << std::endl;
  }
  /// Remove unused indices from the data and owner maps
  for(auto&& i : indices) {
    if(m_data.find(i) != m_data.end()){
      m_data.erase(i);
    }
    if(m_owner.find(i) != m_owner.end()) {
      m_owner.erase(i);
    }
  }
  if (m_output) {
    m_output << " leaving  purge_unused_samples; indices.size(): " << indices.size() << " data.size(): " << m_data.size() << std::endl;
  }
}

void data_store_conduit::compact_nodes() {
  if (m_super_node) {
    if (m_output) {
      m_output << "RETURNING from data_store_conduit::compact_nodes; m_data.size(): " << m_data.size() << "\n";
    }
    return;
  } else {
    if (m_output) {
      m_output << ">> NOT RETURNING from data_store_conduit::compact_nodes\n";
    }
  }
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

    std::cerr
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
      std::cerr << "\nYOU DO NOT HAVE ENOUGH MEMORY\n"
        << "==============================================================\n\n";
      LBANN_ERROR("insufficient memory to load data\n");
    } else {
      double m = 100 * mem_this_node / a_mem;
      std::cerr << "Estimate that data will consume at least " << m << " % of memory\n"
        << "==============================================================\n\n";
    }
  }

  comm->trainer_barrier();
}

bool data_store_conduit::has_conduit_node(int data_id) const {
  std::unordered_map<int, conduit::Node>::const_iterator t = m_data.find(data_id);
  if (m_output) {
    m_output << "has_conduit_node( " << data_id << " ) = " << (t == m_data.end()) << std::endl;
  }
  return t != m_data.end();
}

void data_store_conduit::set_shuffled_indices(const std::vector<int> *indices) { 
  m_shuffled_indices = indices; 
}

void data_store_conduit::exchange_sample_sizes() {
  if (m_output) {
    m_output << "starting data_store_conduit::exchange_sample_sizes" << std::endl;
  }

  int my_count = m_sample_sizes.size();
  std::vector<int> all_counts(m_np_in_trainer);
  m_comm->all_gather(&my_count, 1, all_counts.data(), 1,  m_comm->get_trainer_comm());

  if (m_output) {
    for (size_t h=0; h<all_counts.size(); h++) {
      m_output << "num samples owned by P_" << h << " is " << all_counts[h] << std::endl;
    }
  }

  std::vector<int> my_sizes(m_sample_sizes.size()*2);
  size_t j = 0;
  for (auto t : m_sample_sizes) {
    my_sizes[j++] = t.first;
    my_sizes[j++] = t.second;
  }

  std::vector<int> other_sizes;
  for (int k=0; k<m_np_in_trainer; k++) {
    other_sizes.resize(all_counts[k]*2);
    if (m_rank_in_trainer == k) {
      m_comm->broadcast<int>(k, my_sizes.data(), all_counts[k]*2,  m_comm->get_trainer_comm());
    } else {
      m_comm->broadcast<int>(k, other_sizes.data(), all_counts[k]*2,  m_comm->get_trainer_comm());
      for (size_t i=0; i<other_sizes.size(); i += 2) {
        if (m_sample_sizes.find(other_sizes[i]) != m_sample_sizes.end()) {
          LBANN_ERROR("duplicate data_id: " + std::to_string(other_sizes[i]));
        }
        m_sample_sizes[other_sizes[i]] = other_sizes[i+1];
      }
    }
  }

  m_have_sample_sizes = true;
}

void data_store_conduit::set_preload() { 
  m_preload = true;
}

void data_store_conduit::get_image_sizes(std::unordered_map<int,int> &file_sizes, std::vector<std::vector<int>> &indices) {
  /// this block fires if image sizes have been precomputed
  if (options::get()->has_string("image_sizes_filename")) {
    LBANN_ERROR("not yet implemented");
    //TODO dah - implement, if this becomes a bottleneck (but I don't think it will)
  }

  else {
    // get list of image file names
    image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
    if (image_reader == nullptr) {
      LBANN_ERROR("data_reader_image *image_reader = dynamic_cast<data_reader_image*>(m_reader) failed");
    }
    const std::vector<image_data_reader::sample_t> &image_list = image_reader->get_image_list();

    // get sizes of files for which I'm responsible
    std::vector<int> my_image_sizes;
    for (size_t h=m_rank_in_trainer; h<m_shuffled_indices->size(); h += m_np_in_trainer) {
      const std::string fn = m_reader->get_file_dir() + '/' + image_list[(*m_shuffled_indices)[h]].first;
      std::ifstream in(fn.c_str());
      if (!in) {
        LBANN_ERROR("failed to open " + fn + " for reading; file_dir: " + m_reader->get_file_dir() + "  fn: " + image_list[h].first + "; role: " + m_reader->get_role());
      }
      in.seekg(0, std::ios::end);
      my_image_sizes.push_back((*m_shuffled_indices)[h]);
      my_image_sizes.push_back(in.tellg());
      in.close();
    }
    int my_count = my_image_sizes.size();

    std::vector<int> counts(m_np_in_trainer);
    m_comm->all_gather<int>(&my_count, 1, counts.data(), 1, m_comm->get_trainer_comm());

    //counts[h*2] contains the image index
    //counts[h*2+1] contains the image sizee

    //fill in displacement vector for gathering the actual image sizes
    std::vector<int> disp(m_np_in_trainer + 1);
    disp[0] = 0;
    for (size_t h=0; h<counts.size(); ++h) {
      disp[h+1] = disp[h] + counts[h];
    }

    std::vector<int> work(image_list.size()*2);
    m_comm->trainer_all_gather<int>(my_image_sizes, work, counts, disp);
    indices.resize(m_np_in_trainer);
    for (int h=0; h<m_np_in_trainer; h++) {
      indices[h].reserve(counts[h]);
      size_t start = disp[h];
      size_t end = disp[h+1];
      for (size_t k=start; k<end; k+= 2) {
        int idx = work[k];
        int size = work[k+1];
        indices[h].push_back(idx);
        file_sizes[idx] = size;
      }
    }
  }
}

void data_store_conduit::compute_image_offsets(std::unordered_map<int,int> &sizes, std::vector<std::vector<int>> &indices) {
  size_t offset = 0;
  for (size_t p=0; p<indices.size(); p++) {
    for (auto idx : indices[p]) {
      if (sizes.find(idx) == sizes.end()) {
        LBANN_ERROR("sizes.find(idx) == sizes.end() for idx: " + std::to_string(idx));
      }
      int sz = sizes[idx];
      m_image_offsets[idx] = offset;
      offset += sz;
    }
  }
}


void data_store_conduit::allocate_shared_segment(std::unordered_map<int,int> &sizes, std::vector<std::vector<int>> &indices) {
  off_t size = 0;

  for (auto &&t : sizes) {
    size += t.second;
  }
  m_mem_seg_length = size;

  struct statvfs stat;
  int x = statvfs("/dev/shm", &stat);
  if (x != 0) {
    LBANN_ERROR("statvfs failed\n");
  }
  size_t avail_mem = stat.f_bsize*stat.f_bavail;
  double percent = 100.0 * m_mem_seg_length / avail_mem;
  std::stringstream msg;
  msg << "  size of required shared memory segment: " << m_mem_seg_length  << "\n"
      << "  available mem: " << avail_mem << "\n"
      << "  required size is " << percent << " percent of available\n";
  if (m_world_master) {
    std::cout << "\nShared memory segment statistics:\n"
              << msg.str() << "\n";
  }
  if (m_mem_seg_length >= avail_mem) {
    LBANN_ERROR("insufficient available memory:\n" + msg.str());
  }

  //need to ensure name is unique across all data readers
  m_seg_name = "/our_town_" + m_reader->get_role();

  //in case a previous run was aborted, attempt to remove the file, which
  //may or may not exist
  shm_unlink(m_seg_name.c_str());
  int node_id = m_comm->get_rank_in_node();
  if (node_id == 0) {
    std::remove(m_seg_name.c_str());
  }
  m_comm->trainer_barrier();

  int shm_fd;

  if (node_id == 0) {
    shm_fd = shm_open(m_seg_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
    if (shm_fd == -1) {
      LBANN_ERROR("shm_open failed");
    }
    int v = ftruncate(shm_fd, size);
    if (v != 0) {
      LBANN_ERROR("ftruncate failed for size: " + std::to_string(size));
    }
    void *m = mmap(0, size, PROT_WRITE | PROT_READ, MAP_SHARED, shm_fd, 0);
    if (m == MAP_FAILED) {
      LBANN_ERROR("mmap failed");
    }
    m_mem_seg = reinterpret_cast<char*>(m);
    std::fill_n(m_mem_seg, m_mem_seg_length, 1);
    int sanity = msync(static_cast<void*>(m_mem_seg), m_mem_seg_length, MS_SYNC);
    if (sanity != 0) {
      LBANN_ERROR("msync failed");
    }
  }  

  m_comm->barrier(m_comm->get_node_comm());

  if (node_id != 0) {
    shm_fd = shm_open(m_seg_name.c_str(), O_RDONLY, 0666);
    if (shm_fd == -1) {
      LBANN_ERROR("shm_open failed for filename: " + m_seg_name);
    }
    void *m = mmap(0, size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (m == MAP_FAILED) {
      LBANN_ERROR("mmap failed");
    }
    m_mem_seg = reinterpret_cast<char*>(m);

    struct stat b;
    int sanity = fstat(shm_fd, &b);
    if (sanity == -1) {
      LBANN_ERROR("fstat failed");
    }
    if (b.st_size != size) {
      LBANN_ERROR("b.st_size= " + std::to_string(b.st_size) + " should be equal to " + std::to_string(size));
    }
  }
  close(shm_fd);
}

void data_store_conduit::preload_local_cache() {
  std::unordered_map<int,int> file_sizes; 
  std::vector<std::vector<int>> indices;

  double tm1 = get_time();
  if (m_world_master) std::cout << "calling get_image_sizes" << std::endl;
  get_image_sizes(file_sizes, indices);
  if (m_world_master) std::cout << "  get_image_sizes time: " << (get_time()-tm1) << std::endl;
  tm1 = get_time();
  //indices[j] contains the indices (wrt m_reader->get_image_list())
  //that P_j will read from disk, and subsequently bcast to all others
  //
  //file_sizes maps an index to its file size
  
  if (m_world_master) std::cout << "calling allocate_shared_segment" << std::endl;
  allocate_shared_segment(file_sizes, indices);
  if (m_world_master) std::cout << "  allocate_shared_segment time: " << (get_time()-tm1) << std::endl;
  tm1 = get_time();

  if (m_world_master) std::cout << "calling read_files" << std::endl;
  std::vector<char> work;
  read_files(work, file_sizes, indices[m_rank_in_trainer]);
  if (m_world_master) std::cout << "  read_files time: " << (get_time()- tm1) << std::endl;
  tm1 = get_time();

  if (m_world_master) std::cout << "calling compute_image_offsets" << std::endl;
  compute_image_offsets(file_sizes, indices);
  if (m_world_master) std::cout << "  compute_image_offsets time: " << (get_time()-tm1) << std::endl;
  tm1 = get_time();

  if (m_world_master) std::cout << "calling exchange_images" << std::endl;
  exchange_images(work, file_sizes, indices);
  if (m_world_master) std::cout << "  exchange_images time: " << (get_time()-tm1) << std::endl;
  tm1 = get_time();

  if (m_world_master) std::cerr << "calling build_conduit_nodes" << std::endl;
  build_conduit_nodes(file_sizes);
  if (m_world_master) std::cerr << "  build_conduit_nodes time: " << (get_time()-tm1) << std::endl;
}

void data_store_conduit::read_files(std::vector<char> &work, std::unordered_map<int,int> &sizes, std::vector<int> &indices) {

  //reserve space for reading this proc's files into a contiguous memory space
  size_t n = 0;
  for (size_t j=0; j<indices.size(); ++j) {
    n += sizes[indices[j]];
  }
  work.resize(n);

  if (m_output) {
    m_output << "data_store_conduit::read_files; requested work size: " << n << std::endl;
  }

  //get the list of images from the data reader
  image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<image_data_reader::sample_t> &image_list = image_reader->get_image_list();

  //read the images
  size_t offset = 0;
  if (m_world_master) std::cerr << "  my num files: " << indices.size() << std::endl;
  for (size_t j=0; j<indices.size(); ++j) {
    int idx = indices[j];
    int s = sizes[idx];
    const std::string fn = m_reader->get_file_dir() + '/' + image_list[idx].first;
    std::ifstream in(fn, std::ios::in | std::ios::binary);
    in.read(work.data()+offset, s);
    in.close();
    offset += s;
  }
  if (m_world_master) std::cout << "  finished reading files\n";
}

void data_store_conduit::build_conduit_nodes(std::unordered_map<int,int> &sizes) {
  image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<image_data_reader::sample_t> &image_list = image_reader->get_image_list();
  for (size_t idx=0; idx<image_list.size(); idx++) {
    int label = image_list[idx].second;
    size_t offset = m_image_offsets[idx];
    size_t sz = sizes[idx];
    conduit::Node &node = m_data[idx];
    node[LBANN_DATA_ID_STR(idx) + "/label"].set(label);
    node[LBANN_DATA_ID_STR(idx) + "/buffer_size"] = sz;
    char *c = m_mem_seg + offset;
    node[LBANN_DATA_ID_STR(idx) + "/buffer"].set_external_char_ptr(c, sz);
  }
}

void data_store_conduit::fillin_shared_images(const std::vector<char> &images, size_t offset) {
  memcpy(m_mem_seg+offset, reinterpret_cast<const void*>(images.data()), images.size()); 
}

void data_store_conduit::exchange_images(std::vector<char> &work, std::unordered_map<int,int> &image_sizes, std::vector<std::vector<int>> &indices) {
  std::vector<char> work2;
  int node_rank = m_comm->get_rank_in_node();
  size_t offset = 0;
  for (int p=0; p<m_np_in_trainer; p++) {
    if (m_rank_in_trainer == p) {
      m_comm->trainer_broadcast<char>(p, work.data(), work.size());
      if (node_rank == 0) {
        fillin_shared_images(work, offset);
      }
    } else {
      size_t sz = 0;
      for (auto idx : indices[p]) {
        sz += image_sizes[idx];
      }
      work2.resize(sz);
      m_comm->trainer_broadcast<char>(p, work2.data(), sz);
      if (node_rank == 0) {
        fillin_shared_images(work2, offset);
      }
    }

    for (size_t r=0; r<indices[p].size(); r++) {
      offset += image_sizes[indices[p][r]];
    }

  }

  m_comm->barrier(m_comm->get_node_comm());
}


}  // namespace lbann

