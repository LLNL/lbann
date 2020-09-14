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
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/commify.hpp"
#include <unordered_set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <unistd.h>
#include <sys/statvfs.h>
#include <cereal/types/unordered_map.hpp>
#include <cereal/archives/binary.hpp>


#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cstdlib>

namespace lbann {

data_store_conduit::data_store_conduit(
  generic_data_reader *reader) :
  m_reader(reader) {
  m_comm = m_reader->get_comm();
  if (m_comm == nullptr) {
    LBANN_ERROR("m_comm is nullptr");
  }

#ifdef LBANN_HAS_DISTCONV
  int num_io_parts = dc::get_number_of_io_partitions();
#else
  int num_io_parts = 1;
#endif // LBANN_HAS_DISTCONV

  m_world_master = m_comm->am_world_master();
  m_trainer_master = m_comm->am_trainer_master();
  m_rank_in_trainer = m_comm->get_rank_in_trainer();
  m_rank_in_world = m_comm->get_rank_in_world();
  m_partition_in_trainer = m_rank_in_trainer/num_io_parts; // needs a better name  which group you are in
  m_offset_in_partition = m_rank_in_trainer%num_io_parts;
  m_np_in_trainer = m_comm->get_procs_per_trainer();
  m_num_partitions_in_trainer = m_np_in_trainer/num_io_parts; // rename this m_num_io_groups_in_trainer

  open_informational_files();

  options *opts = options::get();

  // For use in testing
  if (opts->has_string("data_store_fail")) {
    LBANN_ERROR("data_store_conduit is throwing a fake exception; this is for use during testing");
  }

  if (opts->has_string("data_store_test_checkpoint")
      && opts->has_string("data_store_spill")) {
    LBANN_ERROR("you passed both --data_store_test_checkpoint and --data_store_spill; please use one or the other or none, but not both");
  }
  if (opts->has_string("data_store_test_checkpoint")) {
    setup_checkpoint_test();
  }
  if (opts->has_string("data_store_spill")) {
    setup_spill(opts->get_string("data_store_spill"));
  }

  set_is_local_cache(opts->get_bool("data_store_cache"));
  set_is_preloading(opts->get_bool("preload_data_store"));
  set_is_explicitly_loading(! is_preloading());

  if (is_local_cache()) {
    PROFILE("data_store_conduit is running in local_cache mode");
  } else {
    PROFILE("data_store_conduit is running in multi-message mode");
  }
  if (is_explicitly_loading()) {
    PROFILE("data_store_conduit is explicitly loading");
  } else {
    PROFILE("data_store_conduit is preloading");
  }

  check_query_flags();
}

data_store_conduit::~data_store_conduit() {
  if (m_debug) {
    m_debug->close();
  }
  if (m_profile) {
    m_profile->close();
  }
  if (m_is_local_cache && m_mem_seg) {
    int sanity = shm_unlink(m_seg_name.c_str());
    if (sanity != 0) {
      std::cout << "\nWARNING: shm_unlink failed in data_store_conduit::~data_store_conduit()\n";
    }
    sanity = munmap(reinterpret_cast<void*>(m_mem_seg), m_mem_seg_length);
    if (sanity != 0) {
      std::cout << "\nWARNING: munmap failed in data_store_conduit::~data_store_conduit()\n";
    }
  }
}

void data_store_conduit::setup_checkpoint_test() {
  std::string c = options::get()->get_string("data_store_test_checkpoint");
  if (c == "1") {
    LBANN_ERROR("--data_store_test_checkpoint=1; you probably forgot to specify the spill directory; you must specify --data_store_test_checkpoint=<string>'");
  }
  if (c == "lassen") {
     c = get_lassen_spill_dir();
  }
  m_spill_dir_base = c;
  m_test_dir = c;
  m_run_checkpoint_test = true;
}

std::string data_store_conduit::get_lassen_spill_dir() {
  char * val = std::getenv("BBPATH");
  if (val == NULL) {
    LBANN_ERROR("std::getenv(\"BBPATH\") returned NULL; unable to use burst buffer");
  }
  std::string cc(val);
  return cc + "/data_store";
}


data_store_conduit::data_store_conduit(const data_store_conduit& rhs) {
  copy_members(rhs);
}


data_store_conduit& data_store_conduit::operator=(const data_store_conduit& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  copy_members(rhs);
  return (*this);
}

void data_store_conduit::set_data_reader_ptr(generic_data_reader *reader) {
  m_reader = reader;
  m_debug = 0;
  m_profile = 0;
  open_informational_files();
}

void data_store_conduit::copy_members(const data_store_conduit& rhs) {
  m_other = rhs.m_other;
  m_is_setup = rhs.m_is_setup;
  m_preloading = rhs.m_preloading;
  m_loading_is_complete = rhs.m_loading_is_complete;
  m_explicitly_loading = rhs.m_explicitly_loading;
  m_owner_map_mb_size = rhs.m_owner_map_mb_size;
  m_compacted_sample_size = rhs.m_compacted_sample_size;
  m_is_local_cache = rhs.m_is_local_cache;
  m_node_sizes_vary = rhs.m_node_sizes_vary;
  m_have_sample_sizes = rhs.m_have_sample_sizes;
  m_comm = rhs.m_comm;
  m_world_master = rhs.m_world_master;
  m_trainer_master = rhs.m_trainer_master;
  m_rank_in_trainer = rhs.m_rank_in_trainer;
  m_rank_in_world = rhs.m_rank_in_world;
  m_partition_in_trainer = rhs.m_partition_in_trainer;
  m_offset_in_partition = rhs.m_offset_in_partition;
  m_np_in_trainer = rhs.m_np_in_trainer;
  m_num_partitions_in_trainer = rhs.m_num_partitions_in_trainer;
  m_owner = rhs.m_owner;
  m_shuffled_indices = rhs.m_shuffled_indices;
  m_sample_sizes = rhs.m_sample_sizes;
  m_mem_seg = rhs.m_mem_seg;
  m_mem_seg_length = rhs.m_mem_seg_length;
  m_seg_name = rhs.m_seg_name;
  m_image_offsets = rhs.m_image_offsets;

  // This needs to be false, to ensure a carved out validation set
  // check for sufficient samples
  m_bcast_sample_size = true;

  m_spill = rhs.m_spill;
  m_is_spilled = rhs.m_is_spilled;
  m_spill_dir_base = rhs.m_spill_dir_base;
  m_cur_spill_dir_integer = rhs.m_cur_spill_dir_integer;
  m_cur_spill_dir = rhs.m_cur_spill_dir;
  m_num_files_in_cur_spill_dir = rhs.m_num_files_in_cur_spill_dir;

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
  m_indices_to_send = rhs.m_indices_to_send;
  m_indices_to_recv = rhs.m_indices_to_recv;

  open_informational_files();
}

void data_store_conduit::setup(int mini_batch_size) {
  PROFILE("starting setup(); m_owner.size(): ", m_owner.size());
  m_owner_map_mb_size = mini_batch_size;
  m_is_setup = true;
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
}

void data_store_conduit::spill_preloaded_conduit_node(int data_id, const conduit::Node &node) {
  // note: at this point m_data[data_id] = node
  conduit::Node n3 = node;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    build_node_for_sending(node, n3);
  }
  if (!m_node_sizes_vary) {
    error_check_compacted_node(n3, data_id);
  } else {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sample_sizes[data_id] = n3.total_bytes_compact();
  }

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    spill_conduit_node(node, data_id);
    m_spilled_nodes[data_id] = m_cur_spill_dir_integer;
    m_data.erase(data_id);
  }
}

void data_store_conduit::set_preloaded_conduit_node(int data_id, const conduit::Node &node) {
  // note: at this point m_data[data_id] = node
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_data.find(data_id) == m_data.end()) {
      LBANN_ERROR("(m_data.find(data_id) == m_data.end() for id: ", data_id);
    }
  }

  // TODO: get rid of "m_my_num_indices" -dah, May 2020

  if (is_local_cache()) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_my_num_indices;
    m_data[data_id] = node;
    return;
  }

  if (m_spill) {
    ++m_my_num_indices;
    spill_preloaded_conduit_node(data_id, node);
    return;
  }

  {
    conduit::Node n2 = node;  // node == m_data[data_id]
    std::lock_guard<std::mutex> lock(m_mutex);
    build_node_for_sending(n2, m_data[data_id]);
  }
  if (!m_node_sizes_vary) {
    error_check_compacted_node(m_data[data_id], data_id);
  } else {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sample_sizes[data_id] = m_data[data_id].total_bytes_compact();
  }
}

void data_store_conduit::error_check_compacted_node(const conduit::Node &nd, int data_id) {
  if (m_node_sizes_vary) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(m_mutex_2);
    if (m_compacted_sample_size == 0) {
      m_compacted_sample_size = nd.total_bytes_compact();
      PROFILE("num bytes for nodes to be transmitted: ", nd.total_bytes_compact(), " per node");
    } else if (m_compacted_sample_size != nd.total_bytes_compact() && !m_node_sizes_vary) {
      LBANN_ERROR("Conduit node being added data_id: ", data_id,
                  " is not the same size as existing nodes in the data_store ",
                  m_compacted_sample_size, " != ", nd.total_bytes_compact(),
                  " role: ", m_reader->get_role());
    }
  }
  if (!nd.is_contiguous()) {
    LBANN_ERROR("m_data[",  data_id, "] does not have a contiguous layout");
  }
  if (nd.data_ptr() == nullptr) {
    LBANN_ERROR("m_data[", data_id, "] does not have a valid data pointer");
  }
  if (nd.contiguous_data_ptr() == nullptr) {
    LBANN_ERROR("m_data[", data_id, "] does not have a valid contiguous data pointer");
  }
}


//n.b. Do not put any PROFILE or DEBUG_DS statements in this method,
//     since the threading from the data_reader will cause you grief
void data_store_conduit::set_conduit_node(int data_id, const conduit::Node &node, bool already_have) {

  std::lock_guard<std::mutex> lock(m_mutex);
  // TODO: test whether having multiple mutexes below is better (faster) than
  //       locking this entire call with a single mutex. For now I'm
  //       playing it safe and locking the whole dang thing.
  ++m_my_num_indices;

  if (is_local_cache() && is_preloading()) {
    LBANN_ERROR("you called data_store_conduit::set_conduit_node, but you're running in local cache mode with preloading; something is broken; please contact Dave Hysom");
  }

  {
    //std::lock_guard<std::mutex> lock(m_mutex);
    if (already_have == false && m_data.find(data_id) != m_data.end()) {
      DEBUG_DS("m_data.size: ", m_data.size(), " ERROR: duplicate data_id: ", data_id);
      LBANN_ERROR("duplicate data_id: ", data_id, " in data_store_conduit::set_conduit_node; role: ", m_reader->get_role());
    }
  }

  if (already_have && is_local_cache()) {
    if (m_data.find(data_id) == m_data.end()) {
      LBANN_ERROR("you claim the passed node was obtained from this data_store, but the data_id (", data_id, ") doesn't exist in m_data");
    }
    return;
  }

  if (is_local_cache()) {
    m_data[data_id] = node;
  }

  else {
    if (m_spill) {
  PROFILE("spill!\n");

      //TODO: rethink how we go about exchanging sample sizes.
      //currently, we exchange sample sizes a single time, and
      //the exchange is for all samples. To make this work with
      //spilling we need to compute the sample size by building
      //a node_for_sending (below), then we throw it away.
      //Also, see not in copy_members() about problems with the
      //schema that cause us to rebuild the node_for_sending after
      //copying or loading from disk. I need to revisit this and
      //figure out what's going on.
      conduit::Node n2;
      build_node_for_sending(node, n2);
      error_check_compacted_node(n2, data_id);
      {
    //    std::lock_guard<std::mutex> lock(m_mutex);
        LBANN_ERROR("NOT YET IMPLEMENTED");
        auto key = std::make_pair(data_id, m_offset_in_partition);
        m_owner[key] = m_rank_in_trainer;
        m_sample_sizes[data_id] = n2.total_bytes_compact();
        spill_conduit_node(node, data_id);
        m_spilled_nodes[data_id] = m_cur_spill_dir_integer;
      }
    }

    else {
      //      m_mutex.lock();
      DEBUG_DS("set_conduit_node : rank_in_trainer=", m_rank_in_trainer, " and partition_in_trainer=", m_partition_in_trainer, " offset in partition=", m_offset_in_partition, " with num_partitions=", m_num_partitions_in_trainer);
      auto key = std::make_pair(data_id, m_offset_in_partition);
      m_owner[key] = m_rank_in_trainer;
      build_node_for_sending(node, m_data[data_id]);
      m_sample_sizes[data_id] = m_data[data_id].total_bytes_compact();
      error_check_compacted_node(m_data[data_id], data_id);
      //      m_mutex.unlock();
    }
  }
}

const conduit::Node & data_store_conduit::get_conduit_node(int data_id) const {
  if (is_local_cache()) {
    std::unordered_map<int, conduit::Node>::const_iterator t3 = m_data.find(data_id);
    if (t3 == m_data.end()) {
      LBANN_ERROR("(local cache) failed to find data_id: ", data_id, " in m_data; m_data.size: ", m_data.size());
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
    LBANN_ERROR("failed to find data_id: ", data_id, " in m_minibatch_data; m_minibatch_data.size: ", m_minibatch_data.size(), " and also failed to find it in m_data; m_data.size: ", m_data.size(), "; role: ", m_reader->get_role());
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

  // The following is needed to deal with one-off cases where one or
  // more ranks do not own any samples (i.e, m_data is empty).
  // In this case those processors won't know the size of the compacted
  // nodes, hence, cannot properly set up their recv buffers, hence,
  // mpi throws errors.
  if (m_bcast_sample_size && !m_node_sizes_vary) {
    verify_sample_size();
    m_bcast_sample_size = false;
  }

  double tm5 = get_time();

  /// exchange sample sizes if they are non-uniform (imagenet);
  /// this will only be called once, during the first call to
  /// exchange_data_by_sample at the beginning of the 2nd epoch,
  /// or during the first call th exchange_data_by_sample() during
  /// the first epoch if preloading
  if (m_node_sizes_vary && !m_have_sample_sizes & !m_is_local_cache) {
    double tm3 = get_time();
    exchange_sample_sizes();
    m_exchange_sample_sizes_time += (get_time() - tm3);
  }

  int num_send_req = build_indices_i_will_send(current_pos, mb_size);
  if (m_spill) {
    // TODO
    load_spilled_conduit_nodes();
  }

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
        LBANN_ERROR("failed to find data_id: ", index, " to be sent to ", p, " in m_data");
      }
      const conduit::Node& n = m_data[index];
      const El::byte *s = reinterpret_cast<const El::byte*>(n.data_ptr());
      if(!n.is_contiguous()) {
        LBANN_ERROR("data_id: ", index, " does not have a contiguous layout");
      }
      if(n.data_ptr() == nullptr) {
        LBANN_ERROR("data_id: ", index, " does not have a valid data pointer");
      }
      if(n.contiguous_data_ptr() == nullptr) {
        LBANN_ERROR("data_id: ", index, " does not have a valid contiguous data pointer");
      }

      size_t sz = m_compacted_sample_size;

      if (m_node_sizes_vary) {
        if (m_sample_sizes.find(index) == m_sample_sizes.end()) {
          LBANN_ERROR("m_sample_sizes.find(index) == m_sample_sizes.end() for index: ", index, "; m_sample_sizes.size: ", m_sample_sizes.size());
        }
        sz = m_sample_sizes[index];
      }

      m_comm->nb_tagged_send<El::byte>(s, sz, p, index, m_send_requests[ss++], m_comm->get_trainer_comm());
    }
  }

  // sanity checks
  if (ss != m_send_requests.size()) {
    LBANN_ERROR("ss != m_send_requests.size; ss: ", ss, " m_send_requests.size: ", m_send_requests.size());
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
          LBANN_ERROR("m_sample_sizes.find(index) == m_sample_sizes.end() for index: ", index, "; m_sample_sizes.size(): ", m_sample_sizes.size(), " role: ", m_reader->get_role(), " for index: ", sanity, " of ", indices.size());
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
    LBANN_ERROR("ss != m_recv_buffer.size; ss: ", ss, " m_recv_buffer.size: ", m_recv_buffer.size());
  }
  if (m_recv_requests.size() != m_recv_buffer.size()) {
    LBANN_ERROR("m_recv_requests.size != m_recv_buffer.size; m_recv_requests: ", m_recv_requests.size(), " m_recv_buffer.size: ", m_recv_buffer.size());
  }

  m_start_snd_rcv_time += (get_time() - tm5);

  // wait for all msgs to complete
  tm5 = get_time();
  m_comm->wait_all(m_send_requests);
  m_comm->wait_all(m_recv_requests);
  m_comm->trainer_barrier();
  m_wait_all_time += (get_time() - tm5);

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

  tm5 = get_time();
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
  m_rebuild_time += (get_time() - tm5);

  if (m_spill) {
    // TODO
    m_data.clear();
  }
}

int data_store_conduit::build_indices_i_will_recv(int current_pos, int mb_size) {
  m_indices_to_recv.clear();
  m_indices_to_recv.resize(m_np_in_trainer);
  int k = 0;
  for (int i=current_pos; i< current_pos + mb_size; ++i) {
    auto index = (*m_shuffled_indices)[i];
#ifdef LBANN_HAS_DISTCONV
    int num_ranks_in_partition = dc::get_number_of_io_partitions();
#else
    int num_ranks_in_partition = 1;
#endif // LBANN_HAS_DISTCONV
    if ((((i % m_owner_map_mb_size) % m_num_partitions_in_trainer) * num_ranks_in_partition + m_offset_in_partition) == m_rank_in_trainer) {
      auto key = std::make_pair(index, m_offset_in_partition);
      int owner = m_owner[key];
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
  DEBUG_DS("build_indices_i_will_send; cur pos: ", current_pos, " mb_size: ", mb_size, " m_data.size: ", m_data.size());
  for (int i = current_pos; i < current_pos + mb_size; i++) {
    auto index = (*m_shuffled_indices)[i];
    /// If this rank owns the index send it to the (i%m_np)'th rank
    bool is_mine = false;
    if (m_data.find(index) != m_data.end()) {
      is_mine = true;
    } else if (m_spilled_nodes.find(index) != m_spilled_nodes.end()) {
      is_mine = true;
    }
    if (is_mine) {
#ifdef LBANN_HAS_DISTCONV
      int num_ranks_in_partition = dc::get_number_of_io_partitions();
#else
      int num_ranks_in_partition = 1;
#endif // LBANN_HAS_DISTCONV
      m_indices_to_send[(((i % m_owner_map_mb_size) % m_num_partitions_in_trainer) * num_ranks_in_partition + m_offset_in_partition)].insert(index);

      // Sanity check
      auto key = std::make_pair(index, m_offset_in_partition);
      if (m_owner[key] != m_rank_in_trainer) {
        LBANN_ERROR( "error for i: ", i, " index: ", index, " m_owner: ", m_owner[key], " me: ", m_rank_in_trainer);
      }
      k++;
    }
  }
  return k;
}

void data_store_conduit::build_preloaded_owner_map(const std::vector<int>& per_rank_list_sizes) {
  PROFILE("starting data_store_conduit::build_preloaded_owner_map");
  m_owner.clear();
  int owning_rank = 0;
  size_t per_rank_list_range_start = 0;
  for (size_t i = 0; i < m_shuffled_indices->size(); i++) {
    const auto per_rank_list_size = per_rank_list_sizes[owning_rank];
    if(i == (per_rank_list_range_start + per_rank_list_size)) {
      ++owning_rank;
      per_rank_list_range_start += per_rank_list_size;
    }
    auto key = std::make_pair((*m_shuffled_indices)[i], m_offset_in_partition);
    m_owner[key] = owning_rank;
  }
PROFILE("build_preloaded_owner_map; m_owner_maps_were_exchanged = true");
  m_owner_maps_were_exchanged = true;
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
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_data.find(data_id) != m_data.end()) {
    LBANN_ERROR("we already have a node with data_id= ", data_id);
  }
  return m_data[data_id];
}

void data_store_conduit::compact_nodes() {
  for(auto&& j : *m_shuffled_indices) {
    if(m_data.find(j) != m_data.end()){
      if(! (m_data[j].is_contiguous() && m_data[j].is_compact()) ) {
        /// Repack the nodes because they don't seem to copy correctly
        conduit::Node node = m_data[j]["data"];
        m_data.erase(j);
        build_node_for_sending(node, m_data[j]);
      }
    }
  }
}

int data_store_conduit::get_index_owner(int idx) {
  auto key = std::make_pair(idx, m_offset_in_partition);
  if (m_owner.find(key) == m_owner.end()) {
    LBANN_ERROR(" idx: ", idx, " was not found in the m_owner map; map size: ", m_owner.size());
  }
  return m_owner[key];
}

void data_store_conduit::check_mem_capacity(lbann_comm *comm, const std::string sample_list_file, size_t stride, size_t offset) {
//TODO: this is junky, and isn't called anywhere; rethink!
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
          LBANN_ERROR("units is ", units, " but we only know how to handle kB; please contact Dave Hysom");
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
      LBANN_ERROR("failed to open ", sample_list_file, " for reading");
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
          LBANN_ERROR(" failed to open ", base_dir, '/', filename, " for reading");
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
            LBANN_ERROR("failed to read success flag for ", key);
          }
          if (useme.to_int64() == 1) {
            got_one = true;
            try {
              key = "/" + t;
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, useme);
            } catch (conduit::Error const& e) {
              LBANN_ERROR("failed to load JAG sample: ", key);
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
  return t != m_data.end();
}

void data_store_conduit::set_shuffled_indices(const std::vector<int> *indices) {
  m_shuffled_indices = indices;
}

void data_store_conduit::exchange_sample_sizes() {
  DEBUG_DS("starting data_store_conduit::exchange_sample_sizes");
  int my_count = m_sample_sizes.size();
  std::vector<int> all_counts(m_np_in_trainer);
  m_comm->all_gather(&my_count, 1, all_counts.data(), 1,  m_comm->get_trainer_comm());

  if (m_debug) {
    for (size_t h=0; h<all_counts.size(); h++) {
      DEBUG_DS("num samples owned by P_", h, " is ", all_counts[h]);
    }
  }

  std::vector<size_t> my_sizes(m_sample_sizes.size()*2);
  size_t j = 0;
  for (auto t : m_sample_sizes) {
    my_sizes[j++] = t.first;
    my_sizes[j++] = t.second;
  }

  std::vector<size_t> others;
  for (int k=0; k<m_np_in_trainer; k++) {
    DEBUG_DS("sample sizes for P_", k);
    others.resize(all_counts[k]*2);
    if (m_rank_in_trainer == k) {
      m_comm->broadcast<size_t>(k, my_sizes.data(), all_counts[k]*2,  m_comm->get_trainer_comm());
    } else {
      m_comm->broadcast<size_t>(k, others.data(), all_counts[k]*2,  m_comm->get_trainer_comm());

      for (size_t i=0; i<others.size(); i += 2) {
        if (m_sample_sizes.find(others[i]) != m_sample_sizes.end()) {
          if (m_debug) {
            DEBUG_DS("SAMPLE SIZES for P_", k);
            for (size_t h=0; h<others.size(); h += 2) {
              DEBUG_DS(others[h], " SIZE: ", others[h+1]);
            }
          }
          LBANN_ERROR("m_sample_sizes.find(others[i]) != m_sample_sizes.end() for data_id: ", others[i]);
        }
        m_sample_sizes[others[i]] = others[i+1];
      }
    }
  }

  m_have_sample_sizes = true;
}

void data_store_conduit::set_is_preloading(bool flag) {
  m_preloading = flag;
}

void data_store_conduit::set_is_explicitly_loading(bool flag) {
  m_explicitly_loading = flag;
  if (is_preloading() && is_explicitly_loading()) {
    LBANN_ERROR("flags for both explicit and pre- loading are set; this is an error");
  }
}

void data_store_conduit::set_loading_is_complete() {
  PROFILE("set_loading_is_complete()");
  m_loading_is_complete = true;
  set_is_preloading(false);
  set_is_explicitly_loading(false);
  check_query_flags();

  if (m_run_checkpoint_test) {
    test_checkpoint(m_spill_dir_base);
  }
}

bool data_store_conduit::is_fully_loaded() const {
  if (m_loading_is_complete) {
    return true;
  }
  return false;
}

void data_store_conduit::get_image_sizes(map_is_t &file_sizes, std::vector<std::vector<int>> &indices) {
  /// this block fires if image sizes have been precomputed
  if (options::get()->has_string("image_sizes_filename")) {
    LBANN_ERROR("not yet implemented");
    //TODO dah - implement, if this becomes a bottleneck (but I don't think it will)
  }

  // get list of image file names
  image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
  if (image_reader == nullptr) {
    LBANN_ERROR("data_reader_image *image_reader = dynamic_cast<data_reader_image*>(m_reader) failed");
  }
  const std::vector<image_data_reader::sample_t> &image_list = image_reader->get_image_list();
  std::vector<size_t> my_image_sizes;

  // this block fires if we're exchanging cache data at the end
  // of the first epoch, and the data store was not preloaded
  if (is_explicitly_loading()) {
    for (const auto &t : m_data) {
      int data_id = t.first;
      my_image_sizes.push_back(data_id);
      my_image_sizes.push_back(t.second[LBANN_DATA_ID_STR(data_id) + "/buffer_size"].value());
    }
  }

  else {
    // get sizes of files for which I'm responsible
    for (size_t h=m_rank_in_trainer; h<m_shuffled_indices->size(); h += m_np_in_trainer) {
      ++m_my_num_indices;
      const std::string fn = m_reader->get_file_dir() + '/' + image_list[(*m_shuffled_indices)[h]].first;
      std::ifstream in(fn.c_str());
      if (!in) {
        LBANN_ERROR("failed to open ", fn, " for reading; file_dir: ", m_reader->get_file_dir(), "  fn: ", image_list[h].first, "; role: ", m_reader->get_role());
      }
      in.seekg(0, std::ios::end);
      my_image_sizes.push_back((*m_shuffled_indices)[h]);
      my_image_sizes.push_back(in.tellg());
      in.close();
    }
  }

  // exchange image sizes
  int my_count = my_image_sizes.size();

  std::vector<int> counts(m_np_in_trainer);
  m_comm->all_gather<int>(&my_count, 1, counts.data(), 1, m_comm->get_trainer_comm());

  //my_image_sizes[h*2] contains the image index
  //my_image_sizes[h*2+1] contains the image sizee

  //fill in displacement vector for gathering the actual image sizes
  std::vector<int> disp(m_np_in_trainer + 1);
  disp[0] = 0;
  for (size_t h=0; h<counts.size(); ++h) {
    disp[h+1] = disp[h] + counts[h];
  }

  std::vector<size_t> work(image_list.size()*2);
  m_comm->trainer_all_gather<size_t>(my_image_sizes, work, counts, disp);
  indices.resize(m_np_in_trainer);
  for (int h=0; h<m_np_in_trainer; h++) {
    indices[h].reserve(counts[h]);
    size_t start = disp[h];
    size_t end = disp[h+1];
    for (size_t k=start; k<end; k+= 2) {
      size_t idx = work[k];
      size_t size = work[k+1];
      indices[h].push_back(idx);
      file_sizes[idx] = size;
    }
  }
}

void data_store_conduit::compute_image_offsets(map_is_t &sizes, std::vector<std::vector<int>> &indices) {
  size_t offset = 0;
  for (size_t p=0; p<indices.size(); p++) {
    for (auto idx : indices[p]) {
      if (sizes.find(idx) == sizes.end()) {
        LBANN_ERROR("sizes.find(idx) == sizes.end() for idx: ", idx);
      }
      size_t sz = sizes[idx];
      m_image_offsets[idx] = offset;
      offset += sz;
    }
  }
}

void data_store_conduit::allocate_shared_segment(map_is_t &sizes, std::vector<std::vector<int>> &indices) {
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
  PROFILE(
    "  Shared Memory segment statistics:\n",
    "   size of required shared memory segment: ", utils::commify(m_mem_seg_length), "\n",
    "   available mem: ", utils::commify(avail_mem), "\n",
    "   required size is ", percent, " percent of available");

  if (m_mem_seg_length >= avail_mem) {
    LBANN_ERROR("insufficient available memory:\n", msg.str());
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

  int shm_fd = -1;

  if (node_id == 0) {
    shm_fd = shm_open(m_seg_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
    if (shm_fd == -1) {
      LBANN_ERROR("shm_open failed");
    }
    int v = ftruncate(shm_fd, size);
    if (v != 0) {
      LBANN_ERROR("ftruncate failed for size: ", size);
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
      LBANN_ERROR("shm_open failed for filename: ", m_seg_name);
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
      LBANN_ERROR("b.st_size= ", b.st_size, " should be equal to ", size);
    }
  }
  close(shm_fd);
}

void data_store_conduit::preload_local_cache() {
  exchange_local_caches();
}

void data_store_conduit::exchange_local_caches() {
  PROFILE("Starting exchange_local_caches");
  PROFILE("  is_explicitly_loading(): ", is_explicitly_loading());
  PROFILE("  is_preloading(): ", is_preloading());
  PROFILE("  is_local_cache(): ", is_local_cache());
  PROFILE("  is_fully_loaded: ", is_fully_loaded());

  // indices[j] will contain the indices
  // that P_j will read from disk, and subsequently bcast to all others
  std::vector<std::vector<int>> indices;

  double tm1 = get_time();
  get_image_sizes(m_sample_sizes, indices);
  PROFILE("  get_image_sizes time: ", (get_time()-tm1));

  tm1 = get_time();
  allocate_shared_segment(m_sample_sizes, indices);
  PROFILE("  allocate_shared_segment time: ", (get_time()-tm1));

  std::vector<char> work;
  if (! is_explicitly_loading()) {
    tm1 = get_time();
    read_files(work, m_sample_sizes, indices[m_rank_in_trainer]);
    PROFILE("  read_files time: ", (get_time()- tm1));
  }

  tm1 = get_time();
  compute_image_offsets(m_sample_sizes, indices);
  PROFILE("  compute_image_offsets time: ", (get_time()-tm1));

  tm1 = get_time();
  exchange_images(work, m_sample_sizes, indices);
  PROFILE("  exchange_images time: ", (get_time()-tm1));

  tm1 = get_time();
  build_conduit_nodes(m_sample_sizes);
  PROFILE("  build_conduit_nodes time: ", (get_time()-tm1));

  set_loading_is_complete();

  if (options::get()->get_bool("data_store_test_cache")) {
    test_local_cache_imagenet(20);
  }
}

void data_store_conduit::read_files(std::vector<char> &work, map_is_t &sizes, std::vector<int> &indices) {

  //reserve space for reading this proc's files into a contiguous memory space
  size_t n = 0;
  for (size_t j=0; j<indices.size(); ++j) {
    n += sizes[indices[j]];
  }
  work.resize(n);

  //get the list of images from the data reader
  image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<image_data_reader::sample_t> &image_list = image_reader->get_image_list();

  //read the images
  size_t offset = 0;
  PROFILE("  my num files: ", indices.size());
  for (size_t j=0; j<indices.size(); ++j) {
    int idx = indices[j];
    size_t s = sizes[idx];
    const std::string fn = m_reader->get_file_dir() + '/' + image_list[idx].first;
    std::ifstream in(fn, std::ios::in | std::ios::binary);
    in.read(work.data()+offset, s);
    in.close();
    offset += s;
  }
}

void data_store_conduit::build_conduit_nodes(map_is_t &sizes) {
  image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<image_data_reader::sample_t> &image_list = image_reader->get_image_list();
  for (auto t : sizes) {
    int data_id = t.first;
    int label = image_list[data_id].second;
    if (m_image_offsets.find(data_id) == m_image_offsets.end()) {
      LBANN_ERROR("m_image_offsets.find(data_id) == m_image_offsets.end() for data_id: ", data_id);
    }
    size_t offset = m_image_offsets[data_id];
    if (sizes.find(data_id) == sizes.end()) {
      LBANN_ERROR("sizes.find(data_id) == sizes.end() for data_id: ", data_id);
    }
    size_t sz = sizes[data_id];
    conduit::Node &node = m_data[data_id];
    node[LBANN_DATA_ID_STR(data_id) + "/label"].set(label);
    node[LBANN_DATA_ID_STR(data_id) + "/buffer_size"] = sz;
    char *c = m_mem_seg + offset;
    node[LBANN_DATA_ID_STR(data_id) + "/buffer"].set_external_char_ptr(c, sz);
  }
}

void data_store_conduit::fillin_shared_images(char* images, size_t size, size_t offset) {
  PROFILE("  fillin_shared_images; size: ", utils::commify(size), " offset: ", utils::commify(offset));
  memcpy(reinterpret_cast<void*>(m_mem_seg+offset), reinterpret_cast<const void*>(images), size);
}

void data_store_conduit::exchange_images(std::vector<char> &work, map_is_t &image_sizes, std::vector<std::vector<int>> &indices) {

  // If explicitly loading we need to build "work" (the vector to be broadcast);
  // if preloading, this has already been built in read_files()
  if (is_explicitly_loading()) {
    if (work.size() != 0) {
      LBANN_ERROR("work.size() != 0, but it should be");
    }

    // Compute the required buffer size
    size_t n = 0;
    for (const auto &t : m_data) {
      int data_id = t.first;
      size_t sz = t.second[LBANN_DATA_ID_STR(data_id) + "/buffer_size"].value();
      n += sz;
    }
    work.resize(n);
    PROFILE("  size required for my work buffer: ", work.size());

    // Copy the images into the work vector
    size_t offset2 = 0;
    for (const auto &t : m_data) {
      int data_id = t.first;
      const conduit::Node &node = t.second;
      const char *buf = node[LBANN_DATA_ID_STR(data_id) + "/buffer"].value();
      size_t sz = node[LBANN_DATA_ID_STR(data_id) + "/buffer_size"].value();
      memcpy(work.data()+offset2, reinterpret_cast<const void*>(buf), sz);
      offset2 += sz;
      if (offset2 > work.size()) {
        LBANN_ERROR("offset >= work.size(); offset: ", offset2, " work.size(): ", work.size(), " sz: ", sz);
      }
    }
  }

  int node_rank = m_comm->get_rank_in_node();
  std::vector<char> work2;
  size_t offset = 0;
  for (int p=0; p<m_np_in_trainer; p++) {
    // Count the number of bytes to be broadcast by P_p
    size_t bytes = 0;
    for (auto idx : indices[p]) {
      bytes += image_sizes[idx];
    }
    //PROFILE("  \nP_", p, " has ", utils::commify(bytes), " bytes to bcast");

    // Set up the rounds; due to MPI yuckiness, can bcast at most INT_MAX bytes
    // in a single broadcast
    std::vector<int> rounds;
    int n = bytes/INT_MAX;
    if (n < 0) {
      LBANN_ERROR("(n < 0; that shouldn't be possible; please contact Dave Hysom");
    }
    for (int k=0; k<n; k++) {
      rounds.push_back(INT_MAX);
    }
    int remainder = bytes - (n*INT_MAX);
    rounds.push_back(remainder);

    /*
    PROFILE("  rounds: ");
    for (auto t : rounds) {
      PROFILE("    ", t);
    }
    */

    // Broadcast the rounds of data
    int work_vector_offset = 0;
    for (size_t i=0; i<rounds.size(); i++) {
      int sz = rounds[i];
      //PROFILE("  bcasting ", utils::commify(sz), " bytes");
      if (m_rank_in_trainer == p) {
        m_comm->trainer_broadcast<char>(p, work.data()+work_vector_offset, sz);
        if (node_rank == 0) {
          fillin_shared_images(work.data()+work_vector_offset, sz, offset);
        }
      } else {
        work2.resize(sz);
        m_comm->trainer_broadcast<char>(p, work2.data(), sz);
        if (node_rank == 0) {
          fillin_shared_images(work2.data(), sz, offset);
        }
      }
      work_vector_offset += sz;
      offset += sz;
    }
  }
  m_comm->barrier(m_comm->get_node_comm());
}

void data_store_conduit::exchange_owner_maps() {
  PROFILE("starting exchange_owner_maps;",
          "my owner map size: ", m_owner.size());
  DEBUG_DS("starting exchange_owner_maps;",
        "size: ", m_owner.size());

  int my_count = m_my_num_indices;
  std::vector<int> all_counts(m_np_in_trainer);
  m_comm->all_gather(&my_count, 1, all_counts.data(), 1,  m_comm->get_trainer_comm());

  std::vector<size_t> my_sizes(m_my_num_indices);
  std::vector<std::pair<size_t,size_t>> nodes_i_own(m_owner.size());
  size_t j = 0;
  for (auto t : m_owner) {
    auto slab_id = std::make_pair(t.first.first, t.first.second);
    nodes_i_own[j++] = slab_id;
    DEBUG_DS("I am building the size vector from the owner map for ", t.first.first, ".", t.first.second, " and ", t.second);
  }

  std::vector<std::pair<size_t,size_t>> other_ranks_nodes;
  for (int k=0; k<m_np_in_trainer; k++) {
    other_ranks_nodes.resize(all_counts[k]);
    if (m_rank_in_trainer == k) {
      m_comm->broadcast<std::pair<size_t,size_t>>(k, nodes_i_own.data(), all_counts[k],  m_comm->get_trainer_comm());
      if(m_debug) {
        int c = 0;
        for(auto i : nodes_i_own) {
          DEBUG_DS("k=", k,  ": nodes_i_own[", c, "]=", i.first, ".", i.second);
          c++;
        }
      }
    } else {
      m_comm->broadcast<std::pair<size_t,size_t>>(k, other_ranks_nodes.data(), all_counts[k],  m_comm->get_trainer_comm());
      if(m_debug) {
        int c = 0;
        for(auto i : other_ranks_nodes) {
          DEBUG_DS("k=", k,  ": other_ranks_nodes[", c, "]=", i.first, ".", i.second);
          c++;
        }
      }
      for (size_t i=0; i<other_ranks_nodes.size(); ++i) {
        auto key = other_ranks_nodes[i];
        // Check to make sure that I don't own this
        if (m_owner.find(key) != m_owner.end()) {

          if (m_debug) {
            auto slab_id = other_ranks_nodes[i];
            DEBUG_DS("data_store_conduit::exchange_owner_maps, duplicate data_id: ", slab_id.first, ".", slab_id.second, "; k= ", k, "\nm_owner:\n");
            for (auto t : m_owner) DEBUG_DS("data_id: ", t.first.first, " / ", t.first.second, " owner: ", t.second);
            DEBUG_DS("\nother_ranks_nodes[k]: ");
            for (auto t : other_ranks_nodes) DEBUG_DS(t.first, ".", t.second, " ");
          }

          LBANN_ERROR("duplicate data_id: ", other_ranks_nodes[i].first, ".",
                      other_ranks_nodes[i].second, " role: ", m_reader->get_role(), "; m_owner[",other_ranks_nodes[i].first, ".", other_ranks_nodes[i].second,"] = ", m_owner[key]);
        }
        m_owner[key] = k;
      }
    }

  }
  PROFILE("leaving data_store_conduit::exchange_owner_maps\n",
          "my owner map size: ", m_owner.size());
  m_owner_maps_were_exchanged = true;
PROFILE("exchange_owner_maps; m_owner_maps_were_exchanged = true");
  set_loading_is_complete();

  PROFILE("LEAVING exchange_owner_maps;",
          "my owner map size: ", m_owner.size());
}

void data_store_conduit::profile_timing() {
  if (m_exchange_time == 0) {
    return;
  }
  if (m_exchange_time > 0.) {
    PROFILE(
        "\n",
        "Exchange Data Timing:\n",
        "  exchange_mini_batch_data: ", m_exchange_time, "\n",
        "  exchange sample sizes:    ", m_exchange_sample_sizes_time, "\n",
        "  start sends and rcvs:     ", m_start_snd_rcv_time, "\n",
        "  wait alls:                ", m_wait_all_time, "\n",
        "  unpacking rcvd nodes:     ", m_rebuild_time, "\n\n");

    if (options::get()->get_bool("data_store_min_max_timing")) {
      std::vector<double> send;
      static int count = 5;
      send.reserve(count);
      send.push_back(m_exchange_time);
      send.push_back(m_exchange_sample_sizes_time);
      send.push_back(m_start_snd_rcv_time);
      send.push_back(m_wait_all_time);
      send.push_back(m_rebuild_time);
      if (m_trainer_master) {
        std::vector<double> rcv_max(count);
        std::vector<double> rcv_min(count);
        m_comm->trainer_reduce<double>(send.data(), count, rcv_max.data(), El::mpi::MAX);
        m_comm->trainer_reduce<double>(send.data(), count, rcv_min.data(), El::mpi::MIN);
        PROFILE(
          "Exchange Data MAX Timing:\n",
          "  exchange_mini_batch_data: ", rcv_max[0], "\n",
          "  exchange sample sizes:    ", rcv_max[1], "\n",
          "  start sends and rcvs:     ", rcv_max[2], "\n",
          "  wait alls:                ", rcv_max[3], "\n",
          "  unpacking rcvd nodes:     ", rcv_max[4], "\n\n");
        PROFILE(
          "Exchange Data MIN Timing:\n",
          "  exchange_mini_batch_data: ", rcv_min[0], "\n",
          "  exchange sample sizes:    ", rcv_min[1], "\n",
          "  start sends and rcvs:     ", rcv_min[2], "\n",
          "  wait alls:                ", rcv_min[3], "\n",
          "  unpacking rcvd nodes:     ", rcv_min[4], "\n\n");
      } else {
        m_comm->trainer_reduce<double>(send.data(), count, 0, El::mpi::MAX);
        m_comm->trainer_reduce<double>(send.data(), count, 0, El::mpi::MIN);
      }
    }

    m_exchange_sample_sizes_time = 0.;
    m_start_snd_rcv_time = 0.;
    m_wait_all_time = 0.;
    m_rebuild_time = 0.;
    m_exchange_time = 0.;
  }
}

void data_store_conduit::exchange_mini_batch_data(size_t current_pos, size_t mb_size) {
  if (is_local_cache() && is_fully_loaded()) {
    return;
  }

  if (m_reader->at_new_epoch() && is_local_cache() && is_explicitly_loading()) {
    exchange_local_caches();
    return;
  }

  if (m_reader->at_new_epoch()) {
    PROFILE("\nExchange_mini_batch_data");
    PROFILE("  is_explicitly_loading(): ", is_explicitly_loading());
    PROFILE("  is_local_cache(): ", is_local_cache());
    PROFILE("  is_fully_loaded: ", is_fully_loaded());
    if (! is_local_cache()) {
      profile_timing();
    }
  }

  double tm1 = get_time();

  // when not running in preload mode, exchange owner maps after the 1st epoch
  if (m_reader->at_new_epoch() && ! is_preloading() && !is_local_cache()) {
    PROFILE("calling exchange_owner_maps");
    if (!m_owner_maps_were_exchanged) {
      exchange_owner_maps();
    }

    else {
      PROFILE("  owner_maps were already exchanged; returning");
    }
    m_owner_maps_were_exchanged = true;
PROFILE("exchange_mini_batch_data; m_owner_maps_were_exchanged = true");
    /*
     * TODO
    if (m_spill) {
      m_is_spilled = true;
      m_metadata.close();
      save_state();
    }
    */
  }

  exchange_data_by_sample(current_pos, mb_size);
  m_exchange_time += (get_time() - tm1);
}

void data_store_conduit::flush_debug_file() {
  if (!m_debug) {
    return;
  }
  m_debug->close();
  m_debug->open(m_debug_filename.c_str(), std::ios::app);
}

void data_store_conduit::flush_profile_file() const {
  if (!m_profile) {
    return;
  }
  m_profile->close();
  m_profile->open(m_profile_filename.c_str(), std::ios::app);
}

size_t data_store_conduit::get_num_global_indices() const {
  size_t n = m_comm->trainer_allreduce<size_t>(m_data.size());
  return n;
}

void data_store_conduit::test_checkpoint(const std::string &checkpoint_dir) {
  if (m_world_master) {
    std::cout << "starting data_store_conduit::test_checkpoint for role: "
              << m_reader->get_role() << std::endl;
    print_partial_owner_map(10);
    std::cout << "\nHere are some private variables before clearing them:\n";
    print_variables();
    std::cout << "\nCalling write_checkpoint()" << std::endl;
  }
  write_checkpoint(checkpoint_dir);

  // clear or reset private variables
  auto sanity = m_owner;
  m_owner.clear();
  m_sample_sizes.clear();
  m_data.clear();

  m_is_setup = false;
  m_preloading = false;
  m_explicitly_loading = true;
  m_owner_map_mb_size = 0;
  m_compacted_sample_size = 0;
  m_node_sizes_vary = true;

  if (m_world_master) {
    std::cout << "\nHere are some private variables after clearing them:\n";
    print_variables();
  }

  if (m_world_master) {
    std::cout << "Cleared the owner map; m_owner.size(): " << m_owner.size()
              << std::endl
              << "Calling load_checkpoint" << std::endl;
  }
  load_checkpoint(checkpoint_dir, nullptr);
  if (m_world_master) {
    std::cout << "Here is part of the re-loaded owner map; map.size(): " << m_owner.size() << std::endl;
    print_partial_owner_map(10);
    std::cout << "\nHere are some private variables after reloading:\n";
    print_variables();
  }

  //check that the owner map was correctly loaded
  for (auto t : m_owner) {
    if (sanity.find(t.first) == sanity.end()) {
      LBANN_ERROR("sanity.find(t.first) == sanity.end() for t.first= ", t.first.first, ":", t.first.second);
    } else if (sanity[t.first] != m_owner[t.first]) {
      LBANN_ERROR("sanity[t.first] != m_owner[t.first] for t.first= ", t.first.first, ":", t.first.second, " and m_owner[t.first]= ", m_owner[t.first]);
    }
  }

  m_comm->global_barrier();
}

void data_store_conduit::make_dir_if_it_doesnt_exist(const std::string &dir_name) {
  int node_rank = m_comm->get_rank_in_node();
  if (node_rank == 0) {
    bool exists = file::directory_exists(dir_name);
    if (!exists) {
      PROFILE("data_store_conduit; the directory '", dir_name, "' doesn't exist; creating it");
      file::make_directory(dir_name);
    }
  }
}

void data_store_conduit::setup_spill(std::string base_dir) {
  if (base_dir == "lassen") {
     base_dir = get_lassen_spill_dir();
  }
  m_spill_dir_base = base_dir;
  m_spill = true;
  m_cur_spill_dir_integer = -1;
  m_num_files_in_cur_spill_dir = m_max_files_per_directory;
  PROFILE("base directory for spilling: ", m_spill_dir_base);

  // create directory structure for spilling data
  make_dir_if_it_doesnt_exist(m_spill_dir_base);
  m_comm->trainer_barrier();
  make_dir_if_it_doesnt_exist(get_conduit_dir());
  PROFILE("base directory for spilling conduit nodes: ", get_conduit_dir());

  // open metadata file; this will contains the file pathnames of spilled
  // conduit nodes
  const std::string fnn = get_metadata_fn();
  m_metadata.open(fnn.c_str());
  if (!m_metadata) {
    LBANN_ERROR("failed to open ", fnn, " for writing");
  }
  PROFILE("will write metadata to file: ", get_metadata_fn());

  //n.b. must do this here, instead of only in spill_conduit_node(),
  //     in case a reader (e.g, validation reader) has no data
  open_next_conduit_spill_directory();
}

void data_store_conduit::write_checkpoint(std::string dir_name) {
  // if we're spilling data, everything has already been written to file
  if (m_is_spilled) {
    return;
  }
  double tm1 = get_time();
  setup_spill(dir_name);

  // cerealize all non-conduit::Node variables
  save_state();

  // save conduit Nodes
  m_metadata << get_conduit_dir() << "\n";
  DEBUG_DS("m_data.size: ", m_data.size());
  for (auto t : m_data) {
    spill_conduit_node(t.second["data"], t.first);
  }
  m_metadata.close();
  PROFILE("time to write checkpoint: ", (get_time() - tm1));
}

void data_store_conduit::save_state() {
  // checkpoint remaining state using cereal
  const std::string fn = get_cereal_fn();
  std::ofstream os(fn);
  if (!os) {
    LBANN_ERROR("failed to open ", fn, " for writing");
  }

  {
  cereal::XMLOutputArchive archive(os);
    archive(CEREAL_NVP(m_my_num_indices),
            CEREAL_NVP(m_owner_maps_were_exchanged),
            CEREAL_NVP(m_is_setup),
            CEREAL_NVP(m_preloading),
            CEREAL_NVP(m_loading_is_complete),
            CEREAL_NVP(m_explicitly_loading),
            CEREAL_NVP(m_owner_map_mb_size),
            CEREAL_NVP(m_compacted_sample_size),
            CEREAL_NVP(m_is_local_cache),
            CEREAL_NVP(m_node_sizes_vary),
            CEREAL_NVP(m_have_sample_sizes),
            CEREAL_NVP(m_owner),
            CEREAL_NVP(m_sample_sizes));
  }
  os.close();
}

void data_store_conduit::load_checkpoint(std::string dir_name, generic_data_reader *reader) {
  double tm1 = get_time();
  PROFILE("starting data_store_conduit::load_checkpoint");

  // Sanity check that checkpoint directories exist
  m_spill_dir_base = dir_name;
  bool exists = file::directory_exists(m_spill_dir_base);
  if (!exists) {
    LBANN_ERROR("cannot load data_store from file, since the specified directory ", dir_name, "doesn't exist");
  }
  const std::string conduit_dir = get_conduit_dir();
  exists = file::directory_exists(conduit_dir);
  if (!exists) {
    LBANN_ERROR("cannot load data_store from file, since the specified directory '", conduit_dir, "' doesn't exist");
  }

  // Read checkpoint for all essential variables except conduit Nodes
  const std::string fn = get_cereal_fn();
  std::ifstream in(fn);
  if (!in) {
    LBANN_ERROR("failed to open ", m_cereal_fn, " for reading");
  }
  cereal::XMLInputArchive iarchive(in);
  iarchive(CEREAL_NVP(m_my_num_indices),
           m_owner_maps_were_exchanged, m_is_setup,
           m_preloading, m_loading_is_complete,
           m_explicitly_loading, m_owner_map_mb_size,
           m_compacted_sample_size, m_is_local_cache,
           m_node_sizes_vary, m_have_sample_sizes,
           m_owner, m_sample_sizes);

  if (reader != nullptr) {
#ifdef LBANN_HAS_DISTCONV
    int num_io_parts = dc::get_number_of_io_partitions();
#else
    int num_io_parts = 1;
#endif // LBANN_HAS_DISTCONV

    m_reader = reader;
    m_comm = m_reader->get_comm();
    m_shuffled_indices = &(m_reader->get_shuffled_indices());
    m_world_master = m_comm->am_world_master();
    m_trainer_master = m_comm->am_trainer_master();
    m_rank_in_trainer = m_comm->get_rank_in_trainer();
    m_rank_in_world = m_comm->get_rank_in_world();
    m_partition_in_trainer = m_rank_in_trainer/num_io_parts; // needs a better name  which group you are in
    m_offset_in_partition = m_rank_in_trainer%num_io_parts;
    m_np_in_trainer = m_comm->get_procs_per_trainer();
    m_num_partitions_in_trainer = m_np_in_trainer/num_io_parts; // rename this m_num_io_groups_in_trainer
  }

  // Open metadata filename; this is in index re, checkpointed conduit filenames
  const std::string metadata_fn = get_metadata_fn();
  std::ifstream metadata(metadata_fn);
  if (!metadata) {
    LBANN_ERROR("failed to open ", metadata_fn, " for reading");
  }

  // Error check that the conduit base directory name is correct
  std::string base_dir;
  getline(metadata, base_dir);
  if (conduit_dir != base_dir) {
    LBANN_ERROR("conduit_dir != base_dir (", conduit_dir, ", ", base_dir);
  }

  // Load conduit Nodes
  std::string tmp;
  int sample_id;
  while (metadata >> tmp >> sample_id) {
    if (tmp.size() > 2) {
      const std::string fn2 = base_dir + "/" + tmp;
      conduit::Node nd;
      nd.load(fn2);
      build_node_for_sending(nd, m_data[sample_id]);
    }
  }
  metadata.close();

  m_was_loaded_from_file = true;
  PROFILE("time to load checkpoint: ", (get_time() - tm1));
}

void data_store_conduit::print_variables() {
  if (!m_world_master) {
    return;
  }
  std::cout << "m_is_setup: " << m_is_setup << std::endl
            << "m_preloading: " << m_preloading << std::endl
            << "m_explicitly_loading: " << m_explicitly_loading << std::endl
            << "m_owner_map_mb_size: " << m_owner_map_mb_size << std::endl
            << "m_compacted_sample_size: " << m_compacted_sample_size << std::endl
            << "m_node_sizes_vary: " << m_node_sizes_vary << std::endl;
}

std::string data_store_conduit::get_conduit_dir() const {
  return m_spill_dir_base + "/conduit_" + m_reader->get_role() + "_" + std::to_string(m_rank_in_world);
}

std::string data_store_conduit::get_cereal_fn() const {
  return m_spill_dir_base + '/' + m_cereal_fn + "_" + m_reader->get_role() + "_" + std::to_string(m_rank_in_world) + ".xml";
}

std::string data_store_conduit::get_metadata_fn() const {
  return m_spill_dir_base + "/metadata_" + m_reader->get_role() + "_" + std::to_string(m_rank_in_world);
}

void data_store_conduit::open_next_conduit_spill_directory() {
  if (m_num_files_in_cur_spill_dir != m_max_files_per_directory) {
    return;
  }
  m_num_files_in_cur_spill_dir = 0;
  m_cur_spill_dir_integer += 1;
  m_cur_spill_dir = get_conduit_dir() + "/" + to_string(m_cur_spill_dir_integer);
  DEBUG_DS("calling file::directory_exists(", m_cur_spill_dir, ")");
  bool exists = file::directory_exists(m_cur_spill_dir);
  DEBUG_DS("exists? ", exists);
  if (!exists) {
    file::make_directory(m_cur_spill_dir);
  }
}

void data_store_conduit::spill_conduit_node(const conduit::Node &node, int data_id) {
  if (!m_metadata.is_open()) {
    LBANN_ERROR("metadata file is not open");
  }

  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_num_files_in_cur_spill_dir == m_max_files_per_directory) {
    open_next_conduit_spill_directory();
  }

  const std::string fn = m_cur_spill_dir + "/" + std::to_string(data_id);
  node.save(fn);
  m_metadata <<  m_cur_spill_dir_integer << "/" << data_id << " " << data_id << std::endl;
  m_spilled_nodes[data_id] = m_cur_spill_dir_integer;
  ++m_num_files_in_cur_spill_dir;
}

void data_store_conduit::load_spilled_conduit_nodes() {
  m_data.clear();

  for (const auto &v : m_indices_to_send) {
    for (const auto &id : v) {
      map_ii_t::const_iterator it = m_spilled_nodes.find(id);
      if (it == m_spilled_nodes.end()) {
        LBANN_ERROR("it == m_spilled_nodes.end() for sample_id: ", id, "; m_spilled_nodes.size: ", m_spilled_nodes.size());
      }
      const std::string fn = get_conduit_dir() + "/" + std::to_string(it->second) + "/" + std::to_string(id);
      //PROFILE("loading conduit file: ", fn);
      conduit::Node node;
      node.load(fn);
      build_node_for_sending(node, m_data[id]);
    }
  }
}

void data_store_conduit::open_informational_files() {
  options *opts = options::get();
  if (m_comm == nullptr) {
    LBANN_ERROR("m_comm == nullptr");
  }

  // optionally, each <rank, reader_role> pair opens a debug file
  if (opts->get_bool("data_store_debug") && !m_debug && m_reader != nullptr) {
    m_debug_filename = m_debug_filename_base + "_" + m_reader->get_role() + "." + std::to_string(m_comm->get_rank_in_world()) + ".txt";
    m_debug = new std::ofstream(m_debug_filename.c_str());
    if (!m_debug) {
      LBANN_ERROR("failed to open ", m_debug_filename, " for writing");
    }
  }

  // optionally, <P_0, reader_role> pair opens a file for writing
  if (opts->get_bool("data_store_profile") && m_world_master && !m_profile && m_reader != nullptr) {
    m_profile_filename = m_profile_filename_base + "_" + m_reader->get_role() + ".txt";
    m_profile = new std::ofstream(m_profile_filename.c_str());
    if (!m_profile) {
      LBANN_ERROR("failed to open ", m_profile_filename, " for writing");
    }
  }
}

void data_store_conduit::print_partial_owner_map(int n) {
  std::cout << "\nHere is part of the owner map; m_owner.size(): " << m_owner.size() << std::endl;
  std::map<std::pair<size_t,size_t>, int> m;
  for (auto t : m_owner) {
    m[t.first] = t.second;
  }
  int j = 0;
  for (auto t : m) {
    std::cout << "  sample_id: " << t.first.first << ":" << t.first.second << " owner: " << t.second << std::endl;
    if (j++ >= 10) break;
  }
}

void data_store_conduit::set_profile_msg(std::string s) {
  PROFILE(s);
}

void data_store_conduit::test_imagenet_node(int index, bool dereference) {
  image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
  if (image_reader == nullptr) {
    LBANN_ERROR("data_reader_image *image_reader = dynamic_cast<data_reader_image*>(m_reader) failed");
  }

  int data_id = index;
  if (dereference) {
    data_id = (*m_shuffled_indices)[index];
  }
  if (m_image_offsets.find(data_id) == m_image_offsets.end()) {
    LBANN_ERROR("m_image_offsets.find(data_id) == m_image_offsets.end()");
  }

  if (m_image_offsets.find(data_id) == m_image_offsets.end()) {
    LBANN_ERROR("m_image_offsets.find(data_id) == m_image_offsets.end() for data_id: ", data_id);
  }

  if (m_sample_sizes.find(data_id) == m_sample_sizes.end()) {
    LBANN_ERROR("failed to find data_id ", data_id, " in the image_sizes map");
  }
  size_t szz = m_sample_sizes[data_id];
  PROFILE("test_imagenet_node() for data_id: ", utils::commify(data_id), " at offset: ", utils::commify(m_image_offsets[data_id]), " image size: ", utils::commify(szz));
  if (m_image_offsets[data_id] >= INT_MAX) {
    PROFILE("    WARNING: offset is >= INT_MAX!");
  }

  std::cout << "testing sample_id: "<< utils::commify(data_id)<< " stored at offset: "<< utils::commify(m_image_offsets[data_id]);
  if (m_image_offsets[data_id] >= INT_MAX) {
    std::cout << "; (>= INT_MAX)\n";
  } else {
    std::cout << std::endl;
  }
  conduit::Node nd1;
  image_reader->load_conduit_node_from_file(data_id, nd1);
  char *buf1 = nd1[LBANN_DATA_ID_STR(data_id) + "/buffer"].value();
  size_t size1 = nd1[LBANN_DATA_ID_STR(data_id) + "/buffer_size"].value();

  const conduit::Node &nd2 = get_conduit_node(data_id);
  const char *buf2 = nd2[LBANN_DATA_ID_STR(data_id) + "/buffer"].value();
  size_t size2 = nd2[LBANN_DATA_ID_STR(data_id) + "/buffer_size"].value();

  if (size1 != size2) {
    PROFILE("buffer sizes mismatch: size of buffer read from file does not match buffer size from cache; from file: ", size1, " from cache: ", size2, " for data_id: ", data_id);



    if (m_world_master) {
      const conduit::Schema &s = nd2.schema();
      s.print();
      nd2.print();
    }



    LBANN_ERROR("buffer sizes mismatch: size of buffer read from file does not match buffer size from cache; from file: ", size1, " from cache: ", size2, " for deta_id: ", data_id);
  }
  for (size_t i=0; i<size1; i++) {
    if (buf1[i] != buf2[i]) {
      PROFILE("buffer mismatch for char #", i+1, " of ", size1, "; image buffer read from file does not match buffer from conduit node");
      LBANN_ERROR("buffer mismatch for char #", i+1, " of ", size1, "; image buffer read from file does not match buffer from conduit node");
    }
  }
  PROFILE("    PASSED!");
}


bool data_store_conduit::test_local_cache_imagenet(int n) {
  if (!m_world_master) {
    return true;
  }
  PROFILE("\nStarting data_store_conduit::test_local_cache_imagenet(", n, ")");
  if (n < 0 || n > (int)m_shuffled_indices->size()) {
    n = m_shuffled_indices->size();
  }

  // edge cases: get images with smallest and largest offsets in the cache
  size_t max_offset = 0;
  size_t min_offset = 200000000;
  size_t id_max = 0;
  size_t id_min = 0;
  for (auto t :  m_image_offsets) {
    if (t.second > max_offset) {
      id_max = t.first;
      max_offset = t.second;
    }
    if (t.second < min_offset) {
      id_min = t.first;
      min_offset = t.second;
    }
  }

  // test image with smallest offset
  test_imagenet_node(id_min, false);

  // test n randomly selected images
  for (int h=0; h<n; ++h) {
    const int index = random() % m_shuffled_indices->size();
    test_imagenet_node(index);
  }

  // test image with largest offset
  test_imagenet_node(id_max, false);

  if (m_world_master) std::cout<< "  All tests passed\n";
  PROFILE("  All tests passed\n.");
  return true;
}

void data_store_conduit::check_query_flags() const {
  if (m_explicitly_loading && m_preloading) {
    LBANN_ERROR("is_explicitly_loading() && is_preloading() are both true, but should not be");
  }
  if (m_loading_is_complete && m_explicitly_loading) {
    LBANN_ERROR("is_fully_loaded() && is_explicitly_loading() are both true, but should not be");
  }
  if (m_loading_is_complete && m_preloading) {
    LBANN_ERROR("is_fully_loaded() && is_preloading() are both true, but should not be");
  }
}

void data_store_conduit::clear_owner_map() {
    m_owner_maps_were_exchanged = false;
    m_owner.clear();
}

void data_store_conduit::verify_sample_size() {
  // Note: m_compacted_sample_size is set during calls to set_conduit_node() or
  //  set_preloaded_conduit_node(). Hence, if these are not called (i.e, the
  //  rank does not own any data), m_compacted_sample_size will be zero.
  //  This method ensures that all ranks know the sample size, whether or not
  //  they own any samples
  int max_samples = m_comm->trainer_allreduce<int>(m_compacted_sample_size, El::mpi::MAX);
  if (max_samples <= 0) {
    LBANN_ERROR("sample size, which is needed for data exchange, is invalid; should be > 0, but value is: ", max_samples, "; this indicates there is insufficient data. Role: ", m_reader->get_role());
  }
  if (m_compacted_sample_size != 0 && max_samples != m_compacted_sample_size) {
    LBANN_ERROR("m_compacted_sample_size = ", m_compacted_sample_size, " but max_samples = ", max_samples, "; values should be identical");
  }
  m_compacted_sample_size = max_samples;
}

size_t data_store_conduit::get_mem_usage() {
  size_t r = 0;
  for (const auto &t : m_data) {
    const conduit::Node &nd = t.second;
    if (!nd.is_contiguous()) {
      LBANN_ERROR("node does not have a contiguous layout");
    }
/*
    if (nd.data_ptr() == nullptr) {
      nd.print();
      sleep(1);
      LBANN_ERROR("node does not have a valid data pointer");
    }
*/
    if (nd.contiguous_data_ptr() == nullptr) {
      LBANN_ERROR("node does not have a valid contiguous data pointer");
    }
    r += nd.total_bytes_compact();
  }
  return r;
}


}  // namespace lbann
