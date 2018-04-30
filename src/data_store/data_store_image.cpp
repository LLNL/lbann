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

#include "lbann/data_store/data_store_image.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/options.hpp"

namespace lbann {

data_store_image::~data_store_image() {
}

void data_store_image::setup() {

  if (m_master) std::cerr << "starting data_store_image::setup(); calling generic_data_store::setup()\n";
  generic_data_store::setup();

  set_name("data_store_image");

  //@todo needs to be designed and implemented!
  if (! m_in_memory) {
    LBANN_ERROR("not yet implemented");
  } 
  
  else {
    if (m_master) std::cerr << "data_store_image - calling get_minibatch_index_vector\n";
    get_minibatch_index_vector();

    if (m_master) std::cerr << "data_store_image - calling exchange_mb_indices\n";
    exchange_mb_indices();

    if (m_master) std::cerr << "data_store_image - calling get_my_datastore_indices\n";
    get_my_datastore_indices();

    if (m_master) std::cerr << "data_store_image - calling get_file_sizes\n";
    double tma = get_time();
    get_file_sizes();
    size_t num_bytes = get_global_num_file_bytes();
    if (m_master) std::cerr << "data_store_image - get_file_sizes time: " << get_time() - tma << " num files: " << m_file_sizes.size() << " data set size: " << ((double)num_bytes/1000000) << " MB\n";

    if (m_master) std::cerr << "data_store_image - calling report_memory_constrains\n";
    report_memory_constraints();

    if (m_master) std::cerr << "data_store_image - calling read_files\n";
    tma = get_time();
    read_files();
    if (m_master) std::cerr << "read_files time: " << get_time() - tma << "\n";

    if (m_master) std::cerr << "data_store_image - calling exchange_data\n";
    exchange_data();

    if (m_extended_testing) {
      if (m_master) std::cerr << "data_store_image - calling extended_testing\n";
      extended_testing();
    }
  }
}


void data_store_image::get_data_buf(int data_id, std::vector<unsigned char> *&buf, int multi_idx) {
  std::stringstream err;
  int index = data_id * m_num_img_srcs + multi_idx;
  if (m_my_minibatch_data.find(index) == m_my_minibatch_data.end()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find index: " << index << " m_num_img_srcs: " << m_num_img_srcs
        << " multi_idx: " << multi_idx << " in m_my_data_hash; role: "
        << m_reader->get_role();
    throw lbann_exception(err.str());
  }

  buf = &m_my_minibatch_data[index];
}

void data_store_image::load_file(const std::string &dir, const std::string &fn, unsigned char *p, size_t sz) {
  std::string imagepath;
  if (dir != "") {
    imagepath = dir + fn;
  } else {
    imagepath = fn;
  }
  std::ifstream in(imagepath.c_str(), std::ios::in | std::ios::binary);
  if (!in) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << imagepath << " for reading"
        << "; dir: " << dir << "  fn: " << fn;
    throw lbann_exception(err.str());
  }
  in.read((char*)p, sz);
  if ((int)sz != in.gcount()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to read " << sz << " bytes from " << imagepath
        << " num bytes read: " << in.gcount();
    throw lbann_exception(err.str());
  }
  in.close();
}

void data_store_image::exchange_data() {
  double tm1 = get_time();
  std::stringstream err;

  //build map: proc -> global indices that proc needs for this epoch, and
  //                   which I own
  std::unordered_map<int, std::unordered_set<int>> proc_to_indices;
  for (size_t p=0; p<m_all_minibatch_indices.size(); p++) {
    for (auto idx : m_all_minibatch_indices[p]) {
      int index = (*m_shuffled_indices)[idx];
      if (m_my_datastore_indices.find(index) != m_my_datastore_indices.end()) {
        proc_to_indices[p].insert(index);
      }
    }
  }

  //start sends
  std::vector<std::vector<El::mpi::Request<unsigned char>>> send_req(m_np);
  for (int p=0; p<m_np; p++) {
    send_req[p].resize(proc_to_indices[p].size()*m_num_img_srcs);
    size_t jj = 0;
    for (auto idx : proc_to_indices[p]) {
      for (size_t k=0; k<m_num_img_srcs; k++) {
        int index = idx*m_num_img_srcs+k;
        if (m_file_sizes.find(index) == m_file_sizes.end()) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << " m_file_sizes.find(" << index << ") failed";
          throw lbann_exception(err.str());
        }
        int len = m_file_sizes[index];
        m_comm->nb_tagged_send<unsigned char>(
            m_data[index].data(), len, p, index, 
            send_req[p][jj++], m_comm->get_model_comm());

      }
    }
    if (jj != send_req[p].size()) throw lbann_exception("ERROR 1");
  } //start sends

  //build map: proc -> global indices that proc owns that I need
  proc_to_indices.clear();
  //note: datastore indices are global; no need to consult shuffled_indices
  for (auto idx : m_my_minibatch_indices_v) {
    int index = (*m_shuffled_indices)[idx];
    int owner = get_index_owner(index);
    proc_to_indices[owner].insert(index);
  }
  
  //start recvs
  m_my_minibatch_data.clear();
  std::vector<std::vector<El::mpi::Request<unsigned char>>> recv_req(m_np);
  for (auto t : proc_to_indices) {
    int owner = t.first;
    size_t jj = 0;
    const std::unordered_set<int> &s = t.second;
    recv_req[owner].resize(s.size()*m_num_img_srcs);
    for (auto idx : s) {
      for (size_t k=0; k<m_num_img_srcs; k++) {
        size_t index = idx*m_num_img_srcs+k;
        if (m_file_sizes.find(index) == m_file_sizes.end()) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << " m_file_sizes.find(" << index << ") failed"
              << " m_file_sizes.size(): " << m_file_sizes.size()
              << " m_my_minibatch_indices_v.size(): " << m_my_minibatch_indices_v.size();
        }
        size_t len = m_file_sizes[index];
        m_my_minibatch_data[index].resize(len);
        m_comm->nb_tagged_recv<unsigned char>(
            m_my_minibatch_data[index].data(), len, owner, 
            index, recv_req[owner][jj++], m_comm->get_model_comm());
      }
    }
  }

  //wait for sends to finish
  for (size_t i=0; i<send_req.size(); i++) {
    m_comm->wait_all<unsigned char>(send_req[i]);
  }

  //wait for recvs to finish
  for (size_t i=0; i<recv_req.size(); i++) {
    m_comm->wait_all<unsigned char>(recv_req[i]);
  }

  if (m_master) {
    std::cerr << "role: " << m_reader->get_role() 
              << "; time for exchange_data: " << get_time() - tm1 << "\n";
  }
}


void data_store_image::exchange_file_sizes(
  std::vector<int> &my_global_indices,
  std::vector<int> &my_num_bytes,
  int num_global_indices) {

  std::vector<int> rcv_counts(m_np);
  int nbytes = my_global_indices.size();
  m_comm->model_all_gather<int>(nbytes, rcv_counts);

  std::vector<int> disp(m_num_readers); 
  disp[0] = 0;
  for (int h=1; h<(int)m_num_readers; h++) {
    disp[h] = disp[h-1] + rcv_counts[h-1];
  }
  std::vector<int> all_global_indices(num_global_indices);
  std::vector<int> all_num_bytes(num_global_indices);

  m_comm->all_gather<int>(my_global_indices, all_global_indices, rcv_counts, disp, m_comm->get_world_comm());

  m_comm->all_gather<int>(my_num_bytes, all_num_bytes, rcv_counts, disp, m_comm->get_world_comm());

  for (size_t j=0; j<all_global_indices.size(); j++) {
    m_file_sizes[all_global_indices[j]] = all_num_bytes[j];
  }
}

size_t data_store_image::get_global_num_file_bytes() {
  size_t n = get_my_num_file_bytes();
  size_t g = 0;
  if (m_master) {
    g = m_comm->reduce(n, m_comm->get_world_comm());
  } else {
    m_comm->reduce(n, 0, m_comm->get_world_comm());
  }
  return g;
}

size_t data_store_image::get_my_num_file_bytes() {
  size_t count = 0;
  for (auto idx : m_my_datastore_indices) {
    for (size_t i=0; i<m_num_img_srcs; i++) {
      int index = idx*m_num_img_srcs + i;
      if (m_file_sizes.find(index) == m_file_sizes.end()) {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
            << " failed to find " << idx << " in m_file_sizes; count: " << count
            << " m_file_sizes.size(): " << m_file_sizes.size();
        throw lbann_exception(err.str());
      }  
      count += m_file_sizes[index];
    }
  }
  return count;
}

size_t data_store_image::get_available_memory() {
  std::ifstream in("/proc/meminfo");
  std::string line;
  size_t size;
  bool found = false;
  std::string name;
  std::string units;
  while (! in.eof()) {
    getline(in, line);
    std::stringstream s(line);
    s >> name >> size >> units;
    if (name.find("MemFree") != std::string::npos) {
      found = true;
      break;
    }
  }
  in.close();

  if (!found) {
    if (m_master) {
      std::cerr <<
        "\nWARNING: data_store_image::get_available_memory failed\n"
        "failed to find 'MemFree in /proc/meminfo\n"
        "therefore we cannot advise whether you have enough resources\n"
        "to contain all data files in memory\n"; 
    }
    return 0;
  }
  return size;
}


//note: this could be done on P_0 with no communication,
//      but it's a cheap operation, so I'm coding it the
//      easy way
void data_store_image::report_memory_constraints() {
  size_t count = get_my_num_file_bytes();

  std::vector<long long> counts(m_np);
  if (m_master) {
    m_comm->gather<long long>(count, counts.data(), m_comm->get_world_comm());
  } else {
    m_comm->gather<long long>(count, 0, m_comm->get_world_comm());
  }

  double global = get_global_num_file_bytes()/1000000;

  if (!m_master) { 
    return; 
  }

  /// determine the amount of memory required for files for all
  /// processors on this node
  double required = 0;
  for (int p=0; p<m_np; p++) {
    if (m_comm->is_rank_node_local(p, m_comm->get_world_comm())) {
      required += counts[p];
    }
  }
  required /= 1000000;

  double available = get_available_memory();
  if (available == 0) {
    std::cerr << required << " kB of memory are required for files on this node\n";
    return;
  }
  available /= 1000;

  double percent = required / available * 100.0;
  std::cerr << "\n"
            << "===============================================\n"
            << "Memory Constraints for: " << m_reader->get_role() << "\n" 
            << "Global data set size:               " << global << " MB\n"
            << "Required for data set on this node: " << required << " MB\n"
            << "Available memory on this node: "      << available << " MB\n"
            << "Required is " << percent << " % of Available\n"
            << "===============================================\n\n";

  double limit = 0.8;
  if (options::get()->has_float("mem_limit")) {
    limit = options::get()->get_float("mem_limit");
  }
  if (required > limit*available) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "You have insufficient memory to hold all required files;\n"
        << "required is > 80% of available\n"
        << "quitting now, so you don't waste your time\n";
  }
}



}  // namespace lbann
