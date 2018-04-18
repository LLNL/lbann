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
    if (m_master) std::cerr << "calling get_minibatch_index_vector\n";
    get_minibatch_index_vector();

    if (m_master) std::cerr << "calling exchange_mb_indices\n";
    exchange_mb_indices();

    if (m_master) std::cerr << "calling get_my_datastore_indices\n";
    get_my_datastore_indices();

    if (m_master) std::cerr << "calling get_file_sizes\n";
    double tma = get_time();
    get_file_sizes();
    if (m_master) std::cerr << "get_file_sizes time: " << get_time() - tma << "\n";

    if (m_master) std::cerr << "calling allocate_memory\n";
    allocate_memory();

    if (m_master) std::cerr << "calling read_files\n";
    tma = get_time();
    read_files();
    if (m_master) std::cerr << "read_files time: " << get_time() - tma << "\n";

    if (m_master) std::cerr << "calling exchange_data\n";
    exchange_data();

    if (m_extended_testing) {
      if (m_master) std::cerr << "calling extended_testing\n";
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

void data_store_image::allocate_memory() {
  size_t m = 0;
  for (auto t : m_my_datastore_indices) {
    for (size_t i=0; i<m_num_img_srcs; i++) {
      m += m_file_sizes[t*m_num_img_srcs+i];
    }    
  }    
  m_data.resize(m);
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
  if (m_master) std::cerr << "starting exchange_data\n";
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
  std::vector<std::vector<MPI_Request>> send_req(m_np);
  std::vector<std::vector<MPI_Status>> send_status(m_np);
  for (int p=0; p<m_np; p++) {
    send_req[p].resize(proc_to_indices[p].size()*m_num_img_srcs);
    send_status[p].resize(proc_to_indices[p].size()*m_num_img_srcs);
    size_t jj = 0;
    for (auto idx : proc_to_indices[p]) {
      for (size_t k=0; k<m_num_img_srcs; k++) {
        int index = idx*m_num_img_srcs+k;
        if (m_file_sizes.find(index) == m_file_sizes.end()) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << " m_file_sizes.find(" << index << ") failed";
          throw lbann_exception(err.str());
        }
        if (m_offsets.find(index) == m_offsets.end()) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << " m_offets.find(" << index << ") failed";
          throw lbann_exception(err.str());
        }
        int len = m_file_sizes[index];
        size_t offset = m_offsets[index];
        MPI_Isend(m_data.data()+offset, len, MPI_BYTE, p, index, m_mpi_comm, &(send_req[p][jj++]));
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
  std::vector<std::vector<MPI_Request>> recv_req(m_np);
  std::vector<std::vector<MPI_Status>> recv_status(m_np);
  for (auto t : proc_to_indices) {
    int owner = t.first;
    size_t jj = 0;
    const std::unordered_set<int> &s = t.second;
    recv_req[owner].resize(s.size()*m_num_img_srcs);
    recv_status[owner].resize(s.size()*m_num_img_srcs);
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
        MPI_Irecv(m_my_minibatch_data[index].data(), len, MPI_BYTE, owner, index, m_mpi_comm, &(recv_req[owner][jj++]));
      }
    }
  }

  //wait for sends to finish
  for (size_t i=0; i<send_req.size(); i++) {
    MPI_Waitall(send_req[i].size(), send_req[i].data(), send_status[i].data());
  }

  //wait for recvs to finish
  for (size_t i=0; i<recv_req.size(); i++) {
    MPI_Waitall(recv_req[i].size(), recv_req[i].data(), recv_status[i].data());
  }
}


void data_store_image::exchange_file_sizes(std::vector<Triple> &my_file_sizes, int num_global_indices) {
  std::vector<int> num_bytes(m_np);
  int bytes = my_file_sizes.size()*sizeof(Triple);
  MPI_Allgather(&bytes, 1, MPI_INT,
                 num_bytes.data(), 1, MPI_INT,
                 m_mpi_comm);

  std::vector<int> disp(m_num_readers); 
  disp[0] = 0;
  for (int h=1; h<(int)m_num_readers; h++) {
    disp[h] = disp[h-1] + num_bytes[h-1];
  }
  std::vector<Triple> global_file_sizes(num_global_indices);

  //@todo: couldn't get m_comm->model_gatherv to work
  //m_comm->model_gatherv(&my_file_sizes[0], my_file_sizes.size(), 
   //                     &global_file_sizes[0], &num_images[0], &disp[0]);
  MPI_Allgatherv(my_file_sizes.data(), my_file_sizes.size()*sizeof(Triple), MPI_BYTE,
                 global_file_sizes.data(), num_bytes.data(), disp.data(), MPI_BYTE,
                 m_mpi_comm);

  size_t j = 0;
  for (auto t : global_file_sizes) {
    m_file_sizes[t.global_index] = t.num_bytes;
    m_offsets[t.global_index] = t.offset;
    if (t.num_bytes <= 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " 
        << "num_bytes  <= 0 (" << t.num_bytes << ")"
        << " for # " << j << " of " << global_file_sizes.size();
      throw lbann_exception(err.str());
    }
    //m_owner_mapping[t.global_index] = t.rank;
    ++j;
  }
}

}  // namespace lbann
