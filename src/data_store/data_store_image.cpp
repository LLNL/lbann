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
#include "lbann/utils/timer.hpp"

namespace lbann {

data_store_image::~data_store_image() {
  MPI_Win_free( &m_win );
}

void data_store_image::setup() {

  if (m_master) std::cerr << "starting data_store_image::setup(); calling generic_data_store::setup()\n";
  generic_data_store::setup();

  //@todo needs to be designed and implemented!
  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
    m_buffers.resize( omp_get_max_threads() );
  } 
  
  else {
    // get list of all indices used in all calls to 
    // generic_data_reader::fetch_data
    size_t s2 = 0;
    for (auto t1 : (*m_minibatch_indices)) {
      s2 += t1.size();
    }
    m_my_minibatch_indices.reserve(s2);
    for (auto t1 : (*m_minibatch_indices)) {
      for (auto t2 : t1) {
        m_my_minibatch_indices.push_back(t2);
      }
    }
    m_my_minibatch_data.resize(m_my_minibatch_indices.size());
    if (m_master) std::cerr << "num minibatch indices: " << m_my_minibatch_indices.size() << "\n";

    if (m_master) std::cerr << "calling get_my_datastore_indices\n";
    get_my_datastore_indices();

    if (m_master) std::cerr << "calling compute_my_filenames\n";
    compute_my_filenames();

    if (m_master) std::cerr << "calling compute_owner_mapping\n";
    compute_owner_mapping();

    if (m_master) std::cerr << "calling compute_num_images\n";
    compute_num_images();

    if (m_master) std::cerr << "calling get_file_sizes\n";
    get_file_sizes();

    if (m_master) std::cerr << "calling allocate_memory\n";
    allocate_memory();

    if (m_master) std::cerr << "calling read_files\n";
    read_files();
    MPI_Win_create((void*)&m_data[0], m_data.size(), 1, MPI_INFO_NULL, m_comm->get_model_comm().comm, &m_win);
    exchange_data();
  }
}

void data_store_image::get_data_buf(int data_id, std::vector<unsigned char> *&buf, int tid, int multi_idx) {
  int index = data_id*m_num_multi+multi_idx;
  if (m_my_data_hash.find(index) == m_my_data_hash.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find data_id: " << data_id << " in m_my_data_hash";
    throw lbann_exception(err.str());
  }
  int idx = m_my_data_hash[index]; 
  buf = &m_my_minibatch_data[idx];
}

void data_store_image::allocate_memory() {
  size_t m = 0;
  for (auto t : m_my_shuffled_indices) {
    m += m_file_sizes[t];
  }    
  m_data.resize(m);
}

void data_store_image::load_file(const std::string &dir, const std::string &fn, unsigned char *p, size_t sz) {
  std::string imagepath = dir + fn;
  std::ifstream in(imagepath.c_str(), std::ios::in | std::ios::binary);
  if (!in) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for reading";
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

void data_store_image::read_files() {
  for (size_t j=0; j<m_my_files.size(); j++) {
    size_t shuffled_index = m_my_shuffled_indices[j];
    size_t offset = m_offsets[shuffled_index];
    load_file(m_dir, m_my_files[j], &m_data[offset], m_file_sizes[shuffled_index]);
  }
}

void data_store_image::exchange_data() {
  double tm1 = get_time();
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  m_my_data_hash.clear();

  for (size_t j = 0; j<m_my_minibatch_indices.size(); j++) {
    size_t idx = (*m_shuffled_indices)[m_my_minibatch_indices[j]];
    int offset = m_offsets[idx];
    if (m_file_sizes.find(idx) == m_file_sizes.end()) {
      std::stringstream err;
      err << __FILE__ << " :: " << __LINE__ << " :: "
          << idx << " is not a key in m_file_sizes";
      throw lbann_exception(err.str());
    }
    int file_len = m_file_sizes[idx];
    if (file_len <= 0) {
      std::stringstream err;
      err << __FILE__ << " :: " << __LINE__ << " :: "
          << " j: " << j << " of " <<  m_my_minibatch_indices.size() 
          << "; file_len: " << file_len << " (is <= 0)";
      throw lbann_exception(err.str());
    }    
    if (file_len > 100000000) {
      std::stringstream err;
      err << __FILE__ << " :: " << __LINE__ << " :: "
          << "j: " << j << " file_len: " << file_len << " (is > 100000000)";
      throw lbann_exception(err.str());
    }    
    if (j >= m_my_minibatch_data.size()) {
      std::stringstream err;
      err << __FILE__  << " :: " << __LINE__ << " :: "
        << " j: " << j << " is >= m_my_data.size(): "
        << m_my_minibatch_data.size();
      throw lbann_exception(err.str());
    }
    m_my_minibatch_data[j].resize(file_len);
    m_my_data_hash[idx] = j;
    int owner = m_owner_mapping[idx];
    MPI_Get(&m_my_minibatch_data[j][0], file_len, MPI_BYTE,
            owner, offset, file_len, MPI_BYTE, m_win);
  }
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  double tm2 = get_time();
  if (m_rank == 0) {
    std::cout << "data_store_image::exchange_data() time: " << tm2 - tm1 << std::endl;
  }
}


}  // namespace lbann
