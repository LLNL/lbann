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

void data_store_image::setup(bool test_dynamic_cast, bool run_tests) {

  if (m_master) std::cerr << "starting data_store_image::setup(); calling generic_data_store::setup()\n";
  generic_data_store::setup();

  //@todo needs to be designed and implemented!
  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
    //m_buffers.resize( omp_get_max_threads() );
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
    m_my_minibatch_data.resize(m_my_minibatch_indices.size()*m_num_img_srcs);
    if (m_master) std::cerr << "num minibatch indices: " << m_my_minibatch_indices.size() << "\n";

    if (m_master) std::cerr << "calling get_my_datastore_indices\n";
    get_my_datastore_indices();

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

//@todo probably don't need tid
void data_store_image::get_data_buf(int data_id, std::vector<unsigned char> *&buf, int tid, int multi_idx) {
  int index = data_id * m_num_img_srcs + multi_idx;
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
  for (auto t : m_my_global_indices) {
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
        << "; dir: " << dir;
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
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  m_my_data_hash.clear();

  size_t jj = 0;
  for (size_t j = 0; j<m_my_minibatch_indices.size(); j++) {
    size_t base_index = (*m_shuffled_indices)[m_my_minibatch_indices[j]];
    for (size_t i=0; i<m_num_img_srcs; i++) {
      size_t idx = base_index*m_num_img_srcs+i;

      if (m_file_sizes.find(idx) == m_file_sizes.end()) {
        std::stringstream err;
        err << __FILE__ << " :: " << __LINE__ << " :: "
            << idx << " is not a key in m_file_sizes";
        throw lbann_exception(err.str());
      }

      if (jj >= m_my_minibatch_data.size()) {
        std::stringstream err;
        err << __FILE__  << " :: " << __LINE__ << " :: "
          << " jj: " << jj << " is >= m_my_data.size(): "
          << m_my_minibatch_data.size();
        throw lbann_exception(err.str());
      }

      if (m_owner_mapping.find(idx) == m_owner_mapping.end()) {
        std::stringstream err;
        err << __FILE__  << " :: " << __LINE__ << " :: "
          << " m_owner_mapping.find(idx) failed";
        throw lbann_exception(err.str());
      }

      if (m_offsets.find(idx) == m_offsets.end()) {
        std::stringstream err;
        err << __FILE__  << " :: " << __LINE__ << " :: "
          << " m_offsets.find(idx) failed";
        throw lbann_exception(err.str());
      }

      int file_len = m_file_sizes[idx];
      m_my_data_hash[idx] = jj;
      int owner = m_owner_mapping[idx];
      int offset = m_offsets[idx];

      m_my_minibatch_data[jj].resize(file_len);
      MPI_Get(&m_my_minibatch_data[jj][0], file_len, MPI_BYTE,
              owner, offset, file_len, MPI_BYTE, m_win);
      ++jj;
    }
  }
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  double tm2 = get_time();
  if (m_rank == 0) {
    std::cout << "data_store_image::exchange_data() time: " << tm2 - tm1 << std::endl;
  }
}

void data_store_image::exchange_file_sizes(std::vector<Triple> &my_file_sizes, int num_global_indices) {

  //exchange files sizes
  std::vector<Triple> global_file_sizes(num_global_indices);
  std::vector<int> disp(m_num_readers); 
  disp[0] = 0;
  for (int h=1; h<(int)m_num_readers; h++) {
    disp[h] = disp[h-1] + m_num_images[h-1]*sizeof(Triple)*m_num_img_srcs;
  }

  for (size_t j=0; j<m_num_images.size(); j++) {
    m_num_images[j] *= sizeof(Triple)*m_num_img_srcs;
  }

  //m_comm->model_gatherv(&my_file_sizes[0], my_file_sizes.size(), 
   //                     &global_file_sizes[0], &num_images[0], &disp[0]);
  MPI_Allgatherv(&my_file_sizes[0], my_file_sizes.size()*sizeof(Triple), MPI_BYTE,
                 &global_file_sizes[0], &m_num_images[0], &disp[0], MPI_BYTE,
                 m_comm->get_model_comm().comm);

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
    m_owner_mapping[t.global_index] = t.rank;
    ++j;
  }
}

}  // namespace lbann
