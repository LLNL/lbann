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

#include "lbann/data_store/data_store_imagenet.hpp"
#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"

#include <sys/stat.h>

namespace lbann {

data_store_imagenet::~data_store_imagenet() {
  MPI_Win_free( &m_win );
}

void data_store_imagenet::setup() {
  double tm1 = get_time();
  if (m_rank == 0) {
    std::cout << "starting data_store_imagenet::setup() for data reader with role: " << m_reader->get_role() << std::endl;
  }

  generic_data_store::setup();

  //optionally run some tests at the end of setup()
  bool run_tests = false;
  if (options::get()->has_bool("test_data_store") && options::get()->get_bool("test_data_store")) {
    run_tests = true;
  }
  
  //@todo needs to be designed and implemented!
  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
    m_buffers.resize( omp_get_max_threads() );
  } 
  
  else {
    //sanity check
    imagenet_reader *reader = dynamic_cast<imagenet_reader*>(m_reader);
    if (reader == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "dynamic_cast<imagenet_reader*>(m_reader) failed";
      throw lbann_exception(err.str());
    }

    // fill in global index -> owning processor
    for (size_t j=0; j<m_num_global_indices; j++) {
      int owner = j % m_num_readers;
      m_owner_mapping[(*m_shuffled_indices)[j]] = owner;
    }

    m_my_data.resize(m_my_minibatch_indices.size());

    get_file_sizes();
    allocate_memory();
    read_files();
    MPI_Win_create((void*)&m_data[0], m_data.size(), 1, MPI_INFO_NULL, m_comm->get_model_comm().comm, &m_win);
    exchange_data();
    if (run_tests) {
      test_file_sizes();
      test_data();
    }  

    MPI_Barrier(m_comm->get_model_comm().comm);
    double tm2 = get_time();
    if (m_rank == 0) {
      std::cout << "data_store setup time: " << tm2 - tm1 << std::endl;
    }
  }
}

void data_store_imagenet::get_data_buf(int data_id, std::vector<unsigned char> *&buf, int tid) {
  if (m_my_data_hash.find(data_id) == m_my_data_hash.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find data_id: " << data_id << " in m_my_data_hash";
    throw lbann_exception(err.str());
  }
  int index = m_my_data_hash[data_id];
  buf = &m_my_data[index];
}

void data_store_imagenet::get_data_buf(std::string dir, std::string filename, std::vector<unsigned char> *&buf, int tid) {
  static int idx = 0;
  std::vector<unsigned char> &b = m_buffers[tid];
  std::string imagepath = dir + filename;
  if (m_master && idx == 0) {
    std::cout << "data_store_imagenet: READING: " << imagepath << std::endl << std::endl;
    ++idx;
  }  
  std::ifstream in(imagepath.c_str(), std::ios::in | std::ios::binary);
  if (! in.good()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << imagepath << " for reading";
    throw lbann_exception(err.str());
  }
  in.seekg(0, std::ios::end);
  size_t sz = in.tellg();
  in.seekg(0, std::ios::beg);
  b.resize(sz);
  in.read((char*)&b[0], sz*sizeof(unsigned char));
  in.close();
  buf = &m_buffers[tid];
}

size_t get_file_size(std::string dir, std::string fn) {
  std::string imagepath = dir + fn;
  struct stat st;
  if (stat(imagepath.c_str(), &st) != 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "stat failed for dir: " << dir
        << " and fn: " << fn;
    throw lbann_exception(err.str());
  }
  return st.st_size;   
}

struct Triple {
   int global_index;
   int num_bytes;
   size_t offset;
};

void data_store_imagenet::get_file_sizes() {
  std::vector<int> num_images(m_num_readers, 0);
  for (size_t j=0; j<m_num_global_indices; j++) {
    num_images[j % m_num_readers] += 1;
  }
  //num_images[j] contains the number of image files owned by P_j

  //get the number of bytes for files owned by this processor;
  //my_file_sizes[j] is a global index (wrt m_reader->get_image_list()),
  //and my_file_sizes[j+1] is the corresponding file size
  imagenet_reader *reader = dynamic_cast<imagenet_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  int my_num_images = num_images[m_rank];
  std::string file_dir = m_reader->get_file_dir();

  //construct a vector or Triples 
  std::vector<Triple> my_file_sizes(my_num_images);
  size_t cur_offset = 0;
  for (size_t j=0; j<m_my_datastore_indices.size(); j++) {
    if (j >= image_list.size()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " << " j: " << j 
        << " >= image_list.size() [" << image_list.size() << "]"; 
      throw lbann_exception(err.str());
    }
    size_t index = m_my_datastore_indices[j];
    my_file_sizes[j].global_index = index;
    my_file_sizes[j].num_bytes = get_file_size(file_dir, image_list[index].first);
    my_file_sizes[j].offset = cur_offset;
    cur_offset += my_file_sizes[j].num_bytes;
    if (my_file_sizes[j].num_bytes ==0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " << " j: " << j 
        << " file size is 0 (" << file_dir << "/" + image_list[index].first;
      throw lbann_exception(err.str());
    }
  }

  //exchange files sizes
  std::vector<Triple> global_file_sizes(m_num_global_indices);
  std::vector<int> disp(m_num_readers); 
  disp[0] = 0;
  for (int h=1; h<(int)m_num_readers; h++) {
    disp[h] = disp[h-1] + num_images[h-1]*sizeof(Triple);
  }

  for (size_t j=0; j<num_images.size(); j++) {
    num_images[j] *= sizeof(Triple);
  }

  //m_comm->model_gatherv(&my_file_sizes[0], my_file_sizes.size(), 
   //                     &global_file_sizes[0], &num_images[0], &disp[0]);
  MPI_Allgatherv(&my_file_sizes[0], my_file_sizes.size()*sizeof(Triple), MPI_BYTE,
                        &global_file_sizes[0], &num_images[0], &disp[0], MPI_BYTE, m_comm->get_model_comm().comm);


  for (auto t : global_file_sizes) {
    m_file_sizes[t.global_index] = t.num_bytes;
    m_offsets[t.global_index] = t.offset;
    if (t.num_bytes <= 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " 
        << "num_bytes  <= 0 (" << t.num_bytes << ")";
      throw lbann_exception(err.str());
    }
  }
}

void data_store_imagenet::allocate_memory() {
  size_t m = 0;
  for (auto t : m_my_datastore_indices) {
    m += m_file_sizes[t];
  }    
  m_data.resize(m);
}


void data_store_imagenet::load_file(const std::string &dir, const std::string &fn, unsigned char *p, size_t sz) {
  std::string imagepath = dir + fn;
  std::ifstream in(imagepath.c_str(), std::ios::in | std::ios::binary);
  if (!in) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for reading";
    throw lbann_exception(err.str());
  }
  in.read((char*)p, sz);
  if (!in) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to read " << sz << " bytes from " << fn
        << " num bytes read: " << in.gcount();
    throw lbann_exception(err.str());
  }
  in.close();
}

void data_store_imagenet::read_files() {
  imagenet_reader *reader = dynamic_cast<imagenet_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  std::string file_dir = m_reader->get_file_dir();
  for (auto t : m_my_datastore_indices) {
    load_file(file_dir, image_list[t].first, &m_data[m_offsets[t]], m_file_sizes[t]);
  }
}

void data_store_imagenet::exchange_data() {
  double tm1 = get_time();
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  m_my_data_hash.clear();
  for (size_t j = 0; j<m_my_minibatch_indices.size(); j++) {
    int idx = (*m_shuffled_indices)[m_my_minibatch_indices[j]];
    int offset = m_offsets[idx];
    int file_len = m_file_sizes[idx];
    if (file_len <= 0) {
      throw lbann_exception(
        std::string{} + __FILE__ + " :: " + std::to_string(__LINE__) + " :: "
        + " file_len: " + std::to_string(file_len) + " (is <= 0)");
    }    
    if (file_len > 100000000) {
      throw lbann_exception(
        std::string{} + __FILE__ + " :: " + std::to_string(__LINE__) + " :: "
        + " file_len: " + std::to_string(file_len) + " (is > 100000000)");
    }    
    if (j >= m_my_data.size()) {
      throw lbann_exception(
        std::string{} + __FILE__ + " :: " + std::to_string(__LINE__) + " :: "
        + " j: " + std::to_string(j) + " is >= m_my_data.size(): "
        + std::to_string(m_my_data.size()));
    }
    m_my_data[j].resize(file_len);
    m_my_data_hash[idx] = j;
    int owner = m_owner_mapping[idx];
    MPI_Get(&m_my_data[j][0], file_len, MPI_BYTE,
            owner, offset, file_len, MPI_BYTE, m_win);
  }
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  double tm2 = get_time();
  if (m_rank == 0) {
    std::cout << "data_store_imagenet::exchange_data() time: " << tm2 - tm1 << std::endl;
  }
}

void data_store_imagenet::test_data() {
  std::cerr << m_rank << " :: STARTING data_store_imagenet::test_data()\n";
  imagenet_reader *reader = dynamic_cast<imagenet_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  std::vector<unsigned char> b;
  std::string file_dir = m_reader->get_file_dir();
  for (auto t : m_my_minibatch_indices) {
    int idx = (*m_shuffled_indices)[t];
    //read directly from file
    std::string imagepath = file_dir + image_list[(*m_shuffled_indices)[t]].first;
    std::ifstream in(imagepath.c_str(), std::ios::in | std::ios::binary);
    if (! in.good()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "failed to open " << imagepath << " for reading";
      throw lbann_exception(err.str());
    }

    in.seekg(0, std::ios::end);
    size_t sz = in.tellg();
    in.seekg(0, std::ios::beg);
    b.resize(sz);
    in.read((char*)&b[0], sz);
    in.close();

    //compare to m_my_data
    if (m_my_data_hash.find(idx) == m_my_data_hash.end()) {
      std::cerr << m_rank << " :: my index: " << idx << " not found in m_my_data_hash\n";
      throw lbann_exception("ERROR #1");
    }
    size_t index = m_my_data_hash[idx];
    std::vector<unsigned char> &v = m_my_data[index];
    if (b != v) {
      std::stringstream err;
      err << "ERROR #2; b.size: " << b.size() << "  v.size: " << v.size() << std::endl;
      throw lbann_exception(err.str());
    } 
  }
  std::cerr << "rank: " << m_rank << " role: " << m_reader->get_role() << " :: data_store_imagenet::test_data: PASSES!\n";
}

void data_store_imagenet::test_file_sizes() {
  std::cerr << m_rank << " :: STARTING data_store_imagenet::test_file_sizes()\n";
  imagenet_reader *reader = dynamic_cast<imagenet_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  std::string file_dir = m_reader->get_file_dir();
  for (auto t : m_file_sizes) {
    size_t len = get_file_size(file_dir, image_list[t.first].first);
    if (t.second != len || len == 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "m_file_sizes[" << t.first << "] = " << t.second
          << " but actual size appears to be " << len;
      throw lbann_exception(err.str());
    }
  }
  std::cerr << "rank: " << m_rank << " role: " << m_reader->get_role() << " :: data_store_imagenet::test_file_sizes: PASSES!\n";
}


}  // namespace lbann
